"""
模型读取模块 - 加载和读取.gndc模型文件
"""
import torch
import torch.nn as nn
import numpy as np
import json
import struct
import os
from datetime import datetime
from typing import Optional, Tuple

from .NDCCCDCNetwork import GeoCCDCNetwork

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    import zlib


class CCDCWrapper(nn.Module):
    """CCDC模型包装器"""
    
    def __init__(self, ccdc_model):
        super().__init__()
        self.model = ccdc_model
    
    def forward(self, x_spatial, t_time):
        out = self.model(x_spatial, t_time)
        if isinstance(out, tuple):
            pred_mean = out[0]
            log_var = out[1]
            # 限制防爆范围
            log_var = torch.clamp(log_var, min=-7.0, max=5.0)
            # 计算标准差: std = exp(0.5 * log_var)
            pred_std = torch.exp(0.5 * log_var)
            return pred_mean, pred_std
        return out, None


class NDCReader:
    """NOMAD模型读取器"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset_meta = {}
        self.model_config = {}
        self.empirical_null_profile = None
        self.residuals = None
        self.spatial_std = None  # 🚀 新增：保存每个像素在时间轴上的真实标准差 (RMSE)
    
    def load(self, gndc_path: str):
        """加载并解析.gndc模型文件（支持动态量化格式）"""
        print(f"[NDCReader] Loading model from: {gndc_path}")
        
        with open(gndc_path, 'rb') as f:
            if HAS_ZSTD:
                dctx = zstd.ZstdDecompressor()
            else:
                dctx = None
            
            raw_h_len = f.read(4)
            if len(raw_h_len) < 4:
                raise ValueError("File too short or corrupt (Header length).")
            h_len = struct.unpack("I", raw_h_len)[0]
            header = json.loads(f.read(h_len).decode('utf-8'))
            self.empirical_null_profile = header.get('empirical_null_profile')
            
            self.dataset_meta = header['dataset_meta']
            model_cfg = header.get('model_config', {})
            self.model_config = model_cfg
            self.dataset_meta['ablation_mode'] = model_cfg.get('ablation_mode', 'full')
            file_type = header.get('file_type', 'legacy')
            
            q_info = header.get('quantization_info', {})
            mode = q_info.get('mode', 'int16')
            scale_w = q_info.get('scale', 1.0)
            is_int = q_info.get('is_int', True)
            
            raw_w_len = f.read(4)
            if len(raw_w_len) < 4:
                raise ValueError("File too short (Weights length).")
            w_len = struct.unpack("I", raw_w_len)[0]
            w_compressed = f.read(w_len)
            
            if HAS_ZSTD:
                w_bytes = dctx.decompress(w_compressed)
            else:
                w_bytes = zlib.decompress(w_compressed)
            
            mode_to_np = {
                'float32': np.float32,
                'float16': np.float16,
                'int16': np.int16,
                'int8': np.int8
            }
            target_dtype = mode_to_np.get(mode, np.int16)
            
            w_float = np.frombuffer(w_bytes, dtype=target_dtype).astype(np.float32)
            if is_int:
                w_float /= scale_w
                
            n_bands = self.dataset_meta['shape'][3]
            t_start = datetime.fromisoformat(str(self.dataset_meta['global_start_date']).replace("Z", ""))
            t_end = datetime.fromisoformat(str(self.dataset_meta['global_end_date']).replace("Z", ""))
            years = (t_end - t_start).days / 365.25
            
            # 🚀 同步加入 n_hidden_layers，防止模型加载时结构与训练时不匹配
            self.model = CCDCWrapper(GeoCCDCNetwork(
                n_bands=n_bands,
                time_span_years=model_cfg.get('time_span_years', years),
                xy_hash_size=model_cfg.get('xy_hash_size', 19),
                n_levels=model_cfg.get('n_levels', 16),
                base_resolution=model_cfg.get('base_resolution', 32),
                per_level_scale=model_cfg.get('per_level_scale', 1.18),
                features_per_level=model_cfg.get('features_per_level', 2),
                decoder_hidden=model_cfg.get('decoder_hidden', 128),
                n_harmonics=model_cfg.get('n_harmonics', 3),
                n_hidden_layers=model_cfg.get('n_hidden_layers', 2),
                ablation_mode=model_cfg.get('ablation_mode', 'full'),
                anchor_dynamic_state=model_cfg.get('anchor_dynamic_state', False),
                time_varying_uncertainty=model_cfg.get(
                    'time_varying_uncertainty', False
                ),
                horizon_uncertainty=model_cfg.get('horizon_uncertainty', False),
            ).to(self.device))

            expected_numel = sum(param.numel() for param in self.model.parameters())
            if w_float.size != expected_numel:
                raise ValueError(
                    "Model weight count does not match the architecture in the GNDC header: "
                    f"file={w_float.size}, expected={expected_numel}, "
                    f"ablation_mode={model_cfg.get('ablation_mode', 'full')}."
                )
            
            ptr = 0
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    numel = param.numel()
                    param.data.copy_(
                        torch.from_numpy(w_float[ptr: ptr + numel]).view_as(param).to(self.device)
                    )
                    ptr += numel
            
            if file_type != "weights_only":
                raw_r_marker = f.read(4)
                if raw_r_marker and len(raw_r_marker) == 4:
                    r_len = struct.unpack("I", raw_r_marker)[0]
                    if r_len > 0 and header.get('has_residuals'):
                        r_compressed = f.read(r_len)
                        r_bytes = dctx.decompress(r_compressed) if HAS_ZSTD else zlib.decompress(r_compressed)
                        
                        T, H, W, C = self.dataset_meta['shape']
                        
                        # 动态获取残差类型和缩放因子（兼容老版本默认 int16）
                        res_info = header.get('residual_info', {'dtype': 'int16', 'scale': 10000.0})
                        r_dtype_str = res_info.get('dtype', 'int16')
                        r_scale = res_info.get('scale', 10000.0)
                        
                        dtype_map = {'int16': np.int16, 'int8': np.int8, 'float16': np.float16, 'float32': np.float32}
                        target_r_dtype = dtype_map.get(r_dtype_str, np.int16)
                        
                        # 移除 offset=16，使用动态类型读取并缩放
                        res_raw = np.frombuffer(r_bytes, dtype=target_r_dtype)
                        self.residuals = res_raw.astype(np.float32).reshape(T, H, W, C) / r_scale
                        
                        # 🚀 [新增]: 秒算全局的真实 RMSE，供其他评估脚本使用
                        self.spatial_std = np.std(self.residuals, axis=0)
        
        self.model.eval()
        print(f"[NDCReader] SUCCESS: Loaded weights in {mode} format.")
    
    def predict(self, coords_spatial, t_time):
        """推理预测"""
        with torch.no_grad():
            # 🚀 确保 return 存在
            return self.model(coords_spatial.to(self.device), t_time.to(self.device))
    
    def reconstruct_all(self, model_gndc: str, residual_gndc: str) -> np.ndarray:
        """通过两个GNDC文件重建原始遥感影像"""
        self.load(model_gndc)
        T, H, W, C = self.dataset_meta['shape']
        use_log = self.dataset_meta.get('use_log', False)
        
        print(f"[Reconstruct] Reading residuals from {os.path.basename(residual_gndc)}...")
        with open(residual_gndc, 'rb') as f:
            if HAS_ZSTD:
                dctx = zstd.ZstdDecompressor()
            else:
                dctx = None
                
            # 1. 正常读取头文件并解析
            h_len = struct.unpack("I", f.read(4))[0]
            res_header_bytes = f.read(h_len)
            res_header = json.loads(res_header_bytes.decode('utf-8')) # 解析独立残差头
            
            # 2. 读取残差数据体并解压
            r_len_bytes = f.read(4)
            if not r_len_bytes:
                raise ValueError("Residual file incomplete.")
            r_len = struct.unpack("I", r_len_bytes)[0]
            r_bytes = f.read(r_len)
            
            if HAS_ZSTD:
                r_bytes = dctx.decompress(r_bytes)
            else:
                r_bytes = zlib.decompress(r_bytes)
            
            # 3. 动态解码
            res_info = res_header.get('residual_info', {'dtype': 'int16', 'scale': 10000.0})
            r_dtype_str = res_info.get('dtype', 'int16')
            r_scale = res_info.get('scale', 10000.0)
            target_r_dtype = {'int16': np.int16, 'int8': np.int8, 'float16': np.float16, 'float32': np.float32}.get(r_dtype_str, np.int16)
            
            res_data = np.frombuffer(r_bytes, dtype=target_r_dtype) # 移除 offset
            residuals = res_data.reshape(T, H, W, C).astype(np.float32) / r_scale
        
        print(f"[Reconstruct] Running model inference...")
        reconstructed_volume = np.zeros((T, H, W, C), dtype=np.float32)
        
        offset_x, offset_y = 0.5 / W, 0.5 / H
        gy, gx = torch.meshgrid(
            torch.linspace(offset_y, 1.0 - offset_y, H, device=self.device),
            torch.linspace(offset_x, 1.0 - offset_x, W, device=self.device),
            indexing='ij'
        )
        coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        t_norms = torch.tensor(self.dataset_meta['t_norms'], device=self.device)
        
        for t_idx in range(T):
            t_val = t_norms[t_idx].view(1, 1).expand(H * W, 1)
            pred_mean, _ = self.predict(coords, t_val)
            
            frame_pred = pred_mean.view(H, W, C).cpu().numpy()
            frame_final = frame_pred + residuals[t_idx]
            
            if use_log:
                frame_final = np.expm1(frame_final)
            
            reconstructed_volume[t_idx] = frame_final
        
        print(f"[Reconstruct] Successfully restored {T} frames.")
        return reconstructed_volume


def load_model(model_path: str, device: Optional[str] = None) -> NDCReader:
    """便捷函数：加载模型"""
    reader = NDCReader(device)
    reader.load(model_path)
    return reader
