"""
压缩引擎模块 - 训练神经网络模型 (整合自适应超大批次流式训练)
"""
import os
import torch
import torch.nn as nn
import numpy as np
import struct
import json
import gc
import inspect
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional
from tqdm import tqdm

from .NDCCCDCNetwork import GeoCCDCNetwork
from .NDCUtils import fast_compress, fast_decompress
from .NDCDatasetLoader import DatasetLoader


class HeteroscedasticLoss(nn.Module):
    """异方差损失函数 (支持时间加权)"""
    def __init__(self, ablation_mode: str = "full"):
        super().__init__()
        self.ablation_mode = ablation_mode
    
    def forward(self, pred_mean, log_var, target, weights=None):
        if self.ablation_mode == "no_prob":
            log_var = torch.zeros_like(pred_mean)
        log_var = torch.clamp(log_var, min=-7.0, max=5.0)
        precision = torch.exp(-log_var)
        diff = (pred_mean - target) ** 2
        
        # 逐点计算基础 loss
        loss = 0.5 * precision * diff + 0.5 * log_var
        
        # 应用时间权重
        if weights is not None:
            loss = loss * weights
            
        return torch.mean(loss)


class NDCCompressor:
    """NOMAD压缩引擎 - 训练神经网络模型"""
    
    def __init__(
        self,
        device: str = "cuda",
        train_dtype: str = "float16",
        global_start_date: Optional[datetime] = None,
        global_end_date: Optional[datetime] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_amp = (train_dtype == 'float16')
        self.meta_info = {}
        self.model = None
        self.model_type = "KPlanes"
        self.global_start_date = global_start_date
        self.global_end_date = global_end_date
        resolved_cache = Path(
            cache_dir
            or os.environ.get("SINC_CACHE_DIR", "")
            or (Path.cwd() / ".ndc_cache")
        ).expanduser().absolute()
        self.loader = DatasetLoader(
            dtype=np.float16 if self.use_amp else np.float32,
            cache_dir=str(resolved_cache),
        )
        self.config = {}
    
    def load_dataset_folder(
        self,
        folder_path: str,
        expected_bands: int = 7,
        scale_factor: float = 0.0001,
        valid_range: tuple = (0, 1.2),
        p99_removal: bool = False,
        temporal_bin_days: Optional[int] = None
    ):
        """加载数据集"""
        print(f"[NDC] Loading dataset from: {folder_path}")
        self.volume, self.nodata_mask_vol, loaded_meta = self.loader.load_folder(
            folder_path=folder_path,
            expected_bands=expected_bands,
            scale_factor=scale_factor,
            valid_range=valid_range,
            p99_removal=p99_removal,
            temporal_bin_days=temporal_bin_days,
            global_start_date=self.global_start_date,
            global_end_date=self.global_end_date
        )
        self.meta_info.update(loaded_meta)
        return self.volume, self.nodata_mask_vol
    
    # [修改点 1]：引入 valid_threshold 参数，替换硬编码
    def prepare_training_data_optimized(self, use_log: bool = False, valid_threshold: float = 0.001):
        """准备训练数据"""
        print(f"[Optimized] Preparing Coordinate Grid & Filtering Noise...")
        N, H, W, C = self.volume.shape
        self.cpu_values = torch.from_numpy(self.volume.reshape(-1, C))
        self.cpu_mask = ~torch.from_numpy(self.nodata_mask_vol.reshape(-1).astype(bool))
        
        noise_mask = (self.cpu_values > valid_threshold).all(dim=-1)
        self.cpu_mask = self.cpu_mask & noise_mask
        
        if use_log:
            torch.log1p_(self.cpu_values)
            self.meta_info['use_log'] = True
        
        self.meta_info['data_range'] = [0.0, 1.0]
        self.cpu_values[~self.cpu_mask] = 0
        self.time_lut = torch.tensor(self.meta_info['t_norms'], dtype=torch.float32, device=self.device)
        
        offset_x, offset_y = 0.5 / W, 0.5 / H
        gy, gx = torch.meshgrid(
            torch.linspace(offset_y, 1.0 - offset_y, H),
            torch.linspace(offset_x, 1.0 - offset_x, W),
            indexing='ij'
        )
        self.xyz_cache = torch.stack([gx.flatten(), gy.flatten()], dim=-1).float()
        
        del self.volume
        del self.nodata_mask_vol
        gc.collect()
    
    def build_kplanes_model(
        self,
        model_type: str = "CCDC",
        xy_hash_size: int = 19,
        t_hash_size: int = 17,
        n_levels: int = 16,
        per_level_scale: float = 1.5,
        base_resolution: int = 32,
        features_per_level: int = 2,
        decoder_hidden: int = 128,
        n_hidden_layers: int = 2,
        n_harmonics: int = 3,
        ablation_mode: str = "full",
        anchor_dynamic_state: bool = True,
        time_varying_uncertainty: bool = True,
        horizon_uncertainty: bool = True,
    ):
        """构建模型"""
        self.model_type = model_type
        
        if model_type == "CCDC":
            years = max(((self.global_end_date - self.global_start_date).days / 365.25), 0.1)
            
            # [修改点 2]：将 n_hidden_layers 写入配置字典，传给底层网络
            self.config = {
                "type": "GeoCCDC",
                "n_bands": self.meta_info['n_bands'],
                "time_span_years": years,
                "xy_hash_size": xy_hash_size,
                "t_hash_size": t_hash_size,
                "n_levels": n_levels,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "features_per_level": features_per_level,
                "decoder_hidden": decoder_hidden,
                "n_harmonics": n_harmonics,
                "n_steps": 1,
                "is_probabilistic": ablation_mode != "no_prob",
                "n_hidden_layers": n_hidden_layers,
                "ablation_mode": ablation_mode,
                "architecture_revision": 2,
                "anchor_dynamic_state": anchor_dynamic_state,
                "time_varying_uncertainty": time_varying_uncertainty,
                "horizon_uncertainty": horizon_uncertainty,
            }
            
            print(
                f"[Physics] Building GeoCCDC V6.2. Scale={per_level_scale}, "
                f"BaseRes={base_resolution}, Ablation={ablation_mode}"
            )
            
            sig = inspect.signature(GeoCCDCNetwork.__init__)
            valid_params = [p.name for p in sig.parameters.values() if p.name != 'self']
            constructor_kwargs = {k: v for k, v in self.config.items() if k in valid_params}
            
            self.model = GeoCCDCNetwork(**constructor_kwargs).to(self.device)
        else:
            raise NotImplementedError(f"Model type {model_type} is not yet supported")
    
    def fit(
        self,
        epochs: int = 500,
        lr: float = 0.01,
        batch_size_points: int = 2 ** 19,
        lambda_reg: float = 0.005,
        lambda_dynamic_magnitude: float = 0.005,
    ):
        if self.model is None: raise ValueError("Model not built.")
        self.config["training_objective"] = {
            "data_term": "heteroscedastic_gaussian_nll",
            "warmup": "none",
            "temporal_regularizer": "total_variation_on_adjacent_queries",
            "lambda_temporal_tv": float(lambda_reg),
            "dynamic_magnitude_regularizer": "l1",
            "lambda_dynamic_magnitude": float(lambda_dynamic_magnitude),
            "dynamic_anchor": "D(z,t)-D(z,0)",
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        ablation_mode = self.config.get("ablation_mode", "full")
        criterion = HeteroscedasticLoss(ablation_mode=ablation_mode)
        N_times, H, W = len(self.meta_info['timestamps']), self.meta_info['height'], self.meta_info['width']
        spatial_size = H * W
        total_points = N_times * spatial_size
        
        # 1. 计算内存占用，决定是否走流式
        bytes_v = self.cpu_values.numel() * self.cpu_values.element_size()
        bytes_m = self.cpu_mask.numel() * self.cpu_mask.element_size()
        data_size_gb = (bytes_v + bytes_m) / (1024 ** 3)
        use_streaming = data_size_gb > 6.0  # 超过 6GB 自动开启流式
        
        pbar = tqdm(total=epochs, desc="Training [Heteroscedastic NLL]")
        
        if not use_streaming:
            # ==========================================
            # 模式一：[Mode: Full GPU] 小数据，直接榨干显卡
            # ==========================================
            print(f"[Mode] Full GPU - Data size: {data_size_gb:.2f} GB")
            gpu_v = self.cpu_values.view(N_times, spatial_size, -1).to(self.device)
            gpu_m = self.cpu_mask.view(N_times, spatial_size).to(self.device)
            gpu_c = self.xyz_cache.to(self.device)
            gpu_t = self.time_lut.to(self.device)
            
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = None
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    rand_idx = torch.randint(0, total_points, (batch_size_points,), device=self.device)
                    m_pt = gpu_m.view(-1)[rand_idx]
                    
                    if m_pt.any():
                        valid_idx = rand_idx[m_pt]
                        f_idx = torch.div(valid_idx, spatial_size, rounding_mode='floor')
                        s_idx = valid_idx % spatial_size
                        targets = gpu_v.view(-1, self.meta_info['n_bands'])[valid_idx]
                        
                        t_norms_batch = gpu_t[f_idx].unsqueeze(-1)
                        out = self.model.forward_with_state(gpu_c[s_idx], t_norms_batch)
                        pred_mean, log_var, learned_decay, all_deltas, dynamic_state = out
                        
                        time_span_years = self.config.get('time_span_years', 3.0)
                        delta_t_years = (1.0 - t_norms_batch) * time_span_years
                        
                        time_weights = torch.exp(-learned_decay * delta_t_years * 0.5) + 0.5
                        main_loss = criterion(pred_mean, log_var, targets, weights=time_weights)
                        
                        tv_loss = (
                            lambda_reg
                            * sum(torch.mean(torch.abs(d)) for d in all_deltas)
                            if all_deltas
                            else pred_mean.new_zeros(())
                        )
                        magnitude_loss = (
                            lambda_dynamic_magnitude
                            * torch.mean(torch.abs(dynamic_state))
                            if self.model.uses_dynamic_branch
                            else pred_mean.new_zeros(())
                        )
                        reg_loss = tv_loss + magnitude_loss
                        
                        loss = main_loss + reg_loss
                
                if loss is None:
                    pbar.update(1)
                    pbar.set_postfix({'loss': 'skipped-no-valid-samples'})
                    scheduler.step()
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        else:
            # ==========================================
            # 模式二：[Mode: Threaded Streaming] 大数据，多线程异步搬运
            # ==========================================
            print(f"[Mode] Threaded Streaming - Data size: {data_size_gb:.2f} GB")
            super_batch_size = batch_size_points * 20
            data_queue = queue.Queue(maxsize=3)
            stop_event = threading.Event()
            
            cpu_v = self.cpu_values.view(-1, self.meta_info['n_bands'])
            cpu_m = self.cpu_mask.view(-1)
            cpu_c = self.xyz_cache
            cpu_t = self.time_lut.cpu()
            
            def producer():
                while not stop_event.is_set():
                    rand_idx = torch.randint(0, total_points, (super_batch_size,))
                    m_pt = cpu_m[rand_idx]
                    
                    if not m_pt.any():
                        continue
                        
                    valid_idx = rand_idx[m_pt]
                    f_idx = torch.div(valid_idx, spatial_size, rounding_mode='floor')
                    s_idx = valid_idx % spatial_size
                    
                    c_batch = cpu_c[s_idx]
                    t_batch = cpu_t[f_idx].unsqueeze(-1)
                    v_batch = cpu_v[valid_idx]
                    
                    data_queue.put((v_batch.pin_memory(), c_batch.pin_memory(), t_batch.pin_memory()))

            t_thread = threading.Thread(target=producer, daemon=True)
            t_thread.start()
            
            steps_completed = 0
            while steps_completed < epochs:
                v_super, c_super, t_super = data_queue.get()
                
                g_v = v_super.to(self.device, non_blocking=True)
                g_c = c_super.to(self.device, non_blocking=True)
                g_t = t_super.to(self.device, non_blocking=True)
                
                total_valid = g_v.shape[0]
                inner_steps = max(1, total_valid // batch_size_points)
                
                for i in range(inner_steps):
                    if steps_completed >= epochs:
                        break
                        
                    start_idx = i * batch_size_points
                    end_idx = min(start_idx + batch_size_points, total_valid)
                    
                    if start_idx >= end_idx:
                        break
                        
                    batch_targets = g_v[start_idx:end_idx]
                    batch_c = g_c[start_idx:end_idx]
                    batch_t = g_t[start_idx:end_idx]
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        out = self.model.forward_with_state(batch_c, batch_t)
                        pred_mean, log_var, learned_decay, all_deltas, dynamic_state = out
                        
                        time_span_years = self.config.get('time_span_years', 3.0)
                        delta_t_years = (1.0 - batch_t) * time_span_years
                        
                        time_weights = torch.exp(-learned_decay * delta_t_years * 0.5) + 0.5
                        main_loss = criterion(pred_mean, log_var, batch_targets, weights=time_weights)
                        
                        tv_loss = (
                            lambda_reg
                            * sum(torch.mean(torch.abs(d)) for d in all_deltas)
                            if all_deltas
                            else pred_mean.new_zeros(())
                        )
                        magnitude_loss = (
                            lambda_dynamic_magnitude
                            * torch.mean(torch.abs(dynamic_state))
                            if self.model.uses_dynamic_branch
                            else pred_mean.new_zeros(())
                        )
                        reg_loss = tv_loss + magnitude_loss
                        loss = main_loss + reg_loss
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    
                    steps_completed += 1
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            stop_event.set()
            while not data_queue.empty():
                try: data_queue.get_nowait()
                except queue.Empty: break
            t_thread.join()
            
        pbar.close()
    
    # [修改点 3]：引入 save_residuals 和 residual_dtype 参数
    def save_dual_gndc(self, model_filename: str, residual_filename: str, quant_mode: str = "int16", save_residuals: bool = True, residual_dtype: str = "int16", empirical_null_profile: Optional[Mapping] = None):
        """
        保存模型为指定精度的GNDC文件（已修复元数据嵌套与量化偏差Bug）
        """
        # 1. 定义数据类型与量化范围映射
        dtype_map = {
            'float32': {'np': np.float32, 'max': None, 'is_int': False},
            'float16': {'np': np.float16, 'max': None, 'is_int': False},
            'int16':   {'np': np.int16,   'max': 32767,  'is_int': True},
            'int8':    {'np': np.int8,    'max': 127,    'is_int': True}
        }
        
        info = dtype_map.get(quant_mode, dtype_map['int16'])
        target_np = info['np']
        
        # 2. 导出模型权重并进行量化
        print(f"[NDC] Exporting weights as {quant_mode}...")
        all_params = np.concatenate([p.detach().cpu().numpy().flatten() for p in self.model.parameters()])
        
        scale_w = 1.0
        if info['is_int']:
            # 动态计算Scale以最大化利用位深
            max_val = np.max(np.abs(all_params)) + 1e-9
            scale_w = info['max'] / max_val
            processed_weights = (all_params * scale_w).round().clip(-info['max'], info['max']-1).astype(target_np)
        else:
            processed_weights = all_params.astype(target_np)

        # 压缩权重
        weights_bytes = fast_compress(processed_weights.tobytes())
        
        # 3. 核心修复：清理元数据 (去壳逻辑)
        actual_meta = self.meta_info.get('dataset_meta', self.meta_info)
        
        if 'global_start_date' not in actual_meta:
            actual_meta['global_start_date'] = str(self.global_start_date)
        if 'global_end_date' not in actual_meta:
            actual_meta['global_end_date'] = str(self.global_end_date)
        if 'shape' not in actual_meta:
            N_times = len(actual_meta.get('timestamps', []))
            H, W = actual_meta.get('height', 0), actual_meta.get('width', 0)
            C = actual_meta.get('n_bands', 0)
            actual_meta['shape'] = [N_times, H, W, C]

        # 4. 构造完整 Header
        header = {
            "version": "0.8.0",
            "file_type": "weights_only",
            "model_config": self.config,  
            "quantization_info": {
                "mode": quant_mode, 
                "scale": float(scale_w),
                "is_int": info['is_int']
            },
            "dataset_meta": actual_meta
        }
        if empirical_null_profile is not None:
            header["empirical_null_profile"] = dict(empirical_null_profile)
            header["calibration_profile_type"] = "global_terminal_state_empirical_null"
        
        # 5. 写入权重文件
        with open(model_filename, 'wb') as f:
            h_json = json.dumps(header, default=str).encode('utf-8')
            f.write(struct.pack("I", len(h_json)))
            f.write(h_json)
            f.write(struct.pack("I", len(weights_bytes)))
            f.write(weights_bytes)

        # ========= [修改点 4]：增加残差保存拦截 =========
        if not save_residuals:
            print("[NDC] Skipping residuals export as per YAML config (save_residuals: False).")
            return
        # ==============================================

        # ==========================================================
        # 🚀 核心修复：将量化后的权重反注回模型，让残差吸收量化误差！
        # ==========================================================
        print(f"[NDC] Injecting quantized {quant_mode} weights back for accurate residual calculation...")
        dequantized_weights = processed_weights.astype(np.float32) / scale_w
        with torch.no_grad():
            ptr = 0
            for p in self.model.parameters():
                numel = p.numel()
                p.copy_(torch.from_numpy(dequantized_weights[ptr:ptr+numel]).view_as(p).to(self.device))
                ptr += numel
        # ==========================================================
            
        # 6. 处理残差文件
        print(f"[NDC] Calculating and exporting true residuals...")
        self.model.eval()
        
        N_times, H, W = len(actual_meta['timestamps']), actual_meta['height'], actual_meta['width']
        C = actual_meta['n_bands']
        
        with torch.no_grad():
            gpu_c = self.xyz_cache.to(self.device)
            gpu_t = self.time_lut.to(self.device)
            
            all_preds = []
            # 逐帧推理以防止显存溢出
            for t_idx in range(N_times):
                t_val = gpu_t[t_idx].view(1, 1).expand(H * W, 1)
                out = self.model(gpu_c, t_val)
                pred = out[0] if isinstance(out, tuple) else out
                all_preds.append(pred.cpu())
                
            full_pred = torch.stack(all_preds, dim=0).view(N_times, H * W, C)
            obs_val = self.cpu_values.view(N_times, H * W, C)
            
            # 计算真实残差: 观测值 - 预测值
            res_float = (obs_val - full_pred).numpy()
            
            # ========= [修改点 5]：使用 YAML 传入的残差类型进行动态转换 =========
            if residual_dtype == 'int16':
                res_raw = np.round(res_float * 10000).clip(-32768, 32767).astype(np.int16)
            elif residual_dtype == 'int8':
                res_raw = np.round(res_float * 100).clip(-128, 127).astype(np.int8)
            elif residual_dtype == 'float16':
                res_raw = res_float.astype(np.float16)
            else:
                res_raw = res_float.astype(np.float32)
            # ====================================================================
            
            # 极限压缩残差数据
            res_compressed_bytes = fast_compress(res_raw.tobytes())

        # 更新残差 Header 标识
        header['file_type'] = "residual_only"
        header['has_residuals'] = True
        
        # 根据残差类型动态记录缩放倍率，供 Reader 解码使用
        scale_r = 1.0
        if residual_dtype == 'int16':
            scale_r = 10000.0
        elif residual_dtype == 'int8':
            scale_r = 100.0
            
        header['residual_info'] = {
            'dtype': residual_dtype,
            'scale': scale_r
        }
        
        with open(residual_filename, 'wb') as f:
            h_json = json.dumps(header, default=str).encode('utf-8')
            f.write(struct.pack("I", len(h_json)))
            f.write(h_json)
            # 写入真实的残差长度和数据
            f.write(struct.pack("I", len(res_compressed_bytes)))
            f.write(res_compressed_bytes)

        print(f"[NDC] True Residuals ({residual_dtype}) export complete.")
