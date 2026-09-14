import os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ground.core.NDCReader import NDCReader
    from ground.core.NDCUtils import normalize_time, extract_date_from_filename
    from common.constants import DEFAULT_CHUNK_SIZE
except ImportError:
    # 这里的路径处理是为了确保在不同环境下运行都能找到依赖
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ground.core.NDCReader import NDCReader
    from ground.core.NDCUtils import normalize_time, extract_date_from_filename
    DEFAULT_CHUNK_SIZE = 262144

class OnboardInference:
    def __init__(self, model_path, device=None):
        """
        初始化推理引擎
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Initializing on device: {self.device}")
        
        self.reader = NDCReader(device=self.device)
        self.reader.load(model_path)
        
        self.meta = self.reader.dataset_meta
        self.model = self.reader.model
        self.empirical_null_profile = self.reader.empirical_null_profile

        # 补全元数据逻辑 (针对 Weights-only 文件)
        if not self.meta or 'n_bands' not in self.meta:
            print("[Warning] Weights-only model detected. Manually injecting metadata...")
            if not self.meta: self.meta = {}
            self.meta['n_bands'] = 7 
            self.meta.setdefault('global_start_date', "2017-03-02T00:00:00Z")
            self.meta.setdefault('global_end_date', "2020-02-10T23:59:59Z")
            self.meta.setdefault('data_range', [0.0, 1.0])

        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.band_indices = list(range(self.meta['n_bands']))
        
        print(f"[Inference] Model loaded: {self.meta['n_bands']} bands")

    @staticmethod
    def _log_variance_to_standard_deviation(log_variance):
        """Convert the network log-variance output to standard deviation."""

        return torch.exp(0.5 * torch.clamp(log_variance, min=-7.0, max=5.0))
        
    def _create_coords_grid(self, H, W):
        # 确保使用 (idx + 0.5) / size 逻辑
        offset_x, offset_y = 0.5 / W, 0.5 / H
        cols = torch.linspace(offset_x, 1.0 - offset_x, W, device=self.device)
        rows = torch.linspace(offset_y, 1.0 - offset_y, H, device=self.device)
        gy, gx = torch.meshgrid(rows, cols, indexing='ij')
        return torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    
    def predict_frame(self, target_datetime, H, W, chunk_size=None):
        """
        推理单帧影像，返回预测的均值和基于概率网络预测的标准差
        """
        # 1. 时间归一化
        gs = datetime.fromisoformat(str(self.meta['global_start_date']).replace("Z", "+00:00"))
        ge = datetime.fromisoformat(str(self.meta['global_end_date']).replace("Z", "+00:00"))
        t_norm = normalize_time(target_datetime, gs, ge)
        
        chunk = chunk_size or self.chunk_size
        coords = self._create_coords_grid(H, W)
        
        C = len(self.band_indices)
        mean = np.zeros((H * W, C), dtype=np.float32)
        std_net = np.zeros((H * W, C), dtype=np.float32)
        
        dmin, dmax = self.meta['data_range']
        use_log = self.meta.get('use_log', False)
        
        # 2. 运行模型预测（同时获取均值和标准差）
        self.model.eval()
        with torch.no_grad():
            for i in range(0, H * W, chunk):
                end = min(i + chunk, H * W)
                n_points = end - i
                
                t_in = torch.full((n_points, 1), t_norm, device=self.device)
                out = self.model(coords[i:end], t_in)
                
                if isinstance(out, tuple):
                    pred_mean_batch = out[0]
                    pred_std_batch = self._log_variance_to_standard_deviation(out[1])
                else:
                    pred_mean_batch = out
                    pred_std_batch = torch.full_like(pred_mean_batch, 0.01)
                
                # 还原反射率尺度 (先反归一化)
                pred_mean_batch = pred_mean_batch * (dmax - dmin) + dmin
                
                # 逆对数变换
                if use_log:
                    pred_mean_batch = torch.expm1(pred_mean_batch)
                
                mean[i:end] = pred_mean_batch.cpu().numpy()
                std_net[i:end] = pred_std_batch.cpu().numpy()
        
        mean_reshaped = mean.reshape(H, W, C)
        
        # 3. 智能选择标准差来源
        if getattr(self.reader, 'spatial_std', None) is not None:
            # 如果有残差文件，优先使用残差文件算出来的真实物理 RMSE
            std_final = self.reader.spatial_std * (dmax - dmin)
        else:
            # 🚀 如果只有权重文件，完美回退到神经网络自身学习到的异方差 (Aleatoric Uncertainty)
            # 将网络输出的标准差缩放到真实的反射率尺度
            std_final = std_net.reshape(H, W, C) * (dmax - dmin)
            
        return mean_reshaped, std_final
    
    def predict_frame_by_tif(self, tif_path, chunk_size=None):
        """
        通过TIF文件推理
        """
        from common.scene_inference import read_observation

        obs, _, _, _ = read_observation(
            tif_path,
            expected_bands=self.meta["n_bands"],
            source_band_indices=self.meta.get("source_band_indices"),
            source_band_names=self.meta.get("source_band_names"),
        )
        H, W, _ = obs.shape
        
        date_dt = extract_date_from_filename(os.path.basename(tif_path))
        mean, std = self.predict_frame(date_dt, H, W, chunk_size)
        return mean, std, obs
    
    # --- 修复 AttributeError: 补全缺失的方法 ---
    def get_meta(self):
        """获取模型元数据"""
        return self.meta
    
    def get_band_indices(self):
        """获取使用的波段索引"""
        return self.band_indices

    def get_empirical_null_profile(self):
        """Return the calibration profile embedded in the GNDC, if present."""

        return self.empirical_null_profile

    def require_current_model_contract(self, expected_ablation_mode=None):
        """Reject legacy GNDC files in experiments using the revised method."""

        config = self.reader.model_config
        revision = int(config.get("architecture_revision", 0))
        required_flags = {
            "anchor_dynamic_state": True,
            "time_varying_uncertainty": True,
            "horizon_uncertainty": True,
        }
        failures = []
        if revision < 2:
            failures.append(f"architecture_revision={revision}, expected >= 2")
        for key, expected in required_flags.items():
            if bool(config.get(key, False)) is not expected:
                failures.append(f"{key}={config.get(key)!r}, expected {expected}")
        if expected_ablation_mode is not None:
            actual_mode = config.get("ablation_mode", "full")
            if actual_mode != expected_ablation_mode:
                failures.append(
                    f"ablation_mode={actual_mode!r}, expected {expected_ablation_mode!r}"
                )
        if not bool(self.meta.get("input_quality_screened", False)):
            failures.append("input_quality_screened metadata is absent or false")
        if "last_training_observation_date" not in self.meta:
            failures.append("last_training_observation_date metadata is absent")
        n_bands = int(self.meta.get("n_bands", 0))
        source_indices = self.meta.get("source_band_indices")
        source_names = self.meta.get("source_band_names")
        if source_indices is not None and len(source_indices) != n_bands:
            failures.append(
                f"source_band_indices has {len(source_indices)} entries, expected {n_bands}"
            )
        if source_names is not None and len(source_names) != n_bands:
            failures.append(
                f"source_band_names has {len(source_names)} entries, expected {n_bands}"
            )
        if n_bands == 5 and (source_indices is None or source_names is None):
            failures.append("five-band models require explicit source-band metadata")
        if failures:
            raise RuntimeError(
                "This GNDC predates the revised SINC model contract and cannot be "
                "used for manuscript experiments. Retrain it with the current "
                "trainer. Details: " + "; ".join(failures)
            )
        return config


def load_model(model_path, device=None):
    return OnboardInference(model_path, device)
