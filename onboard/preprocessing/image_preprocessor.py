"""
影像预处理模块 - 读取和预处理遥感影像
"""
import os
import warnings
import numpy as np
import rasterio
from pathlib import Path

try:
    from ground.core.NDCUtils import (
        extract_date_from_filename,
        group_dated_paths_by_day,
    )
    from common.constants import (
        DAILY_OBSERVATION_PROTOCOL,
        DEFAULT_SCALE_FACTOR,
        DEFAULT_VALID_RANGE,
        VALID_THRESHOLD,
    )
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ground.core.NDCUtils import (
        extract_date_from_filename,
        group_dated_paths_by_day,
    )
    from common.constants import (
        DAILY_OBSERVATION_PROTOCOL,
        DEFAULT_SCALE_FACTOR,
        DEFAULT_VALID_RANGE,
        VALID_THRESHOLD,
    )


class ImagePreprocessor:
    def __init__(self, 
                 scale_factor=DEFAULT_SCALE_FACTOR,
                 valid_range=DEFAULT_VALID_RANGE,
                 valid_threshold=VALID_THRESHOLD):
        """
        初始化预处理器
        
        Args:
            scale_factor: 缩放因子（如0.0001将DN值转为反射率）
            valid_range: 有效值范围 (min, max)
            valid_threshold: 有效像素阈值
        """
        self.scale_factor = scale_factor
        self.valid_range = valid_range
        self.valid_threshold = valid_threshold
        
    def load_tif(
        self,
        tif_path,
        expected_bands=None,
        source_band_indices=None,
        source_band_names=None,
    ):
        """
        加载并预处理TIF文件
        
        Args:
            tif_path: TIF文件路径
            expected_bands: 期望的波段数
            
        Returns:
            obs: 预处理后的影像 (H, W, C)
            meta: 影像元数据
        """
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
            source_count = src.count
            selected_indices = (
                list(range(1, source_count + 1))
                if source_band_indices is None
                else [int(value) for value in source_band_indices]
            )
            if expected_bands is not None and len(selected_indices) != expected_bands:
                if source_band_indices is None:
                    raise ValueError(
                        f"Raster contains {source_count} bands but the model expects "
                        f"{expected_bands}; provide an explicit source_band_indices mapping"
                    )
                raise ValueError(
                    f"Expected {expected_bands} selected bands, got "
                    f"{len(selected_indices)}"
                )
            if not selected_indices or min(selected_indices) < 1:
                raise ValueError(
                    "source_band_indices must be one-based positive integers"
                )
            if max(selected_indices) > source_count:
                raise ValueError(
                    f"Requested source band {max(selected_indices)} but {tif_path} "
                    f"contains only {source_count} bands"
                )
            if source_band_names and any(src.descriptions):
                actual_names = [src.descriptions[index - 1] for index in selected_indices]
                if actual_names != list(source_band_names):
                    raise ValueError(
                        f"Configured source bands {list(source_band_names)} do not "
                        f"match GeoTIFF descriptions {actual_names} in {tif_path}"
                    )

            raw_data = (
                src.read(indexes=selected_indices)
                .transpose(1, 2, 0)
                .astype(np.float32)
            )
            C = raw_data.shape[-1]
            
            transform = src.transform
            crs = src.crs.to_wkt() if src.crs else None
            
            date_str = None
            date_dt = extract_date_from_filename(os.path.basename(tif_path))
            if date_dt:
                date_str = date_dt.strftime("%Y-%m-%d")
        
        obs = raw_data.copy()
        
        finite_values = obs[np.isfinite(obs)]
        if finite_values.size and float(np.max(finite_values)) > 10.0:
            obs *= self.scale_factor
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if self.valid_threshold is not None:
            obs[obs <= self.valid_threshold] = 0.0
        
        if self.valid_range:
            obs[(obs < self.valid_range[0]) | (obs > self.valid_range[1])] = 0.0
        
        meta = {
            'path': tif_path,
            'date': date_dt,
            'date_str': date_str,
            'shape': (H, W, C),
            'transform': list(transform)[:6],
            'crs': crs,
            'width': W,
            'height': H,
            'bands': C,
            'source_bands': source_count,
            'source_band_indices': selected_indices,
            'source_band_names': list(source_band_names or []),
        }
        
        return obs, meta
    
    def load_tif_sequence(
        self,
        folder_path,
        date_start=None,
        date_end=None,
        pattern="*.tif",
        expected_bands=None,
        source_band_indices=None,
        source_band_names=None,
    ):
        """
        加载文件夹中的TIF序列
        
        Args:
            folder_path: 文件夹路径
            date_start: 开始日期 (datetime)
            date_end: 结束日期 (datetime)
            pattern: 文件匹配模式
            
        Returns:
            images: 影像列表 [(obs, meta), ...]
        """
        folder = Path(folder_path)
        dated_groups = group_dated_paths_by_day(folder.glob(pattern))
        
        images = []
        
        for date_dt, source_paths in dated_groups:
            
            if date_start and date_dt and date_dt < date_start:
                continue
            if date_end and date_dt and date_dt > date_end:
                continue
            
            try:
                daily_arrays = []
                meta = None
                for tif_path in source_paths:
                    current, current_meta = self.load_tif(
                        str(tif_path),
                        expected_bands=expected_bands,
                        source_band_indices=source_band_indices,
                        source_band_names=source_band_names,
                    )
                    valid = np.isfinite(current)
                    valid &= current > self.valid_threshold
                    valid &= current <= self.valid_range[1]
                    daily_arrays.append(np.where(valid, current, np.nan))
                    if meta is None:
                        meta = current_meta
                    elif (
                        current.shape != daily_arrays[0].shape
                        or current_meta["transform"] != meta["transform"]
                        or current_meta["crs"] != meta["crs"]
                    ):
                        raise ValueError(
                            f"Same-date raster shape mismatch: {tif_path}"
                        )
                if len(daily_arrays) == 1:
                    obs = daily_arrays[0]
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="All-NaN slice encountered"
                        )
                        obs = np.nanmedian(np.stack(daily_arrays, axis=0), axis=0)
                obs = np.nan_to_num(obs, nan=0.0).astype(np.float32, copy=False)
                meta["date"] = date_dt
                meta["date_str"] = date_dt.strftime("%Y-%m-%d")
                meta["source_paths"] = [str(path) for path in source_paths]
                meta["source_file_count"] = len(source_paths)
                meta["temporal_observation_protocol"] = DAILY_OBSERVATION_PROTOCOL
                images.append((obs, meta))
            except Exception as e:
                print(f"[Warning] Failed to load daily observation {date_dt.date()}: {e}")
        
        print(f"[Preprocessor] Loaded {len(images)} images from {folder_path}")
        
        return images
    
    def create_nodata_mask(self, obs):
        """
        创建无效数据掩膜
        
        Args:
            obs: 影像数据 (H, W, C)
            
        Returns:
            nodata_mask: 无效数据掩膜 (H, W)
        """
        nodata_mask = np.all(obs == 0, axis=-1)
        return nodata_mask
    
    def validate_input(self, obs, meta, expected_bands=None):
        """
        验证输入数据的有效性
        
        Args:
            obs: 影像数据
            meta: 元数据
            expected_bands: 期望的波段数
            
        Returns:
            is_valid: 是否有效
            message: 验证消息
        """
        if obs is None or obs.size == 0:
            return False, "Empty image data"
        
        H, W, C = obs.shape
        
        if H == 0 or W == 0:
            return False, "Invalid dimensions"
        
        if expected_bands and C != expected_bands:
            return False, f"Band count mismatch: expected {expected_bands}, got {C}"
        
        if meta.get('date') is None:
            return False, "No valid date found in filename"
        
        return True, "Valid"


def load_image(tif_path, scale_factor=DEFAULT_SCALE_FACTOR):
    """
    便捷函数：加载单景影像
    
    Args:
        tif_path: TIF文件路径
        scale_factor: 缩放因子
        
    Returns:
        obs, meta
    """
    processor = ImagePreprocessor(scale_factor=scale_factor)
    return processor.load_tif(tif_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_preprocessor.py <tif_path>")
        sys.exit(1)
    
    tif_path = sys.argv[1]
    
    print(f"Loading: {tif_path}")
    processor = ImagePreprocessor()
    obs, meta = processor.load_tif(tif_path)
    
    print(f"Shape: {obs.shape}")
    print(f"Date: {meta.get('date_str')}")
    print(f"Transform: {meta.get('transform')}")
    print(f"Value range: [{obs.min():.4f}, {obs.max():.4f}]")
