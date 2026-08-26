"""
影像预处理模块 - 读取和预处理遥感影像
"""
import os
import numpy as np
import rasterio
from pathlib import Path

try:
    from ground.core.NDCUtils import extract_date_from_filename
    from common.constants import (
        DEFAULT_SCALE_FACTOR, DEFAULT_VALID_RANGE, VALID_THRESHOLD
    )
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ground.core.NDCUtils import extract_date_from_filename
    from common.constants import (
        DEFAULT_SCALE_FACTOR, DEFAULT_VALID_RANGE, VALID_THRESHOLD
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
        
    def load_tif(self, tif_path, expected_bands=None):
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
            C = src.count
            
            if expected_bands and C != expected_bands:
                print(f"[Warning] Expected {expected_bands} bands, got {C}. Using all bands.")
            
            raw_data = src.read().transpose(1, 2, 0).astype(np.float32)
            
            transform = src.transform
            crs = src.crs.to_wkt() if src.crs else None
            
            date_str = None
            date_dt = extract_date_from_filename(os.path.basename(tif_path))
            if date_dt:
                date_str = date_dt.strftime("%Y-%m-%d")
        
        obs = raw_data.copy()
        
        if np.max(obs) > 10:
            obs *= self.scale_factor
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
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
            'bands': C
        }
        
        return obs, meta
    
    def load_tif_sequence(self, folder_path, date_start=None, date_end=None, pattern="*.tif"):
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
        tif_files = sorted(folder.glob(pattern))
        
        images = []
        
        for tif_path in tif_files:
            date_dt = extract_date_from_filename(tif_path.name)
            
            if date_start and date_dt and date_dt < date_start:
                continue
            if date_end and date_dt and date_dt > date_end:
                continue
            
            try:
                obs, meta = self.load_tif(str(tif_path))
                images.append((obs, meta))
            except Exception as e:
                print(f"[Warning] Failed to load {tif_path}: {e}")
        
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
