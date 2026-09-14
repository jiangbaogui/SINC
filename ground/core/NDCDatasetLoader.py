"""
数据加载模块 - 加载遥感影像数据 (0-RAM 内存映射流式优化版)
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
import gc
from tqdm import tqdm
import atexit

from common.constants import (
    DAILY_OBSERVATION_PROTOCOL,
    DEFAULT_VALID_RANGE,
    VALID_THRESHOLD,
)
from .NDCUtils import (
    extract_date_from_filename,
    group_dated_paths_by_day,
    normalize_time,
)


class DatasetLoader:
    """遥感数据集加载器 (支持超大数据集的内存映射)"""
    
    def __init__(self, dtype=np.float16, use_memmap=True, cache_dir="./.ndc_cache"):
        self.dtype_np = dtype
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir
        self._temp_files = []
        
        # 注册退出时的清理函数，防止临时大文件占用硬盘
        atexit.register(self.cleanup)
        
    def cleanup(self):
        """清理硬盘上的临时映射文件"""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"[Warning] Failed to clean up temp file {f}: {e}")
    
    def load_folder(
        self,
        folder_path: str,
        expected_bands: int = 1,
        source_band_indices: Optional[List[int]] = None,
        source_band_names: Optional[List[str]] = None,
        scale_factor: float = 0.1,
        valid_range: Optional[Tuple[float, float]] = None,
        p99_removal: bool = False,
        valid_threshold: float = VALID_THRESHOLD,
        temporal_bin_days: Optional[int] = None,
        roi_window: Optional[Tuple[int, int, int, int]] = None,
        global_start_date: Optional[datetime] = None,
        global_end_date: Optional[datetime] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        
        meta_info = {}
        selected_indices = (
            list(range(1, expected_bands + 1))
            if source_band_indices is None
            else [int(value) for value in source_band_indices]
        )
        if len(selected_indices) != expected_bands:
            raise ValueError(
                "source_band_indices must contain exactly expected_bands entries"
            )
        if any(value < 1 for value in selected_indices):
            raise ValueError("source_band_indices must use one-based positive indices")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("source_band_indices cannot contain duplicates")
        selected_names = (
            None
            if source_band_names is None
            else [str(value) for value in source_band_names]
        )
        if selected_names is not None and len(selected_names) != expected_bands:
            raise ValueError(
                "source_band_names must contain exactly expected_bands entries"
            )
        tif_files = glob.glob(os.path.join(folder_path, "**", "*.tif"), recursive=True)
        raw_valid_files = []
        
        for f in tif_files:
            dt = extract_date_from_filename(os.path.basename(f))
            if dt and (global_start_date is None or dt >= global_start_date) and \
                    (global_end_date is None or dt <= global_end_date):
                raw_valid_files.append({"path": f, "date": dt})
        
        if not raw_valid_files:
            raise ValueError("No valid files found in the specified range.")
        
        raw_valid_files.sort(key=lambda x: x["date"])
        
        if temporal_bin_days:
            processed_list = self._temporal_binning(raw_valid_files, temporal_bin_days)
            temporal_protocol = (
                f"{temporal_bin_days}_day_nanmedian_reflectance_gt0_le1_v1"
            )
        else:
            daily_groups = group_dated_paths_by_day(
                (item["path"] for item in raw_valid_files),
                start=global_start_date,
                end=global_end_date,
            )
            processed_list = [
                {
                    "date": date,
                    "files": [{"date": date, "path": str(path)} for path in paths],
                }
                for date, paths in daily_groups
            ]
            temporal_protocol = DAILY_OBSERVATION_PROTOCOL

        meta_info["source_file_count"] = len(raw_valid_files)
        meta_info["temporal_observation_count"] = len(processed_list)
        meta_info["same_day_composite_count"] = sum(
            len(item["files"]) > 1 for item in processed_list
        )
        meta_info["temporal_observation_protocol"] = temporal_protocol
        meta_info["input_reflectance_valid_range"] = [
            float(valid_threshold),
            float(valid_range[1] if valid_range else DEFAULT_VALID_RANGE[1]),
        ]
        
        first_tif = raw_valid_files[0]["path"]
        with rasterio.open(first_tif) as src:
            if max(selected_indices) > src.count:
                raise ValueError(
                    f"Requested source band {max(selected_indices)} but {first_tif} "
                    f"contains only {src.count} bands"
                )
            if selected_names and any(src.descriptions):
                actual_names = [src.descriptions[index - 1] for index in selected_indices]
                if actual_names != selected_names:
                    raise ValueError(
                        f"Configured source bands {selected_names} do not match "
                        f"GeoTIFF descriptions {actual_names} in {first_tif}"
                    )
            if roi_window:
                col_off, row_off, w, h = roi_window
                meta_info['width'], meta_info['height'] = w, h
                meta_info['transform'] = list(rasterio.windows.transform(Window(*roi_window), src.transform))[:6]
            else:
                meta_info['width'], meta_info['height'] = src.width, src.height
                meta_info['transform'] = list(src.transform)[:6]
            
            meta_info['crs'] = src.crs.to_wkt() if src.crs else ""
            meta_info['n_bands'] = expected_bands
            meta_info['source_band_indices'] = selected_indices
            if selected_names is not None:
                meta_info['source_band_names'] = selected_names
        
        N = len(processed_list)
        H, W, C = meta_info['height'], meta_info['width'], expected_bands
        
        # =========================================================
        # 🚀 优化 1：内存映射 (Memmap) 替代全量 RAM 分配
        # =========================================================
        if self.use_memmap:
            os.makedirs(self.cache_dir, exist_ok=True)
            vol_path = os.path.join(self.cache_dir, f"vol_{id(self)}.dat")
            mask_path = os.path.join(self.cache_dir, f"mask_{id(self)}.dat")
            self._temp_files.extend([vol_path, mask_path])
            
            print(f"[Loader] Allocating 0-RAM memory map (NVMe Disk Cache)...")
            volume = np.memmap(vol_path, dtype=self.dtype_np, mode='w+', shape=(N, H, W, C))
            nodata_mask = np.memmap(mask_path, dtype=bool, mode='w+', shape=(N, H, W))
        else:
            volume = np.zeros((N, H, W, C), dtype=self.dtype_np)
            nodata_mask = np.zeros((N, H, W), dtype=bool)
        
        final_timestamps = []
        final_t_norms = []
        
        for i, bin_info in enumerate(tqdm(processed_list, desc="Processing Volume (Streaming)")):
            bin_data_list = []
            for f_info in bin_info["files"]:
                with rasterio.open(f_info["path"]) as src:
                    if max(selected_indices) > src.count:
                        raise ValueError(
                            f"Requested source band {max(selected_indices)} but "
                            f"{f_info['path']} contains only {src.count} bands"
                        )
                    if selected_names and any(src.descriptions):
                        actual_names = [
                            src.descriptions[index - 1] for index in selected_indices
                        ]
                        if actual_names != selected_names:
                            raise ValueError(
                                f"Configured source bands {selected_names} do not match "
                                f"GeoTIFF descriptions {actual_names} in {f_info['path']}"
                            )
                    read_win = Window(*roi_window) if roi_window else None
                    data = src.read(indexes=selected_indices, window=read_win)
                    data = data.transpose(1, 2, 0).astype(np.float32)
                    
                    data *= scale_factor
                    if valid_threshold is not None:
                        data[data <= valid_threshold] = np.nan
                    if valid_range:
                        data[(data < valid_range[0]) | (data > valid_range[1])] = np.nan
                    
                    bin_data_list.append(data)
            
            # 融合当前时间窗口的切片
            if len(bin_data_list) > 1:
                frame_data = np.nanmedian(np.stack(bin_data_list), axis=0).astype(self.dtype_np)
            else:
                frame_data = bin_data_list[0].astype(self.dtype_np)
            
            # =========================================================
            # 🚀 优化 2：逐帧落盘与逐帧计算掩膜 (绝不将所有帧堆积在内存)
            # =========================================================
            volume[i] = frame_data
            nodata_mask[i] = np.isnan(frame_data).any(axis=-1)
            
            if self.use_memmap:
                volume.flush()       # 立即将这一帧写入硬盘
                nodata_mask.flush()  # 立即将这一帧的掩膜写入硬盘
            
            dt = bin_info["date"]
            final_timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            final_t_norms.append(normalize_time(dt, global_start_date, global_end_date))
            
            # 及时释放内存
            del bin_data_list, frame_data
            
        meta_info['timestamps'] = final_timestamps
        meta_info['t_norms'] = final_t_norms
        meta_info['trans_matrix'] = meta_info['transform']
        
        # =========================================================
        # 🚀 优化 3：分块 P99 计算，防止拉爆内存
        # =========================================================
        if p99_removal:
            print("[Loader] Calculating P99 on random chunks to prevent OOM...")
            # 随机抽样几帧来计算 P99，而不是对整个数组算
            sample_frames = np.random.choice(N, min(N, 5), replace=False)
            sample_data = np.concatenate([volume[idx].flatten() for idx in sample_frames])
            p99 = np.nanpercentile(sample_data[~np.isnan(sample_data)], 99.9)
            del sample_data
            
            # 逐帧应用 P99
            for i in tqdm(range(N), desc="Applying P99 Removal"):
                frame = volume[i]
                mask = frame > p99
                if mask.any():
                    frame[mask] = np.nan
                    volume[i] = frame
                    nodata_mask[i] = nodata_mask[i] | np.isnan(frame).any(axis=-1)
                
            if self.use_memmap:
                volume.flush()
                nodata_mask.flush()
        
        gc.collect()
        return volume, nodata_mask, meta_info
    
    def _temporal_binning(
        self,
        raw_valid_files: List[dict],
        temporal_bin_days: int
    ) -> List[dict]:
        """时间窗口聚合"""
        print(f"[Loader] Binning data into {temporal_bin_days}-day windows...")
        binned_files = []
        current_bin = []
        bin_start_time = raw_valid_files[0]["date"]
        
        for f_info in raw_valid_files:
            if f_info["date"] < bin_start_time + timedelta(days=temporal_bin_days):
                current_bin.append(f_info)
            else:
                mid_date = bin_start_time + timedelta(days=temporal_bin_days / 2)
                binned_files.append({"date": mid_date, "files": current_bin})
                bin_start_time = f_info["date"]
                current_bin = [f_info]
        
        if current_bin:
            binned_files.append({"date": bin_start_time, "files": current_bin})
        
        return binned_files


def load_dataset(
    folder_path: str,
    expected_bands: int = 7,
    source_band_indices: Optional[List[int]] = None,
    source_band_names: Optional[List[str]] = None,
    scale_factor: float = 0.0001,
    valid_range: Tuple[float, float] = DEFAULT_VALID_RANGE,
    global_start_date: Optional[datetime] = None,
    global_end_date: Optional[datetime] = None,
    temporal_bin_days: Optional[int] = None,
    valid_threshold: float = VALID_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """便捷函数：加载数据集"""
    loader = DatasetLoader()
    return loader.load_folder(
        folder_path=folder_path,
        expected_bands=expected_bands,
        source_band_indices=source_band_indices,
        source_band_names=source_band_names,
        scale_factor=scale_factor,
        valid_range=valid_range,
        global_start_date=global_start_date,
        global_end_date=global_end_date,
        temporal_bin_days=temporal_bin_days,
        valid_threshold=valid_threshold,
    )
