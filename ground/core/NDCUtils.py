"""
工具函数模块 - 压缩、日期处理、参数建议
"""
import re
import math
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Iterable, Optional

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    import zlib


def fast_compress(data_bytes: bytes) -> bytes:
    """快速压缩"""
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(data_bytes)
    return zlib.compress(data_bytes, level=6)


def fast_decompress(data_bytes: bytes) -> bytes:
    """快速解压"""
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data_bytes)
    return zlib.decompress(data_bytes)


def extract_date_from_filename(
    filename: str,
    default_hour: int = 10,
    default_minute: int = 30
) -> Optional[datetime]:
    """从文件名提取日期"""
    
    # 1. 匹配带横杠并以底杠结尾的格式: 2022-07-13_
    match_flood = re.search(r"(\d{4})-(\d{2})-(\d{2})_", filename)
    if match_flood:
        year, month, day = match_flood.groups()
        return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").replace(
            hour=default_hour, minute=default_minute
        )
    
    # 2. 匹配标准的带横杠格式: 2022-07-13
    match_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if match_iso:
        return datetime.strptime(match_iso.group(0), "%Y-%m-%d").replace(
            hour=default_hour, minute=default_minute
        )
        
    # 3. 🚀 [新增] 匹配连续的 8 位数字格式 (例如: GeoDNC_N36E113_20240102.tif)
    # 寻找以下划线开头，后面跟着 8 位数字，并以 . 或 _ 结尾的字符串
    match_compact = re.search(r"_(\d{4})(\d{2})(\d{2})(?:\.|\_|$)", filename)
    if not match_compact:
        # 兜底：直接寻找任意纯 8 位数字 (限制 19xx 或 20xx 开头，避免误伤经纬度坐标)
        match_compact = re.search(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)", filename)
        
    if match_compact:
        year, month, day = match_compact.groups()
        try:
            return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").replace(
                hour=default_hour, minute=default_minute
            )
        except ValueError:
            pass  # 如果提取到的 8 位数字不是合法日期 (比如月份是 99)，则忽略并返回 None

    return None


def group_dated_paths_by_day(
    paths: Iterable[str | Path],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    end_exclusive: Optional[datetime] = None,
) -> list[tuple[datetime, tuple[Path, ...]]]:
    """Group dated raster paths into deterministic acquisition-day observations."""

    grouped: dict[date_type, dict[str, object]] = {}
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        acquisition = extract_date_from_filename(raw_path.name)
        if acquisition is None:
            continue
        if start is not None and acquisition < start:
            continue
        if end is not None and acquisition > end:
            continue
        if end_exclusive is not None and acquisition >= end_exclusive:
            continue
        day = acquisition.date()
        entry = grouped.setdefault(day, {"date": acquisition, "paths": []})
        entry["paths"].append(raw_path)

    return [
        (entry["date"], tuple(entry["paths"]))
        for _, entry in sorted(grouped.items(), key=lambda item: item[0])
    ]


def normalize_time(
    dt: datetime,
    global_start_date: datetime,
    global_end_date: datetime
) -> float:
    """归一化时间到[0, 1]范围"""
    if not isinstance(dt, datetime):
        raise TypeError("Input 'dt' must be a datetime object.")
    
    delta = (dt - global_start_date).total_seconds()
    total_seconds = (global_end_date - global_start_date).total_seconds()
    
    if total_seconds <= 0:
        raise ValueError("Global end date must be after global start date.")
    
    return max(0.0, delta / total_seconds)


def suggest_ndc_config(
    width: int,
    height: int,
    num_timestamps: int,
    base_resolution: int = 32,
    n_levels: int = 16,
    gpu_mem_gb: int = 24,
    force_xy_hash: Optional[int] = None,
    force_t_hash: Optional[int] = None
) -> dict:
    """参数自动建议器"""
    print(f"\n=== GeoNDC Parameter Optimizer ===")
    print(f"Target: {width}x{height} pixels, {num_timestamps} frames.")
    
    target_res = max(width, height) * 1.5
    scale = (target_res / base_resolution) ** (1 / (n_levels - 1))
    
    if force_xy_hash is not None:
        final_xy = force_xy_hash
        print(f"[Config] Using manual XY Hash Size: 2^{final_xy}")
    else:
        total_spatial_pixels = width * height
        ideal_log2 = math.ceil(math.log2(total_spatial_pixels)) + 2
        
        if gpu_mem_gb >= 24:
            max_safe_log2 = 22
        elif gpu_mem_gb >= 12:
            max_safe_log2 = 20
        elif gpu_mem_gb >= 8:
            max_safe_log2 = 19
        else:
            max_safe_log2 = 18
        
        base_requirement = 17
        final_xy = int(min(max(base_requirement, ideal_log2), max_safe_log2))
    
    if force_t_hash is not None:
        final_t = force_t_hash
        print(f"[Config] Using manual T Hash Size: 2^{final_t}")
    else:
        final_t = max(16, int(math.ceil(math.log2(width * num_timestamps))))
    
    decoder_hidden = 128 if final_xy >= 20 else 64
    
    config = {
        "n_levels": n_levels,
        "base_resolution": base_resolution,
        "per_level_scale": round(scale, 4),
        "xy_hash_size": final_xy,
        "t_hash_size": final_t,
        "use_transformer": False,
        "decoder_hidden": decoder_hidden,
        "features_per_level": 2
    }
    
    print(f" -> Result: Scale={config['per_level_scale']}")
    print(f" -> Hash Table: XY=2^{final_xy}, T=2^{final_t}")
    return config


def calculate_end_date(
    start_date_str: str,
    base_res: int,
    n_levels: int,
    scale: float,
    days_per_cell: float = 1.0
) -> str:
    """计算时间覆盖结束日期"""
    n_max = int(base_res * (scale ** (n_levels - 1)))
    required_days = n_max * days_per_cell
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=required_days)
    
    print(f"--- Temporal Coverage Report ---")
    print(f"Max Resolution: {n_max} cells")
    print(f"Start: {start_date_str} -> Capacity End: {end_date.strftime('%Y-%m-%d')}")
    return end_date.strftime('%Y-%m-%d')


if __name__ == "__main__":
    print("--- Applying Experimental Parameters ---")
    #flood
    # my_config = suggest_ndc_config(
    #     width=280,
    #     height=280,
    #     num_timestamps=210,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2022-07-13",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=2.6
    # )
    #shuihua
    # my_config = suggest_ndc_config(
    #     width=465,
    #     height=465,
    #     num_timestamps=83,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2022-04-05",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=1.5
    # )
     #JZ_Fire"
    # my_config = suggest_ndc_config(
    #     width=279,
    #     height=279,
    #     num_timestamps=106,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2018-07-14",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=2.65
    # )
     #earthquake"
    # my_config = suggest_ndc_config(
    #     width=280,
    #     height=280,
    #     num_timestamps=120,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2018-03-12",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=2.5
    # )
  #qixuan"
    # my_config = suggest_ndc_config(
    #     width=268,
    #     height=268,
    #     num_timestamps=96,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2020-11-06",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=4.55
    # )
      #huapo"
    # my_config = suggest_ndc_config(
    #     width=280,
    #     height=279,
    #     num_timestamps=121,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2020-06-06",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=3.5
    # )
    #wildfire
    # my_config = suggest_ndc_config(
    #     width=279,
    #     height=279,
    #     num_timestamps=106,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2021-10-10",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=2.6
    # )
    #kanfa
    # my_config = suggest_ndc_config(
    #     width=279,
    #     height=279,
    #     num_timestamps=42,
    #     base_resolution=16,
    #     n_levels=12,
    #     gpu_mem_gb=24,
    #     force_xy_hash=16,
    #     force_t_hash=16
    # )
    
    # calculate_end_date(
    #     start_date_str="2017-02-14",
    #     base_res=my_config['base_resolution'],
    #     n_levels=my_config['n_levels'],
    #     scale=my_config['per_level_scale'],
    #     days_per_cell=2.9
    # )
    #kanfa1
    my_config = suggest_ndc_config(
        width=279,
        height=279,
        num_timestamps=109,
        base_resolution=16,
        n_levels=12,
        gpu_mem_gb=24,
        force_xy_hash=16,
        force_t_hash=16
    )
    
    calculate_end_date(
        start_date_str="2017-03-02",
        base_res=my_config['base_resolution'],
        n_levels=my_config['n_levels'],
        scale=my_config['per_level_scale'],
        days_per_cell=2.6
    )
