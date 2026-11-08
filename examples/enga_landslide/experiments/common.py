"""Shared case definitions and I/O helpers for manuscript experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from sklearn.metrics import confusion_matrix

from common.scene_inference import (
    predict_frame_with_dynamic,
    predict_frame_with_state,
    read_observation,
    read_observation_group,
)
from common.constants import DEFAULT_VALID_RANGE, VALID_THRESHOLD


BASE_DATA_DIR = Path(os.environ.get("SINC_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))
MODEL_OUTPUT_DIR = BASE_DATA_DIR / "output" / "revised_v3"

CASES = {
    "Flood": {
        "model": MODEL_OUTPUT_DIR / "flood_weights.gndc",
        "train": BASE_DATA_DIR / "flood_S" / "train",
        "image": BASE_DATA_DIR / "flood_S" / "yichang" / "S2_Beijing_2025-07-30_0316.tif",
        "labels": BASE_DATA_DIR / "test" / "FLODD_8.6.CSV",
    },
    "JZ_Fire": {
        "model": MODEL_OUTPUT_DIR / "JZ_Fire_weights.gndc",
        "train": BASE_DATA_DIR / "JZ_Fire" / "train",
        "image": BASE_DATA_DIR / "JZ_Fire" / "yichang" / "S2_JZ_2021-09-11_1903.tif",
        "labels": BASE_DATA_DIR / "test" / "Fire_2021.9.11.csv",
    },
    "Earthquake": {
        "model": MODEL_OUTPUT_DIR / "earthquake_weights.gndc",
        "train": BASE_DATA_DIR / "TE_earthquake" / "train",
        "image": BASE_DATA_DIR / "TE_earthquake" / "yichang" / "S2_TE_2021-02-24_0829.tif",
        "labels": BASE_DATA_DIR / "test" / "earthquake.csv",
    },
    "Landslide": {
        "model": MODEL_OUTPUT_DIR / "Huapo_weights.gndc",
        "train": BASE_DATA_DIR / "Huapo" / "train",
        "image": BASE_DATA_DIR / "Huapo" / "yichang" / "S2_JN_2024-05-28_0048 .tif",
        "labels": BASE_DATA_DIR / "test" / "huopo_2024.5.28.csv",
    },
    "Hurricane": {
        "model": MODEL_OUTPUT_DIR / "jufeng_weights.gndc",
        "train": BASE_DATA_DIR / "jufeng" / "train",
        "image": BASE_DATA_DIR / "jufeng" / "yichang" / "S2_Mcg_2024-10-24_1615.tif",
        "labels": BASE_DATA_DIR / "test" / "jufeng_2024.10.24_1.csv",
    },
    "Wildfire": {
        "model": MODEL_OUTPUT_DIR / "wildfire_weights.gndc",
        "train": BASE_DATA_DIR / "wildfire" / "train",
        "image": BASE_DATA_DIR / "wildfire" / "yichang" / "S2_Sda_2025-01-12_1844.tif",
        "labels": BASE_DATA_DIR / "test" / "wildfire_2025.1.12.csv",
    },
    "Forest_Harvest_1": {
        "model": MODEL_OUTPUT_DIR / "forest_kan_weights.gndc",
        "train": BASE_DATA_DIR / "forest_kan" / "train",
        "image": BASE_DATA_DIR / "forest_kan" / "yichang" / "S2_China_2020-08-27_0310.tif",
        "labels": BASE_DATA_DIR / "test" / "kanfa_2020827.csv",
    },
    "Forest_Harvest_2": {
        "model": MODEL_OUTPUT_DIR / "forest_kanfa1_weights.gndc",
        "train": BASE_DATA_DIR / "forest_kanfa1" / "train",
        "image": BASE_DATA_DIR / "forest_kanfa1" / "yichang" / "S2_Kakinada_2020-03-16_0514.tif",
        "labels": BASE_DATA_DIR / "test" / "kanfa1_2020_316.csv",
    },
}


# Display names used consistently in manuscript figures and result tables.
MANUSCRIPT_SCENE_LABELS = {
    "Flood": "Flood",
    "JZ_Fire": "Vegetation Fire",
    "Earthquake": "Earthquake",
    "Landslide": "Landslide",
    "Hurricane": "Hurricane",
    "Wildfire": "Wildfire",
    "Forest_Harvest_1": "Deforestation 1",
    "Forest_Harvest_2": "Deforestation 2",
}


def compact_cache_root(owner: str | Path, namespace: str) -> Path:
    """Return a deterministic cache root that stays below Windows MAX_PATH."""

    owner = Path(os.path.abspath(owner))
    local_root = owner / f"_{namespace}"
    if os.name != "nt" or len(str(local_root)) <= 180:
        return local_root

    digest = hashlib.sha1(str(owner).casefold().encode("utf-8")).hexdigest()[:10]
    configured_root = os.environ.get("SINC_CACHE_ROOT")
    drive_root = Path(owner.anchor) / "_SINC_CACHE"
    if configured_root:
        return Path(configured_root) / digest / namespace
    return drive_root / digest / namespace


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    false_alarm = fp / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": int(tp + tn + fp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "false_alarm": float(false_alarm),
        "f1": float(f1),
    }


def sample_mask_at_labels(labels_path, mask, transform, crs):
    frame = pd.read_csv(labels_path)
    columns = {column.upper(): column for column in frame.columns}
    lon_column = columns.get("LON") or columns.get("X")
    lat_column = columns.get("LAT") or columns.get("Y")
    label_column = columns.get("GT_LABEL") or columns.get("LABEL")
    if not lon_column or not lat_column or not label_column:
        raise ValueError(f"Required label columns are missing from {labels_path}")

    y_true = []
    y_pred = []
    height, width = mask.shape
    for _, row in frame.iterrows():
        coordinate_values = np.asarray(
            [row[lon_column], row[lat_column]], dtype=np.float64
        )
        if not np.all(np.isfinite(coordinate_values)) or pd.isna(row[label_column]):
            continue
        if crs and crs.to_epsg() != 4326:
            xs, ys = transform_coordinates(
                "EPSG:4326", crs, [coordinate_values[0]], [coordinate_values[1]]
            )
            x, y = xs[0], ys[0]
        else:
            x, y = coordinate_values
        raster_row, raster_col = rasterio.transform.rowcol(transform, x, y)
        if 0 <= raster_row < height and 0 <= raster_col < width:
            y_true.append(int(row[label_column]))
            y_pred.append(int(mask[raster_row, raster_col]))
    return np.asarray(y_true), np.asarray(y_pred)


def save_raster(path, array, source_profile, dtype, nodata=None):
    profile = source_profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate")
    if nodata is not None:
        profile.update(nodata=nodata)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(dtype), 1)


def valid_observation_mask(observation):
    finite = np.all(np.isfinite(observation), axis=-1)
    in_range = np.all(
        (observation > VALID_THRESHOLD)
        & (observation <= DEFAULT_VALID_RANGE[1]),
        axis=-1,
    )
    return finite & in_range


def source_paths_from_record(record) -> tuple[Path, ...]:
    """Return all source rasters represented by one manifest observation."""

    raw_paths = record.get("source_paths")
    if isinstance(raw_paths, str) and raw_paths.strip():
        parsed = json.loads(raw_paths)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("source_paths must encode a non-empty JSON list")
        return tuple(Path(str(path)) for path in parsed)
    return (Path(str(record["path"])),)


def make_rgb(observation):
    rgb = observation[..., [2, 1, 0]].astype(np.float32)
    valid = np.all(np.isfinite(rgb), axis=-1) & np.any(rgb > 0, axis=-1)
    if np.any(valid):
        low, high = np.percentile(rgb[valid], [2, 98])
        if high > low:
            rgb = (rgb - low) / (high - low)
    return np.clip(rgb, 0.0, 1.0)
