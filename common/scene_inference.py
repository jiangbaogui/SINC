"""Generic raster I/O and full-frame SINC inference helpers."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch

from common.constants import DEFAULT_VALID_RANGE, VALID_THRESHOLD
from ground.core.NDCUtils import normalize_time


def read_observation(
    path,
    expected_bands=None,
    source_band_indices=None,
    source_band_names=None,
):
    """Read one reflectance cube and return its spatial metadata."""

    with rasterio.open(path) as source:
        selected_indices = (
            list(range(1, source.count + 1))
            if source_band_indices is None
            else [int(value) for value in source_band_indices]
        )
        if expected_bands is not None and len(selected_indices) != expected_bands:
            if source_band_indices is None:
                raise ValueError(
                    f"Raster contains {source.count} bands but the model expects "
                    f"{expected_bands}; provide an explicit source_band_indices mapping"
                )
            raise ValueError(
                f"Expected {expected_bands} selected bands, got "
                f"{len(selected_indices)}"
            )
        if not selected_indices or min(selected_indices) < 1:
            raise ValueError("source_band_indices must be one-based positive integers")
        if max(selected_indices) > source.count:
            raise ValueError(
                f"Requested source band {max(selected_indices)} but {path} contains "
                f"only {source.count} bands"
            )
        if source_band_names and any(source.descriptions):
            actual_names = [source.descriptions[index - 1] for index in selected_indices]
            if actual_names != list(source_band_names):
                raise ValueError(
                    f"Configured source bands {list(source_band_names)} do not match "
                    f"GeoTIFF descriptions {actual_names} in {path}"
                )
        observation = (
            source.read(indexes=selected_indices)
            .transpose(1, 2, 0)
            .astype(np.float32)
        )
        profile = source.profile.copy()
        profile["count"] = len(selected_indices)
        transform = source.transform
        crs = source.crs
    finite = observation[np.isfinite(observation)]
    if finite.size and float(np.max(finite)) > 10.0:
        observation *= 0.0001
    return observation, profile, transform, crs


def read_observation_group(
    paths,
    expected_bands=None,
    source_band_indices=None,
    source_band_names=None,
):
    """Read one daily observation, compositing same-date rasters by nanmedian."""

    if isinstance(paths, (str, Path)):
        paths = (Path(paths),)
    else:
        paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError("At least one raster is required for a daily observation")

    arrays = []
    reference_profile = None
    reference_transform = None
    reference_crs = None
    for path in paths:
        observation, profile, transform, crs = read_observation(
            path,
            expected_bands=expected_bands,
            source_band_indices=source_band_indices,
            source_band_names=source_band_names,
        )
        if reference_profile is None:
            reference_profile = profile
            reference_transform = transform
            reference_crs = crs
        else:
            aligned = (
                observation.shape == arrays[0].shape
                and profile["height"] == reference_profile["height"]
                and profile["width"] == reference_profile["width"]
                and transform.almost_equals(reference_transform)
                and crs == reference_crs
            )
            if not aligned:
                raise ValueError(f"Same-date raster is not aligned with {paths[0]}: {path}")

        valid = np.isfinite(observation)
        valid &= observation > VALID_THRESHOLD
        valid &= observation <= DEFAULT_VALID_RANGE[1]
        arrays.append(np.where(valid, observation, np.nan))

    if len(arrays) == 1:
        composite = arrays[0]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            composite = np.nanmedian(np.stack(arrays, axis=0), axis=0)
    return (
        composite.astype(np.float32, copy=False),
        reference_profile,
        reference_transform,
        reference_crs,
    )


def _metadata_datetime(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def predict_frame_with_state(inference, target_datetime, height, width):
    """Return mean, uncertainty, dynamic state, and local state variation."""

    start = _metadata_datetime(inference.meta["global_start_date"])
    end = _metadata_datetime(inference.meta["global_end_date"])
    if target_datetime.tzinfo is not None:
        target_datetime = target_datetime.replace(tzinfo=None)
    t_norm = normalize_time(target_datetime, start, end)

    coords = inference._create_coords_grid(height, width)
    chunk_size = inference.chunk_size
    n_points = height * width
    n_bands = len(inference.band_indices)
    mean = np.zeros((n_points, n_bands), dtype=np.float32)
    std = np.zeros((n_points, n_bands), dtype=np.float32)
    dynamic_state = np.zeros((n_points, n_bands), dtype=np.float32)
    dynamic_delta = np.zeros((n_points, n_bands), dtype=np.float32)

    raw_model = inference.model.model
    raw_model.eval()
    with torch.no_grad():
        for first in range(0, n_points, chunk_size):
            last = min(first + chunk_size, n_points)
            t_input = torch.full(
                (last - first, 1), t_norm, device=inference.device
            )
            pred_mean, log_var, _, delta_terms, state_correction = (
                raw_model.forward_with_state(coords[first:last], t_input)
            )
            pred_std = torch.exp(0.5 * torch.clamp(log_var, min=-7.0, max=5.0))
            mean[first:last] = pred_mean.float().cpu().numpy()
            std[first:last] = pred_std.float().cpu().numpy()
            dynamic_state[first:last] = state_correction.float().cpu().numpy()
            if delta_terms:
                dynamic_delta[first:last] = delta_terms[0].float().cpu().numpy()

    data_min, data_max = inference.meta.get("data_range", [0.0, 1.0])
    data_scale = data_max - data_min
    mean = mean * data_scale + data_min
    std *= data_scale
    dynamic_state *= data_scale
    dynamic_delta *= data_scale
    if inference.meta.get("use_log", False):
        mean = np.expm1(mean)

    return (
        mean.reshape(height, width, n_bands),
        std.reshape(height, width, n_bands),
        dynamic_state.reshape(height, width, n_bands),
        dynamic_delta.reshape(height, width, n_bands),
    )


def predict_frame_with_dynamic(inference, target_datetime, height, width):
    """Compatibility wrapper returning local dynamic-state variation."""

    mean, std, _, dynamic_delta = predict_frame_with_state(
        inference, target_datetime, height, width
    )
    return mean, std, dynamic_delta
