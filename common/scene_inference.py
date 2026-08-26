"""Generic raster I/O and full-frame SINC inference helpers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import rasterio
import torch

from ground.core.NDCUtils import normalize_time


def read_observation(path, expected_bands=None):
    """Read one reflectance cube and return its spatial metadata."""

    with rasterio.open(path) as source:
        observation = source.read().transpose(1, 2, 0).astype(np.float32)
        profile = source.profile.copy()
        transform = source.transform
        crs = source.crs
    if np.nanmax(observation) > 10:
        observation *= 0.0001
    if expected_bands is not None:
        observation = observation[..., :expected_bands]
    return observation, profile, transform, crs


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
