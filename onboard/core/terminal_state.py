"""Identify the most recent stable state represented by the dynamic branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class TerminalStateSettings:
    """Label-free settings for selecting post-transition calibration samples."""

    transition_multiplier: float = 3.0
    transition_guard_frames: int = 1
    minimum_stable_observations: int = 3
    minimum_stable_days: int = 30
    minimum_state_change: float = 1e-4

    def to_dict(self) -> dict:
        return asdict(self)


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    """Call nanmedian without emitting warnings for entirely invalid pixels."""

    with np.errstate(invalid="ignore"):
        return np.nanmedian(values, axis=axis)


def terminal_state_masks(
    dynamic_states: np.ndarray,
    valid_masks: np.ndarray,
    acquisition_dates: list[datetime] | np.ndarray | None = None,
    settings: TerminalStateSettings | None = None,
    *,
    selection_method: str = "date_normalized_dynamic_state_transition",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select observations after each pixel's latest learned state transition.

    A transition response is the mean absolute state-change rate between adjacent
    acquisitions, rescaled to the median acquisition interval. For each pixel, a robust
    threshold is estimated from its temporal response series as median + three
    scaled MADs by default. The final calibration mask starts after the latest
    threshold exceedance, applies a short guard interval, and retains only pixels
    with enough post-transition observations.
    """

    settings = settings or TerminalStateSettings()
    states = np.asarray(dynamic_states, dtype=np.float32)
    valid = np.asarray(valid_masks, dtype=bool)
    if states.ndim != 4:
        raise ValueError("dynamic_states must have shape (time, height, width, bands)")
    if valid.shape != states.shape[:3]:
        raise ValueError(
            f"valid_masks shape {valid.shape} does not match {states.shape[:3]}"
        )
    if states.shape[0] < 4:
        raise ValueError("At least four dated state estimates are required")
    if settings.transition_multiplier <= 0:
        raise ValueError("transition_multiplier must be positive")
    if settings.transition_guard_frames < 0:
        raise ValueError("transition_guard_frames cannot be negative")
    if settings.minimum_stable_observations < 2:
        raise ValueError("minimum_stable_observations must be at least two")
    if settings.minimum_stable_days < 0:
        raise ValueError("minimum_stable_days cannot be negative")

    time_count, height, width, _ = states.shape
    if acquisition_dates is None:
        elapsed_days = np.ones(time_count - 1, dtype=np.float32)
        ordinal_days = np.arange(time_count, dtype=np.float32)
        date_handling = "unit_intervals_fallback"
    else:
        dates = np.asarray(acquisition_dates, dtype="datetime64[D]")
        if dates.shape != (time_count,):
            raise ValueError(
                "acquisition_dates must contain one date for each state estimate"
            )
        elapsed_days = np.diff(dates).astype("timedelta64[D]").astype(np.float32)
        if np.any(elapsed_days <= 0):
            raise ValueError("acquisition_dates must be strictly increasing")
        ordinal_days = (dates - dates[0]).astype("timedelta64[D]").astype(np.float32)
        date_handling = "actual_elapsed_days"
    reference_interval_days = float(np.median(elapsed_days))

    state_change = np.full((time_count, height, width), np.nan, dtype=np.float32)
    state_change[0] = 0.0
    pair_valid = valid[1:] & valid[:-1]
    adjacent_change = np.mean(np.abs(states[1:] - states[:-1]), axis=-1)
    interval_scaled_change = (
        adjacent_change
        / elapsed_days[:, None, None]
        * reference_interval_days
    )
    state_change[1:] = np.where(pair_valid, interval_scaled_change, np.nan)

    change_history = state_change[1:]
    center = _nanmedian(change_history, axis=0)
    mad = _nanmedian(np.abs(change_history - center[None, ...]), axis=0)
    robust_sigma = 1.4826 * mad
    finite_history = np.sum(np.isfinite(change_history), axis=0)
    transition_threshold = center + settings.transition_multiplier * np.maximum(
        robust_sigma, settings.minimum_state_change
    )
    transition_threshold[finite_history < 2] = np.nan

    transitions = (
        np.isfinite(state_change)
        & np.isfinite(transition_threshold)[None, ...]
        & (state_change > transition_threshold[None, ...])
    )
    frame_indices = np.arange(time_count, dtype=np.int32)[:, None, None]
    last_transition = np.max(
        np.where(transitions, frame_indices, -1), axis=0
    ).astype(np.int32)

    # A transition at frame i lies between i-1 and i. With a one-frame guard,
    # calibration begins at i+1. Pixels without a detected transition begin at 0.
    stable_start = np.where(
        last_transition >= 0,
        last_transition + settings.transition_guard_frames,
        0,
    )
    post_transition = frame_indices >= stable_start[None, ...]
    stable_masks = valid & post_transition & ~transitions
    stable_count = np.sum(stable_masks, axis=0)
    first_stable_day = np.min(
        np.where(stable_masks, ordinal_days[:, None, None], np.inf), axis=0
    )
    last_stable_day = np.max(
        np.where(stable_masks, ordinal_days[:, None, None], -np.inf), axis=0
    )
    stable_duration_days = last_stable_day - first_stable_day
    enough_history = (
        stable_count >= settings.minimum_stable_observations
    ) & (stable_duration_days >= settings.minimum_stable_days)
    stable_masks &= enough_history[None, ...]

    valid_pixel_count = int(np.count_nonzero(np.any(valid, axis=0)))
    retained_pixel_count = int(np.count_nonzero(enough_history))
    pixels_with_transition = int(np.count_nonzero(last_transition >= 0))
    audit = {
        "selection_method": selection_method,
        "settings": settings.to_dict(),
        "date_handling": date_handling,
        "reference_interval_days": reference_interval_days,
        "frame_count": int(time_count),
        "valid_pixel_count": valid_pixel_count,
        "retained_pixel_count": retained_pixel_count,
        "retained_pixel_fraction": (
            float(retained_pixel_count / valid_pixel_count)
            if valid_pixel_count
            else 0.0
        ),
        "pixels_with_detected_transition": pixels_with_transition,
        "detected_transition_fraction": (
            float(pixels_with_transition / valid_pixel_count)
            if valid_pixel_count
            else 0.0
        ),
        "median_transition_threshold": float(
            np.nanmedian(transition_threshold)
        ),
        "median_post_transition_observations": float(
            np.median(stable_count[enough_history])
        ) if retained_pixel_count else 0.0,
        "median_post_transition_days": float(
            np.median(stable_duration_days[enough_history])
        ) if retained_pixel_count else 0.0,
        "stable_samples_by_frame": [
            int(np.count_nonzero(stable_masks[index]))
            for index in range(time_count)
        ],
    }
    return stable_masks, state_change, audit


def observation_state_masks(
    observations: np.ndarray,
    valid_masks: np.ndarray,
    acquisition_dates: list[datetime] | np.ndarray,
    settings: TerminalStateSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Derive model-neutral stable support from the observed reflectance history.

    This support is intended for controlled comparisons and ablations. It avoids
    allowing any evaluated model, including SINC, to define which pixels are
    available to its competitors.
    """

    return terminal_state_masks(
        observations,
        valid_masks,
        acquisition_dates,
        settings,
        selection_method="date_normalized_observed_reflectance_transition",
    )
