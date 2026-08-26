"""Utilities for calibrating empirical-null thresholds from normal history."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ground.core.NDCUtils import extract_date_from_filename
from onboard.core.empirical_null import (
    DEFAULT_NULL_TRIM_ALPHA,
    DEFAULT_REFERENCE_QUANTILE,
    EmpiricalNullThresholds,
    apply_empirical_null_thresholds,
    calculate_detection_maps,
    estimate_empirical_null_thresholds,
)
from onboard.core.terminal_state import (
    TerminalStateSettings,
    observation_state_masks,
    terminal_state_masks,
)


DEFAULT_ALPHA_CANDIDATES = (
    0.01,
    0.005,
    0.001,
    0.0005,
    0.0001,
    0.00005,
    0.00001,
)
DEFAULT_TARGET_NORMAL_FAR = 0.01
DEFAULT_DATE_FAR_QUANTILE = 0.95
DEFAULT_MAX_SAMPLES_PER_FRAME = 12000
DEFAULT_MIN_VALID_FRACTION = 0.95
DEFAULT_MIN_STABLE_PIXELS_PER_FRAME = 1000
DEFAULT_TRANSITION_MULTIPLIER = 3.0
DEFAULT_TRANSITION_GUARD_FRAMES = 1
DEFAULT_MINIMUM_STABLE_OBSERVATIONS = 3
DEFAULT_MINIMUM_STABLE_DAYS = 30
DEFAULT_MINIMUM_STATE_CHANGE = 1e-4


def dated_images_before(
    directory: str | Path,
    target_date: datetime,
) -> list[tuple[datetime, Path]]:
    """Return one raster per acquisition date strictly before the target date."""

    by_date: dict = {}
    for path in sorted(Path(directory).rglob("*.tif")):
        date = extract_date_from_filename(path.name)
        if date is not None and date < target_date:
            by_date.setdefault(date.date(), (date, path))
    return sorted(by_date.values(), key=lambda item: item[0])


def split_calibration_validation_frames(
    images: Iterable[tuple[datetime, Path]],
) -> tuple[list[tuple[datetime, Path]], list[tuple[datetime, Path]]]:
    """Create deterministic, temporally interleaved normal calibration/validation sets."""

    images = list(images)
    if len(images) < 4:
        raise ValueError("At least four normal-reference dates are required for alpha testing")
    validation_indices = set(range(2, len(images), 3))
    if not validation_indices:
        validation_indices = {len(images) - 1}
    calibration = [item for index, item in enumerate(images) if index not in validation_indices]
    validation = [item for index, item in enumerate(images) if index in validation_indices]
    return calibration, validation


def filter_stable_reference_frames(
    frames: Iterable[dict],
    *,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    minimum_stable_pixels: int = DEFAULT_MIN_STABLE_PIXELS_PER_FRAME,
    minimum_frames: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Retain dates using only data coverage and terminal-state support.

    Residual magnitude and learned state-change magnitude are deliberately not
    used to reject dates. This prevents difficult but valid normal dates from
    being removed before the false-alarm audit.
    """

    frames = list(frames)
    if len(frames) < 4:
        raise ValueError("At least four reference frames are required")
    accepted = []
    audit = []
    for frame in frames:
        scene_valid = np.asarray(frame["valid"], dtype=bool)
        terminal_valid = np.asarray(
            frame.get("calibration_valid", frame["valid"]), dtype=bool
        )
        stable_count = int(np.count_nonzero(terminal_valid))
        valid_fraction = float(np.mean(scene_valid))
        stable_fraction = float(np.mean(terminal_valid))
        reasons = []
        if valid_fraction < min_valid_fraction:
            reasons.append("insufficient_valid_coverage")
        if stable_count < int(minimum_stable_pixels):
            reasons.append("insufficient_terminal_stable_support")
        record = {
            "date": frame["date"].isoformat(),
            "image": str(frame["path"]),
            "valid_fraction": valid_fraction,
            "terminal_stable_pixel_count": stable_count,
            "terminal_stable_fraction": stable_fraction,
            "minimum_valid_fraction": float(min_valid_fraction),
            "minimum_stable_pixels": int(minimum_stable_pixels),
            "response_values_used_for_date_selection": False,
            "accepted": not reasons,
            "exclusion_reasons": reasons,
        }
        audit.append(record)
        if not reasons:
            accepted.append(frame)

    if len(accepted) < int(minimum_frames):
        raise ValueError(
            f"Only {len(accepted)} reference dates satisfy the prespecified "
            f"coverage and terminal-state support criteria; at least "
            f"{minimum_frames} are required. Audit: {audit}"
        )
    return accepted, audit


def collect_frame_distributions(
    images: Iterable[tuple[datetime, Path]],
    predictor: Callable,
    detector,
    read_observation: Callable,
    *,
    expected_bands: int,
    random_seed: int,
    max_samples_per_frame: int = DEFAULT_MAX_SAMPLES_PER_FRAME,
    skip_frames_with_too_few_valid: bool = False,
    retain_observation: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Predict reference dates and retain sampled plus per-frame score distributions.

    The predictor receives ``(date, path, observation)`` and returns
    ``(predicted_mean, predicted_std, state_information, diagnostics)``.
    ``state_information`` should contain the band-specific ``state`` correction
    and may contain its local ``velocity``. A legacy numeric dynamic map remains
    accepted for callers that do not perform terminal-state selection.
    """

    rng = np.random.default_rng(random_seed)
    frames = []
    records = []
    for frame_index, (date, path) in enumerate(images, start=1):
        observation, _, _, _ = read_observation(path, expected_bands=expected_bands)
        predicted_mean, predicted_std, state_information, diagnostics = predictor(
            date, path, observation
        )
        d2, cfar, sam, valid = calculate_detection_maps(
            observation, predicted_mean, predicted_std, detector
        )
        dynamic_state = None
        dynamic = None
        if isinstance(state_information, dict):
            dynamic_state = state_information.get("state")
            dynamic = state_information.get("velocity")
            if dynamic_state is not None:
                dynamic_state = np.asarray(dynamic_state, dtype=np.float32)
                if dynamic_state.ndim != 3 or dynamic_state.shape[:2] != valid.shape:
                    raise ValueError(
                        "Band-specific dynamic state must have shape "
                        f"(height, width, bands), got {dynamic_state.shape}"
                    )
                valid &= np.all(np.isfinite(dynamic_state), axis=-1)
        else:
            dynamic = state_information
        if dynamic is not None:
            dynamic = np.asarray(dynamic, dtype=np.float32)
            if dynamic.ndim == 3:
                dynamic = np.mean(np.abs(dynamic), axis=-1)
            if dynamic.shape != valid.shape:
                raise ValueError(
                    f"Dynamic map shape {dynamic.shape} does not match {valid.shape}"
                )
            valid &= np.isfinite(dynamic)

        valid_indices = np.flatnonzero(valid.ravel())
        if valid_indices.size < 100:
            if not skip_frames_with_too_few_valid:
                raise RuntimeError(f"Too few valid calibration pixels in {path}")
            records.append(
                {
                    "date": date.isoformat(),
                    "image": str(path),
                    "valid_pixel_count": int(valid_indices.size),
                    "sample_count": 0,
                    "valid_fraction": float(np.mean(valid)),
                    "status": "skipped_model_not_initialized",
                    "prediction": diagnostics,
                }
            )
            print(
                f"  Skipped normal reference {frame_index:02d}: {path.name} "
                f"({valid_indices.size} valid model predictions)"
            )
            continue
        if valid_indices.size > max_samples_per_frame:
            valid_indices = rng.choice(
                valid_indices, size=max_samples_per_frame, replace=False
            )

        sampled = {
            "d2": d2.ravel()[valid_indices].astype(np.float32),
            "cfar": cfar.ravel()[valid_indices].astype(np.float32),
            "sam": sam.ravel()[valid_indices].astype(np.float32),
        }
        if dynamic is not None:
            sampled["dynamic"] = dynamic.ravel()[valid_indices].astype(np.float32)

        frames.append(
            {
                "date": date,
                "path": path,
                "d2": d2,
                "cfar": cfar,
                "sam": sam,
                "valid": valid,
                "dynamic": dynamic,
                "dynamic_state": dynamic_state,
                "observation": (
                    np.asarray(observation, dtype=np.float16)
                    if retain_observation
                    else None
                ),
                "samples": sampled,
            }
        )
        records.append(
            {
                "date": date.isoformat(),
                "image": str(path),
                "valid_pixel_count": int(np.count_nonzero(valid)),
                "sample_count": int(valid_indices.size),
                "valid_fraction": float(np.mean(valid)),
                "median_d2": float(np.nanmedian(d2[valid])),
                "median_dynamic": (
                    float(np.nanmedian(dynamic[valid]))
                    if dynamic is not None
                    else None
                ),
                "prediction": diagnostics,
            }
        )
        print(
            f"  Normal reference {frame_index:02d}: {path.name} "
            f"({valid_indices.size} sampled pixels)"
        )
    return frames, records


def apply_terminal_state_selection(
    frames: Iterable[dict],
    *,
    random_seed: int,
    max_samples_per_frame: int = DEFAULT_MAX_SAMPLES_PER_FRAME,
    transition_multiplier: float = DEFAULT_TRANSITION_MULTIPLIER,
    transition_guard_frames: int = DEFAULT_TRANSITION_GUARD_FRAMES,
    minimum_stable_observations: int = DEFAULT_MINIMUM_STABLE_OBSERVATIONS,
    minimum_stable_days: int = DEFAULT_MINIMUM_STABLE_DAYS,
    minimum_state_change: float = DEFAULT_MINIMUM_STATE_CHANGE,
) -> tuple[list[dict], dict]:
    """Restrict calibration samples to each pixel's latest stable state.

    The selection is derived solely from the learned dynamic corrections and
    acquisition order. Target-event labels and target-image residuals are never
    used. Frames are updated in place with ``calibration_valid`` and resampled
    distributions so every downstream calibration path uses the same support.
    """

    frames = list(frames)
    if not frames:
        raise ValueError("No reference frames were supplied")
    missing = [frame["path"] for frame in frames if frame.get("dynamic_state") is None]
    if missing:
        raise ValueError(
            "Terminal-state calibration requires actual dynamic-state outputs; "
            f"missing for {len(missing)} frame(s)"
        )

    states = np.stack([frame["dynamic_state"] for frame in frames], axis=0)
    valid = np.stack([frame["valid"] for frame in frames], axis=0)
    settings = TerminalStateSettings(
        transition_multiplier=float(transition_multiplier),
        transition_guard_frames=int(transition_guard_frames),
        minimum_stable_observations=int(minimum_stable_observations),
        minimum_stable_days=int(minimum_stable_days),
        minimum_state_change=float(minimum_state_change),
    )
    acquisition_dates = [frame["date"] for frame in frames]
    stable_masks, state_change, audit = terminal_state_masks(
        states, valid, acquisition_dates, settings=settings
    )

    rng = np.random.default_rng(random_seed)
    frame_audit = []
    for index, frame in enumerate(frames):
        calibration_valid = stable_masks[index]
        frame["calibration_valid"] = calibration_valid
        frame["dynamic"] = state_change[index]
        valid_indices = np.flatnonzero(calibration_valid.ravel())
        total_count = int(valid_indices.size)
        if valid_indices.size > max_samples_per_frame:
            valid_indices = rng.choice(
                valid_indices, size=max_samples_per_frame, replace=False
            )
        frame["samples"] = {
            "d2": frame["d2"].ravel()[valid_indices].astype(np.float32),
            "cfar": frame["cfar"].ravel()[valid_indices].astype(np.float32),
            "sam": frame["sam"].ravel()[valid_indices].astype(np.float32),
        }
        frame_audit.append(
            {
                "date": frame["date"].isoformat(),
                "image": str(frame["path"]),
                "terminal_stable_pixel_count": total_count,
                "terminal_stable_fraction": float(np.mean(calibration_valid)),
                "sample_count": int(valid_indices.size),
                "median_state_change": (
                    float(np.nanmedian(state_change[index][calibration_valid]))
                    if total_count
                    else None
                ),
            }
        )
    audit["frames"] = frame_audit
    return frames, audit


def apply_observation_state_selection(
    frames: Iterable[dict],
    *,
    random_seed: int,
    sample_responses: bool = True,
    max_samples_per_frame: int = DEFAULT_MAX_SAMPLES_PER_FRAME,
    transition_multiplier: float = DEFAULT_TRANSITION_MULTIPLIER,
    transition_guard_frames: int = DEFAULT_TRANSITION_GUARD_FRAMES,
    minimum_stable_observations: int = DEFAULT_MINIMUM_STABLE_OBSERVATIONS,
    minimum_stable_days: int = DEFAULT_MINIMUM_STABLE_DAYS,
    minimum_state_change: float = DEFAULT_MINIMUM_STATE_CHANGE,
) -> tuple[list[dict], dict]:
    """Apply model-neutral stable support derived from observed reflectance.

    ``sample_responses=False`` returns only the observation-derived support and
    state-change arrays. This mode is used by controlled model comparisons so
    no evaluated method contributes residual values to support selection.
    """

    frames = list(frames)
    if not frames:
        raise ValueError("No reference frames were supplied")
    missing = [frame["path"] for frame in frames if frame.get("observation") is None]
    if missing:
        raise ValueError(
            "Observation-state selection requires retained observations; "
            f"missing for {len(missing)} frame(s)"
        )
    observations = np.stack([frame["observation"] for frame in frames], axis=0)
    valid = np.stack([frame["valid"] for frame in frames], axis=0)
    dates = [frame["date"] for frame in frames]
    settings = TerminalStateSettings(
        transition_multiplier=float(transition_multiplier),
        transition_guard_frames=int(transition_guard_frames),
        minimum_stable_observations=int(minimum_stable_observations),
        minimum_stable_days=int(minimum_stable_days),
        minimum_state_change=float(minimum_state_change),
    )
    stable_masks, state_change, audit = observation_state_masks(
        observations, valid, dates, settings=settings
    )
    rng = np.random.default_rng(random_seed)
    frame_audit = []
    for index, frame in enumerate(frames):
        calibration_valid = stable_masks[index]
        frame["calibration_valid"] = calibration_valid
        frame["dynamic"] = state_change[index]
        valid_indices = np.flatnonzero(calibration_valid.ravel())
        total_count = int(valid_indices.size)
        if valid_indices.size > max_samples_per_frame:
            valid_indices = rng.choice(
                valid_indices, size=max_samples_per_frame, replace=False
            )
        if sample_responses:
            frame["samples"] = {
                "d2": frame["d2"].ravel()[valid_indices].astype(np.float32),
                "cfar": frame["cfar"].ravel()[valid_indices].astype(np.float32),
                "sam": frame["sam"].ravel()[valid_indices].astype(np.float32),
            }
        frame_audit.append(
            {
                "date": frame["date"].isoformat(),
                "image": str(frame["path"]),
                "stable_pixel_count": total_count,
                "stable_fraction": float(np.mean(calibration_valid)),
                "sample_count": int(valid_indices.size),
            }
        )
    audit["frames"] = frame_audit
    audit["response_values_used_for_support_selection"] = False
    return frames, audit


def apply_fixed_calibration_support(
    frames: Iterable[dict],
    support_by_date: dict[str, np.ndarray],
    *,
    random_seed: int,
    state_change_by_date: dict[str, np.ndarray] | None = None,
    max_samples_per_frame: int = DEFAULT_MAX_SAMPLES_PER_FRAME,
) -> tuple[list[dict], dict]:
    """Apply one precomputed, model-neutral support to an evaluated model.

    ``state_change_by_date`` is derived only from the dated observations. It is
    copied into each method's calibration samples so that neither SINC nor any
    competing model determines the support used for comparison.
    """

    frames = list(frames)
    audit_frames = []
    for frame in frames:
        date_key = frame["date"].isoformat()
        if date_key not in support_by_date:
            raise KeyError(f"Missing fixed calibration support for {date_key}")
        support = np.asarray(support_by_date[date_key], dtype=bool)
        if support.shape != frame["valid"].shape:
            raise ValueError(
                f"Fixed support shape {support.shape} does not match "
                f"frame shape {frame['valid'].shape} for {date_key}"
            )
        calibration_valid = support & np.asarray(frame["valid"], dtype=bool)
        frame["calibration_valid"] = calibration_valid
        if state_change_by_date is not None:
            if date_key not in state_change_by_date:
                raise KeyError(f"Missing observed state-change map for {date_key}")
            fixed_state_change = np.asarray(
                state_change_by_date[date_key], dtype=np.float32
            )
            if fixed_state_change.shape != calibration_valid.shape:
                raise ValueError(
                    f"State-change map shape {fixed_state_change.shape} does not match "
                    f"frame shape {calibration_valid.shape} for {date_key}"
                )
            # The estimator retains the historical key for compatibility, but
            # its value is explicitly model-neutral observed state change.
            frame["dynamic"] = fixed_state_change
        valid_indices = np.flatnonzero(calibration_valid.ravel())
        total_count = int(valid_indices.size)
        if valid_indices.size > max_samples_per_frame:
            # Sample each date independently so a frame keeps exactly the same
            # calibration pixels when later dates are added during adaptive
            # convergence testing.
            frame_seed = np.random.SeedSequence(
                [int(random_seed), int(frame["date"].toordinal())]
            )
            frame_rng = np.random.default_rng(frame_seed)
            valid_indices = frame_rng.choice(
                valid_indices, size=max_samples_per_frame, replace=False
            )
        samples = {
            "d2": frame["d2"].ravel()[valid_indices].astype(np.float32),
            "cfar": frame["cfar"].ravel()[valid_indices].astype(np.float32),
            "sam": frame["sam"].ravel()[valid_indices].astype(np.float32),
        }
        frame["samples"] = samples
        audit_frames.append(
            {
                "date": date_key,
                "image": str(frame["path"]),
                "fixed_support_pixel_count": int(np.count_nonzero(support)),
                "retained_pixel_count": total_count,
                "sample_count": int(valid_indices.size),
            }
        )
    return frames, {
        "selection_method": "fixed_model_neutral_observation_support",
        "fixed_observed_state_change": state_change_by_date is not None,
        "frames": audit_frames,
    }


def pool_frame_samples(frames: Iterable[dict]) -> dict[str, np.ndarray]:
    """Concatenate sampled normal distributions across reference frames."""

    frames = list(frames)
    keys = {"d2", "cfar", "sam"}
    return {
        key: np.concatenate([frame["samples"][key] for frame in frames])
        for key in sorted(keys)
    }


def fit_thresholds(
    frames: Iterable[dict],
    *,
    n_bands: int,
    alpha: float,
    case: str,
    method: str,
    calibration_source: str,
    reference_quantile: float = DEFAULT_REFERENCE_QUANTILE,
    state_selection_audit: dict | None = None,
) -> EmpiricalNullThresholds:
    frames = list(frames)
    selection_settings = (state_selection_audit or {}).get("settings", {})
    calibration_dates = sorted(frame["date"] for frame in frames)
    common_arguments = {
        "n_bands": n_bands,
        "alpha": alpha,
        "null_trim_alpha": min(DEFAULT_NULL_TRIM_ALPHA, alpha / 2.0),
        "reference_quantile": float(reference_quantile),
        "case": case,
        "method": method,
        "calibration_source": calibration_source,
        "state_selection_method": (state_selection_audit or {}).get(
            "selection_method"
        ),
        "transition_multiplier": selection_settings.get("transition_multiplier"),
        "transition_guard_frames": selection_settings.get(
            "transition_guard_frames"
        ),
        "minimum_stable_observations": selection_settings.get(
            "minimum_stable_observations"
        ),
        "minimum_stable_days": selection_settings.get("minimum_stable_days"),
        "calibration_start_date": (
            calibration_dates[0].isoformat() if calibration_dates else None
        ),
        "calibration_end_date": (
            calibration_dates[-1].isoformat() if calibration_dates else None
        ),
    }
    return estimate_empirical_null_thresholds(
        pool_frame_samples(frames),
        calibration_frame_count=len(frames),
        **common_arguments,
    )


def normal_frame_far(frame: dict, thresholds: EmpiricalNullThresholds) -> dict:
    """Evaluate the fixed decision on a held-out normal-reference frame."""

    valid = np.asarray(
        frame.get("calibration_valid", frame["valid"]), dtype=bool
    ).copy()
    maps, diagnostics = apply_empirical_null_thresholds(
        frame["d2"], frame["cfar"], frame["sam"], valid, thresholds,
        acquisition_date=frame["date"],
    )
    valid_count = int(np.count_nonzero(valid))
    anomaly_count = int(np.count_nonzero(maps["mask"]))
    return {
        "date": frame["date"].isoformat(),
        "image": str(frame["path"]),
        "valid_pixel_count": valid_count,
        "false_alarm_pixel_count": anomaly_count,
        "normal_false_alarm_rate": (
            float(anomaly_count / valid_count) if valid_count else float("nan")
        ),
        **diagnostics,
    }


def evaluate_alpha_candidates(
    calibration_frames: Iterable[dict],
    validation_frames: Iterable[dict],
    *,
    n_bands: int,
    case: str,
    method: str,
    calibration_source: str,
    alpha_candidates: Iterable[float] = DEFAULT_ALPHA_CANDIDATES,
    target_normal_far: float = DEFAULT_TARGET_NORMAL_FAR,
    date_far_quantile: float = DEFAULT_DATE_FAR_QUANTILE,
    reference_quantile: float = DEFAULT_REFERENCE_QUANTILE,
    state_selection_audit: dict | None = None,
) -> tuple[float, list[dict], dict[float, EmpiricalNullThresholds]]:
    """Select alpha using a high quantile of held-out date-wise normal FAR."""

    calibration_frames = list(calibration_frames)
    validation_frames = list(validation_frames)
    if not 0.0 < float(date_far_quantile) <= 1.0:
        raise ValueError("date_far_quantile must be in (0, 1]")
    candidates = sorted({float(value) for value in alpha_candidates}, reverse=True)
    if not candidates:
        raise ValueError("alpha_candidates cannot be empty")
    if not 0.0 < float(target_normal_far) < 1.0:
        raise ValueError("target_normal_far must be in (0, 1)")
    rows = []
    thresholds_by_alpha = {}
    for alpha in candidates:
        thresholds = fit_thresholds(
            calibration_frames,
            n_bands=n_bands,
            alpha=alpha,
            case=case,
            method=method,
            calibration_source=calibration_source,
            reference_quantile=reference_quantile,
            state_selection_audit=state_selection_audit,
        )
        thresholds_by_alpha[alpha] = thresholds
        frame_rows = [normal_frame_far(frame, thresholds) for frame in validation_frames]
        rates = np.asarray(
            [row["normal_false_alarm_rate"] for row in frame_rows], dtype=np.float64
        )
        finite_rates = rates[np.isfinite(rates)]
        mean_far = float(np.mean(finite_rates)) if finite_rates.size else float("nan")
        robust_date_far = (
            float(np.quantile(finite_rates, date_far_quantile))
            if finite_rates.size
            else float("nan")
        )
        p95_date_far = (
            float(np.quantile(finite_rates, 0.95))
            if finite_rates.size
            else float("nan")
        )
        for frame_row in frame_rows:
            rows.append(
                {
                    "case": case,
                    "method": method,
                    "alpha": alpha,
                    "target_normal_far": target_normal_far,
                    "date_far_quantile": float(date_far_quantile),
                    "reference_quantile": float(reference_quantile),
                    "mean_normal_far": mean_far,
                    "p95_normal_far": p95_date_far,
                    "selection_normal_far": robust_date_far,
                    **frame_row,
                }
            )

    eligible = []
    for alpha in candidates:
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        robust_date_far = (
            alpha_rows[0]["selection_normal_far"]
            if alpha_rows
            else float("inf")
        )
        if (
            np.isfinite(robust_date_far)
            and robust_date_far <= target_normal_far
        ):
            eligible.append(alpha)
    selected = max(eligible) if eligible else min(candidates)
    return selected, rows, thresholds_by_alpha
