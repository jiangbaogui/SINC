"""Empirical-null calibration and anomaly decisions shared by all experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import convolve as scipy_convolve


SCHEMA_VERSION = 4
DEFAULT_ALPHA = 0.01
DEFAULT_NULL_TRIM_ALPHA = 0.001
DEFAULT_RESIDUAL_OFFSET = 0.005
DEFAULT_REFERENCE_QUANTILE = 0.95


@dataclass(frozen=True)
class EmpiricalNullThresholds:
    """Pre-target empirical thresholds fitted on terminal-state residuals."""

    alpha: float
    d2_threshold: float
    cfar_threshold: float
    sam_threshold: float
    empirical_sam_threshold: float
    sample_count: int
    stable_sample_count: int
    null_sample_count: int
    n_bands: int
    preliminary_null_limit: float
    robust_scale: float
    fusion_threshold: float = 1.0
    reference_quantile: float = DEFAULT_REFERENCE_QUANTILE
    calibration_frame_count: int = 0
    case: str | None = None
    method: str | None = None
    calibration_source: str | None = None
    state_selection_method: str | None = None
    transition_multiplier: float | None = None
    transition_guard_frames: int | None = None
    minimum_stable_observations: int | None = None
    minimum_stable_days: int | None = None
    calibration_start_date: str | None = None
    calibration_end_date: str | None = None
    schema_version: int = SCHEMA_VERSION
    decision_rule: str = "global_empirical_null_d2_gate_with_cfar_or_sam"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping) -> "EmpiricalNullThresholds":
        schema_version = int(values.get("schema_version", 0))
        if schema_version < SCHEMA_VERSION:
            raise ValueError(
                "The empirical-null profile predates the terminal-state global "
                "decision contract. Recalibrate the GNDC with the current code."
            )
        decision_rule = values.get("decision_rule")
        expected_rule = cls.__dataclass_fields__["decision_rule"].default
        if decision_rule != expected_rule:
            raise ValueError(
                f"Unsupported empirical-null decision rule {decision_rule!r}; "
                f"expected {expected_rule!r}. Recalibration is required."
            )
        known = cls.__dataclass_fields__
        payload = {key: values[key] for key in known if key in values}
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalNullThresholds":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def thresholds_for_date(self, acquisition_date=None) -> dict[str, float]:
        """Return the pre-target regional profile for any acquisition date."""

        return {
            "d2_threshold": float(self.d2_threshold),
            "cfar_threshold": float(self.cfar_threshold),
            "sam_threshold": float(self.sam_threshold),
            "empirical_sam_threshold": float(self.empirical_sam_threshold),
            "fusion_threshold": float(self.fusion_threshold),
            "threshold_mode": "global_terminal_state_empirical_null",
        }


def _weighted_quantile(values, quantile: float, weights=None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        raise ValueError("No positively weighted finite samples are available")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


def valid_observation_mask(observation: np.ndarray) -> np.ndarray:
    """Return pixels with finite, nonzero reflectance in the supported range."""

    observation = np.asarray(observation)
    finite = np.all(np.isfinite(observation), axis=-1)
    nonzero = ~np.all(observation == 0, axis=-1)
    in_range = np.all((observation >= 0.0) & (observation <= 1.2), axis=-1)
    return finite & nonzero & in_range


def calculate_detection_maps(
    observation: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    detector,
    residual_offset: float = DEFAULT_RESIDUAL_OFFSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate multiband D2, multi-scale CFAR, SAM, and the valid mask."""

    band_indices = detector.band_indices
    obs = np.asarray(observation)[..., band_indices]
    pred = np.asarray(predicted_mean)[..., band_indices]
    std = np.asarray(predicted_std)[..., band_indices]
    valid = valid_observation_mask(obs)
    valid &= np.all(np.isfinite(pred), axis=-1)
    valid &= np.all(np.isfinite(std), axis=-1)

    d2 = np.sum(((obs - pred) / (std + residual_offset)) ** 2, axis=-1)
    fill_value = float(np.median(d2[valid])) if np.any(valid) else 0.0
    d2_filled = np.where(valid, d2, fill_value)

    cfar = np.zeros_like(d2_filled, dtype=np.float32)
    for kernel in detector.cfar_kernels:
        background = scipy_convolve(
            d2_filled, kernel, mode="constant", cval=fill_value
        )
        cfar = np.maximum(cfar, d2_filled - background)

    sam = detector._calculate_sam(obs, pred).astype(np.float32)
    d2 = d2.astype(np.float32)
    d2[~valid] = np.nan
    cfar[~valid] = np.nan
    sam[~valid] = np.nan
    return d2, cfar, sam, valid


def estimate_empirical_null_thresholds(
    samples: Mapping[str, np.ndarray],
    n_bands: int,
    alpha: float = DEFAULT_ALPHA,
    *,
    null_trim_alpha: float = DEFAULT_NULL_TRIM_ALPHA,
    reference_quantile: float = DEFAULT_REFERENCE_QUANTILE,
    calibration_frame_count: int = 0,
    case: str | None = None,
    method: str | None = None,
    calibration_source: str | None = None,
    state_selection_method: str | None = None,
    transition_multiplier: float | None = None,
    transition_guard_frames: int | None = None,
    minimum_stable_observations: int | None = None,
    minimum_stable_days: int | None = None,
    calibration_start_date: str | None = None,
    calibration_end_date: str | None = None,
) -> EmpiricalNullThresholds:
    """Estimate empirical thresholds from quality-controlled normal observations.

    The caller must restrict samples to the latest stable state before this
    function is called. D2, CFAR, and SAM are then treated symmetrically through
    empirical reference quantiles. No Gaussian or chi-square tail assumption is
    imposed on the residuals.
    """

    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be between 0 and 0.5, got {alpha}")
    if not 0.0 < null_trim_alpha < alpha:
        raise ValueError(
            "null_trim_alpha must be positive and smaller than alpha "
            f"({null_trim_alpha} >= {alpha})"
        )
    if not 0.5 < reference_quantile < 1.0:
        raise ValueError(
            "reference_quantile must be between 0.5 and 1.0, got "
            f"{reference_quantile}"
        )

    required = ("d2", "cfar", "sam")
    arrays = {key: np.asarray(samples[key], dtype=np.float64).ravel() for key in required}
    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("D2, CFAR, and SAM calibration arrays must have equal length")

    finite = np.ones(arrays["d2"].size, dtype=bool)
    for values in arrays.values():
        finite &= np.isfinite(values)
    if np.count_nonzero(finite) < 1000:
        raise ValueError("At least 1000 finite normal calibration samples are required")

    arrays = {key: values[finite] for key, values in arrays.items()}
    def q(values, probability, mask=None):
        if mask is not None:
            values = values[mask]
        return _weighted_quantile(values, probability)

    stable = np.ones(arrays["d2"].size, dtype=bool)
    robust_scale = float(q(arrays["d2"], 0.5))
    preliminary_null_limit = float(
        q(arrays["d2"], 1.0 - null_trim_alpha)
    )
    null_samples = stable & (arrays["d2"] <= preliminary_null_limit)
    if np.count_nonzero(null_samples) < 1000:
        null_samples = stable

    d2_reference = float(
        q(arrays["d2"], reference_quantile, null_samples)
    )
    cfar_reference = float(
        q(arrays["cfar"], reference_quantile, null_samples)
    )
    empirical_sam = float(
        q(arrays["sam"], reference_quantile, null_samples)
    )
    sam_reference = empirical_sam
    joint_null_score = np.minimum(
        arrays["d2"][null_samples] / (d2_reference + 1e-12),
        np.maximum(
            arrays["cfar"][null_samples] / (cfar_reference + 1e-12),
            arrays["sam"][null_samples] / (sam_reference + 1e-12),
        ),
    )
    fusion_threshold = _weighted_quantile(
        joint_null_score, 1.0 - alpha
    )
    return EmpiricalNullThresholds(
        alpha=float(alpha),
        d2_threshold=d2_reference,
        cfar_threshold=cfar_reference,
        sam_threshold=sam_reference,
        empirical_sam_threshold=empirical_sam,
        sample_count=int(arrays["d2"].size),
        stable_sample_count=int(np.count_nonzero(stable)),
        null_sample_count=int(np.count_nonzero(null_samples)),
        n_bands=int(n_bands),
        preliminary_null_limit=preliminary_null_limit,
        robust_scale=robust_scale,
        fusion_threshold=fusion_threshold,
        reference_quantile=float(reference_quantile),
        calibration_frame_count=int(calibration_frame_count),
        case=case,
        method=method,
        calibration_source=calibration_source,
        state_selection_method=state_selection_method,
        transition_multiplier=transition_multiplier,
        transition_guard_frames=transition_guard_frames,
        minimum_stable_observations=minimum_stable_observations,
        minimum_stable_days=minimum_stable_days,
        calibration_start_date=calibration_start_date,
        calibration_end_date=calibration_end_date,
    )


def apply_empirical_null_thresholds(
    d2: np.ndarray,
    cfar: np.ndarray,
    sam: np.ndarray,
    valid: np.ndarray,
    thresholds: EmpiricalNullThresholds | Mapping,
    acquisition_date=None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply the joint-null-calibrated D2 AND (CFAR OR SAM) decision."""

    if not isinstance(thresholds, EmpiricalNullThresholds):
        thresholds = EmpiricalNullThresholds.from_dict(thresholds)

    resolved = thresholds.thresholds_for_date(acquisition_date)
    d2_threshold = float(resolved["d2_threshold"])
    cfar_threshold = float(resolved["cfar_threshold"])
    sam_threshold = float(resolved["sam_threshold"])
    joint_threshold = max(float(resolved["fusion_threshold"]), 1e-12)
    d2_ratio = np.maximum(d2 / (d2_threshold + 1e-12), 0.0)
    cfar_ratio = np.maximum(cfar / (cfar_threshold + 1e-12), 0.0)
    sam_ratio = np.maximum(sam / (sam_threshold + 1e-12), 0.0)

    amplitude_mask = valid & (d2_ratio > joint_threshold)
    cfar_mask = valid & (cfar_ratio > joint_threshold)
    sam_mask = valid & (sam_ratio > joint_threshold)
    raw_mask = amplitude_mask & (cfar_mask | sam_mask)

    # score > 1 is exactly equivalent to the binary rule above.
    decision_score = (
        np.minimum(d2_ratio, np.maximum(cfar_ratio, sam_ratio)) / joint_threshold
    )
    decision_score = decision_score.astype(np.float32)
    decision_score[~valid] = np.nan

    maps = {
        "d2": np.asarray(d2, dtype=np.float32),
        "cfar": np.asarray(cfar, dtype=np.float32),
        "sam": np.asarray(sam, dtype=np.float32),
        "fusion": decision_score,
        "mask": raw_mask.astype(np.uint8),
    }
    diagnostics = {
        "decision_rule": thresholds.decision_rule,
        "alpha": thresholds.alpha,
        "d2_threshold": d2_threshold,
        "cfar_threshold": cfar_threshold,
        "sam_threshold": sam_threshold,
        "empirical_sam_threshold": resolved["empirical_sam_threshold"],
        "sam_threshold_source": "terminal_state_empirical_quantile",
        "decision_score_threshold": 1.0,
        "fusion_threshold": joint_threshold,
        "threshold_mode": resolved["threshold_mode"],
        "reference_quantile": thresholds.reference_quantile,
        "effective_d2_threshold": (
            d2_threshold * joint_threshold
        ),
        "effective_cfar_threshold": (
            cfar_threshold * joint_threshold
        ),
        "effective_sam_threshold": (
            sam_threshold * joint_threshold
        ),
        "morphology_applied": False,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "d2_exceedance_count": int(np.count_nonzero(amplitude_mask)),
        "cfar_exceedance_count": int(np.count_nonzero(cfar_mask)),
        "sam_exceedance_count": int(np.count_nonzero(sam_mask)),
        "raw_anomaly_pixel_count": int(np.count_nonzero(raw_mask)),
        "final_anomaly_pixel_count": int(np.count_nonzero(raw_mask)),
    }
    return maps, diagnostics
