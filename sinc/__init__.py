"""Public Python interface for SINC."""

from onboard.core.anomaly_detector import AnomalyDetector
from onboard.core.empirical_null import (
    EmpiricalNullThresholds,
    apply_empirical_null_thresholds,
    calculate_detection_maps,
    estimate_empirical_null_thresholds,
)
from onboard.core.onboard_inference import OnboardInference

# Public name retained for users of the standalone SINC package.
SINCInference = OnboardInference

__all__ = [
    "AnomalyDetector",
    "EmpiricalNullThresholds",
    "OnboardInference",
    "SINCInference",
    "apply_empirical_null_thresholds",
    "calculate_detection_maps",
    "estimate_empirical_null_thresholds",
]
