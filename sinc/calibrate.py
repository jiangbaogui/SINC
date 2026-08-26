"""Calibrate a date-resolved empirical-null profile from the GNDC training history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from experiments.adaptive_detection.empirical_null_calibration import (
    DEFAULT_MINIMUM_STABLE_DAYS,
    DEFAULT_MINIMUM_STABLE_OBSERVATIONS,
    DEFAULT_MINIMUM_STATE_CHANGE,
    DEFAULT_TRANSITION_GUARD_FRAMES,
    apply_terminal_state_selection,
    collect_frame_distributions,
    dated_images_before,
    filter_stable_reference_frames,
    fit_thresholds,
)
from common.scene_inference import predict_frame_with_state, read_observation
from ground.core.GNDCProfile import embed_empirical_null_profile
from onboard.core.anomaly_detector import AnomalyDetector
from onboard.core.onboard_inference import OnboardInference


def _metadata_datetime(value) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def calibrate(
    model_path,
    training_dir,
    output_path,
    *,
    alpha=0.01,
    device="cuda",
    seed=42,
    calibrated_model_path=None,
    transition_guard_frames=DEFAULT_TRANSITION_GUARD_FRAMES,
    minimum_stable_observations=DEFAULT_MINIMUM_STABLE_OBSERVATIONS,
    minimum_stable_days=DEFAULT_MINIMUM_STABLE_DAYS,
    minimum_state_change=DEFAULT_MINIMUM_STATE_CHANGE,
):
    """Estimate thresholds without reading target imagery or disturbance labels."""

    inference = OnboardInference(str(model_path), device=device)
    inference.require_current_model_contract(expected_ablation_mode="full")
    detector = AnomalyDetector(inference.get_meta(), inference.get_band_indices())
    last_training_date = _metadata_datetime(
        inference.meta["last_training_observation_date"]
    )
    references = dated_images_before(
        training_dir, last_training_date + timedelta(microseconds=1)
    )
    if len(references) < 4:
        raise RuntimeError("At least four dated GNDC training images are required")

    def predictor(date, _path, observation):
        height, width, _ = observation.shape
        mean, std, state, velocity = predict_frame_with_state(
            inference, date, height, width
        )
        return mean, std, {"state": state, "velocity": velocity}, {
            "implementation": "SINC GNDC"
        }

    frames, prediction_records = collect_frame_distributions(
        references,
        predictor,
        detector,
        read_observation,
        expected_bands=len(inference.band_indices),
        random_seed=int(seed),
    )
    frames, state_audit = apply_terminal_state_selection(
        frames,
        random_seed=int(seed) + 1,
        transition_guard_frames=int(transition_guard_frames),
        minimum_stable_observations=int(minimum_stable_observations),
        minimum_stable_days=int(minimum_stable_days),
        minimum_state_change=float(minimum_state_change),
    )
    accepted, quality_audit = filter_stable_reference_frames(frames)
    profile = fit_thresholds(
        accepted,
        n_bands=len(inference.band_indices),
        alpha=float(alpha),
        case=Path(training_dir).name,
        method="SINC",
        calibration_source="global_terminal_state_training_residuals",
        state_selection_audit=state_audit,
    )
    profile_path = profile.save(output_path)
    embedded_path = (
        Path(calibrated_model_path)
        if calibrated_model_path is not None
        else profile_path.with_suffix(".gndc")
    )
    embed_empirical_null_profile(model_path, profile, embedded_path)

    audit_path = profile_path.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "model": str(Path(model_path).resolve()),
                "last_training_observation_date": last_training_date.isoformat(),
                "target_images_or_labels_used": False,
                "prediction_records": prediction_records,
                "terminal_state_selection": state_audit,
                "frame_quality_control": quality_audit,
                "accepted_dates": [frame["date"].isoformat() for frame in accepted],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Threshold profile: {profile_path}")
    print(f"Calibrated GNDC: {embedded_path}")
    print(f"Calibration audit: {audit_path}")
    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SINC empirical-null thresholds from the GNDC training history"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--normal-dir", required=True, dest="training_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrated-model", default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transition-guard-frames",
        type=int,
        default=DEFAULT_TRANSITION_GUARD_FRAMES,
    )
    parser.add_argument(
        "--minimum-stable-observations",
        type=int,
        default=DEFAULT_MINIMUM_STABLE_OBSERVATIONS,
    )
    parser.add_argument(
        "--minimum-stable-days", type=int, default=DEFAULT_MINIMUM_STABLE_DAYS
    )
    parser.add_argument(
        "--minimum-state-change", type=float, default=DEFAULT_MINIMUM_STATE_CHANGE
    )
    args = parser.parse_args()
    calibrate(
        args.model,
        args.training_dir,
        args.output,
        alpha=args.alpha,
        device=args.device,
        seed=args.seed,
        calibrated_model_path=args.calibrated_model,
        transition_guard_frames=args.transition_guard_frames,
        minimum_stable_observations=args.minimum_stable_observations,
        minimum_stable_days=args.minimum_stable_days,
        minimum_state_change=args.minimum_state_change,
    )


if __name__ == "__main__":
    main()
