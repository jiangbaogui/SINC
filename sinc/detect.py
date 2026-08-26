"""Public command-line entry point for SINC anomaly detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from onboard.main import OnboardAnomalyDetector


SINCAnomalyDetector = OnboardAnomalyDetector

__all__ = ["OnboardAnomalyDetector", "SINCAnomalyDetector", "main"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SINC anomaly detection")
    parser.add_argument("--model", "-m", required=True, help="Path to a calibrated GNDC")
    parser.add_argument(
        "--input", "-i", required=True, help="Input GeoTIFF file or directory"
    )
    parser.add_argument(
        "--threshold-profile",
        "-t",
        default=None,
        help="Optional empirical-null JSON; otherwise use the profile embedded in the GNDC",
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--device", "-d", default="cuda", help="CUDA device")
    parser.add_argument("--chunk-size", "-c", type=int, default=262144)
    parser.add_argument("--pattern", "-p", default="*.tif")
    parser.add_argument("--pixel-size-m", type=float, default=20.0)
    parser.add_argument("--export-vector", action="store_true")
    parser.add_argument("--apply-morphology", action="store_true")
    args = parser.parse_args(argv)

    detector = OnboardAnomalyDetector(
        model_path=args.model,
        threshold_profile=args.threshold_profile,
        device=args.device,
        chunk_size=args.chunk_size,
        pixel_size_m=args.pixel_size_m,
        export_vector=args.export_vector,
        apply_morphology=args.apply_morphology,
    )
    input_path = Path(args.input)
    if input_path.is_dir():
        detector.detect_batch(input_path, args.output, pattern=args.pattern)
    else:
        detector.detect_single(input_path, args.output)


if __name__ == "__main__":
    main()
