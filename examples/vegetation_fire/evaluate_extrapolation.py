"""Evaluate a chronological Vegetation Fire holdout with pooled reflectance RMSE."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from onboard.core.empirical_null import valid_observation_mask
from onboard.core.onboard_inference import SINCInference
from onboard.preprocessing.image_preprocessor import ImagePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SINC on chronological extrapolation-validation images"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--after-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    after_date = datetime.fromisoformat(args.after_date).date()
    through_date = datetime.fromisoformat(args.through_date).date()
    if through_date <= after_date:
        raise ValueError("--through-date must be later than --after-date")

    dated_images = []
    for image_path in sorted(args.history_dir.glob("*.tif")):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", image_path.name)
        if match is None:
            continue
        image_date = datetime.fromisoformat(match.group(1)).date()
        if after_date < image_date <= through_date:
            dated_images.append((image_date, image_path))
    images = [item[1] for item in dated_images]
    if not images:
        raise FileNotFoundError(
            "No chronological holdout GeoTIFFs were found in the requested date range"
        )

    inference = SINCInference(args.model, device=args.device)
    preprocessor = ImagePreprocessor()
    rows = []
    for image_path in images:
        observation, metadata = preprocessor.load_tif(
            str(image_path),
            expected_bands=inference.meta["n_bands"],
            source_band_indices=inference.meta.get("source_band_indices"),
            source_band_names=inference.meta.get("source_band_names"),
        )
        if metadata["date"] is None:
            raise ValueError(f"No acquisition date found in {image_path.name}")
        height, width, _ = observation.shape
        predicted_mean, _, _, _ = inference.predict_frame_with_details(
            metadata["date"], height, width
        )
        valid = valid_observation_mask(observation)
        valid &= np.all(np.isfinite(predicted_mean), axis=-1)
        if not np.any(valid):
            raise RuntimeError(f"No common-valid pixels in {image_path.name}")
        residual = observation[valid] - predicted_mean[valid]
        rmse = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
        rows.append(
            {
                "image": image_path.name,
                "date": metadata["date_str"],
                "valid_pixel_count": int(np.count_nonzero(valid)),
                "pooled_rmse": rmse,
            }
        )
        print(f"[Extrapolation] {metadata['date_str']}: RMSE={rmse:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Extrapolation] Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
