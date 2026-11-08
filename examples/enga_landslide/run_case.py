"""Reproduce the target-date Enga experiment using the frozen HSTB-IR state."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from experiments.common import calculate_metrics, sample_mask_at_labels, save_raster
from common.scene_inference import read_observation
from onboard.core.anomaly_detector import AnomalyDetector
from onboard.core.empirical_null import (
    EmpiricalNullThresholds,
    apply_empirical_null_thresholds,
    calculate_detection_maps,
)
from sinc_offline_deployment import Deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("enga_output"))
    args = parser.parse_args()
    case = Path(__file__).resolve().parent
    data = case / "data"
    protocol = json.loads((case / "reference" / "protocol.json").read_text(encoding="utf-8"))
    expected = json.loads((case / "reference" / "metrics.json").read_text(encoding="utf-8"))
    target = data / "sentinel2_2024-05-28.tif"
    labels = data / "reference_samples.csv"
    for path, key in ((target, "target_sha256"), (labels, "labels_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Input checksum mismatch: {path.name}")
    date = datetime.fromisoformat(protocol["target_date"])
    observation, profile, transform, crs = read_observation(target, expected_bands=7)
    thresholds = EmpiricalNullThresholds.load(data / "empirical_null.json")
    engine = Deployment(data / "baseline.sinc")
    try:
        engine.inf.require_current_model_contract(expected_ablation_mode="no_temporal")
        mean, scale = (values.reshape(engine.H, engine.W, 5) for values in engine.query(date))
        detector = AnomalyDetector(engine.meta, list(range(5)))
        responses = calculate_detection_maps(observation[..., [1, 2, 4, 5, 6]], mean, scale, detector)
        maps, _ = apply_empirical_null_thresholds(*responses, thresholds, acquisition_date=date)
    finally:
        engine.close()

    mask = np.ascontiguousarray(maps["mask"], dtype=np.uint8)
    observed, predicted = sample_mask_at_labels(labels, mask, transform, crs)
    metrics = calculate_metrics(observed, predicted)
    digest = hashlib.sha256(str(mask.shape).encode("ascii"))
    digest.update(mask.dtype.str.encode("ascii"))
    digest.update(mask.tobytes())
    metrics["mask_sha256"] = digest.hexdigest()
    metrics["matches_reference_mask"] = metrics["mask_sha256"] == expected["mask_sha256"]

    args.output.mkdir(parents=True, exist_ok=True)
    save_raster(args.output / "disturbance_mask.tif", mask, profile, "uint8", nodata=0)
    raster_profile = dict(profile, count=5, dtype="float32", compress="deflate", nodata=-9999.0)
    for name, values in (("predicted_reflectance", mean), ("residual_scale", scale)):
        with rasterio.open(args.output / f"{name}.tif", "w", **raster_profile) as raster:
            raster.write(values.transpose(2, 0, 1))
    result = json.dumps(metrics, indent=2)
    (args.output / "metrics.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
