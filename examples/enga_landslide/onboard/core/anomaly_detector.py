"""Empirical-null anomaly detection for a GNDC probabilistic baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, label as connected_components

from onboard.core.empirical_null import (
    EmpiricalNullThresholds,
    apply_empirical_null_thresholds,
    calculate_detection_maps,
)


class AnomalyDetector:
    """Apply date-resolved empirical-null thresholds to a target observation."""

    def __init__(self, meta_info=None, band_indices=None, threshold_profile=None):
        self.meta = meta_info or {}
        self.band_indices = band_indices or [0, 1, 2, 3, 4, 5, 6]
        self.cfar_scales = [(11, 3), (25, 9), (51, 21)]
        self.cfar_kernels = [
            self._create_donut_kernel(outer, inner)
            for outer, inner in self.cfar_scales
        ]
        self.thresholds = self._load_thresholds(threshold_profile)
        self.last_detection_diagnostics = {}

    @staticmethod
    def _load_thresholds(profile):
        if profile is None:
            return None
        if isinstance(profile, EmpiricalNullThresholds):
            return profile
        if isinstance(profile, (str, Path)):
            return EmpiricalNullThresholds.load(profile)
        return EmpiricalNullThresholds.from_dict(profile)

    @staticmethod
    def _create_donut_kernel(outer_size, inner_size):
        outer_radius = outer_size // 2
        inner_radius = inner_size // 2
        y, x = np.ogrid[
            -outer_radius : outer_radius + 1,
            -outer_radius : outer_radius + 1,
        ]
        ring = (
            (x**2 + y**2 <= outer_radius**2)
            & (x**2 + y**2 > inner_radius**2)
        )
        kernel = np.zeros_like(ring, dtype=np.float32)
        if np.any(ring):
            kernel[ring] = 1.0 / np.count_nonzero(ring)
        return kernel

    @staticmethod
    def _calculate_sam(first, second):
        dot = np.sum(first * second, axis=-1)
        first_norm = np.linalg.norm(first, axis=-1)
        second_norm = np.linalg.norm(second, axis=-1)
        cosine = dot / (first_norm * second_norm + 1e-9)
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    def get_detections(self, mask, score, transform, crs):
        """Vectorize connected anomaly regions for the existing exporter."""
        import rasterio.features
        from shapely.geometry import shape

        mask_uint8 = np.asarray(mask, dtype=np.uint8)
        geometries = rasterio.features.shapes(
            mask_uint8, mask=mask_uint8.astype(bool), transform=transform
        )
        detections = []
        for geometry, value in geometries:
            if value != 1:
                continue
            polygon = shape(geometry)
            detections.append(
                {
                    "area_m2": polygon.area,
                    "centroid": [polygon.centroid.x, polygon.centroid.y],
                    "bounds": polygon.bounds,
                    "geometry": geometry,
                }
            )
        return detections

    @staticmethod
    def _morphology_clean(mask):
        mask = np.asarray(mask, dtype=bool).copy()
        labeled, count = connected_components(mask, structure=np.ones((3, 3)))
        if count:
            sizes = np.bincount(labeled.ravel())
            mask[sizes[labeled] < 20] = False
        y, x = np.ogrid[-1:2, -1:2]
        disk = x**2 + y**2 <= 1
        padded = np.pad(mask, pad_width=1, mode="edge")
        return binary_closing(padded, structure=disk)[1:-1, 1:-1]

    def detect(
        self,
        obs,
        pred_mean,
        pred_std,
        thresholds=None,
        clean=False,
        acquisition_date=None,
        pixel_area_m2=400.0,
    ):
        """Return a binary mask, normalized decision score, and area estimate."""
        selected = self._load_thresholds(thresholds) or self.thresholds
        if selected is None:
            raise ValueError(
                "An empirical-null threshold profile is required. Calibrate the "
                "region on normal reference observations before detection."
            )

        d2, cfar, sam, valid = calculate_detection_maps(
            obs, pred_mean, pred_std, self
        )
        maps, diagnostics = apply_empirical_null_thresholds(
            d2, cfar, sam, valid, selected, acquisition_date=acquisition_date
        )
        mask = maps["mask"].astype(bool)
        score = maps["fusion"]

        raw_count = int(np.count_nonzero(mask))
        if clean and raw_count:
            mask = self._morphology_clean(mask)
        diagnostics.update(
            {
                "morphology_applied": bool(clean),
                "raw_anomaly_pixel_count": raw_count,
                "final_anomaly_pixel_count": int(np.count_nonzero(mask)),
            }
        )
        self.last_detection_diagnostics = diagnostics
        if float(pixel_area_m2) <= 0:
            raise ValueError("pixel_area_m2 must be positive")
        area_km2 = float(np.count_nonzero(mask) * float(pixel_area_m2) / 1e6)
        return mask, score, area_km2
