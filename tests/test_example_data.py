import re
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from ground.core.NDCDatasetLoader import DatasetLoader
from onboard.preprocessing.image_preprocessor import ImagePreprocessor


class VegetationFireDataTests(unittest.TestCase):
    def test_case_history_and_raster_layout(self):
        root = Path(__file__).resolve().parents[1]
        data_root = root / "data" / "Vegetation Fire"
        groups = {
            "train": sorted((data_root / "train").glob("*.tif")),
            "target": sorted((data_root / "target").glob("*.tif")),
        }
        self.assertEqual(
            {name: len(files) for name, files in groups.items()},
            {"train": 106, "target": 1},
        )

        expected_signature = None
        dates = {}
        for group_name, files in groups.items():
            dates[group_name] = []
            for image_path in files:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", image_path.name)
                self.assertIsNotNone(match, image_path.name)
                dates[group_name].append(match.group(1))
                with rasterio.open(image_path) as dataset:
                    signature = (
                        dataset.count,
                        dataset.height,
                        dataset.width,
                        str(dataset.crs),
                        tuple(dataset.descriptions or ()),
                    )
                if expected_signature is None:
                    expected_signature = signature
                self.assertEqual(signature, expected_signature, image_path.name)

        self.assertEqual(expected_signature[0], 7)
        self.assertEqual(
            expected_signature[4],
            ("B2", "B3", "B4", "B5", "B8", "B11", "B12"),
        )
        self.assertLess(max(dates["train"]), min(dates["target"]))
        cutoff = "2020-11-05"
        self.assertEqual(sum(date <= cutoff for date in dates["train"]), 84)
        self.assertEqual(sum(date > cutoff for date in dates["train"]), 22)

    def test_five_band_manuscript_selection(self):
        root = Path(__file__).resolve().parents[1]
        image_path = next((root / "data" / "Vegetation Fire" / "train").glob("*.tif"))
        observation, metadata = ImagePreprocessor().load_tif(
            image_path,
            expected_bands=5,
            source_band_indices=[2, 3, 5, 6, 7],
            source_band_names=["B3", "B4", "B8", "B11", "B12"],
        )
        self.assertEqual(observation.shape[-1], 5)
        self.assertEqual(metadata["source_bands"], 7)
        self.assertEqual(metadata["source_band_indices"], [2, 3, 5, 6, 7])
        self.assertEqual(metadata["source_band_names"], ["B3", "B4", "B8", "B11", "B12"])

    def test_five_band_read_requires_explicit_mapping(self):
        root = Path(__file__).resolve().parents[1]
        image_path = next((root / "data" / "Vegetation Fire" / "train").glob("*.tif"))
        with self.assertRaisesRegex(ValueError, "source_band_indices"):
            ImagePreprocessor().load_tif(image_path, expected_bands=5)

    def test_training_loader_uses_five_band_mapping(self):
        root = Path(__file__).resolve().parents[1]
        train_root = root / "data" / "Vegetation Fire" / "train"
        start = datetime(2018, 7, 14, 10, 30)
        end = datetime(2018, 7, 24, 10, 30)
        volume, nodata, metadata = DatasetLoader(
            dtype=np.float32, use_memmap=False
        ).load_folder(
            train_root,
            expected_bands=5,
            source_band_indices=[2, 3, 5, 6, 7],
            source_band_names=["B3", "B4", "B8", "B11", "B12"],
            scale_factor=0.0001,
            valid_range=(0.0, 1.0),
            valid_threshold=0.0,
            global_start_date=start,
            global_end_date=end,
        )
        self.assertEqual(volume.shape, (2, 279, 279, 5))
        self.assertEqual(nodata.shape, (2, 279, 279))
        self.assertEqual(metadata["source_file_count"], 2)
        self.assertEqual(metadata["temporal_observation_count"], 2)
        self.assertEqual(metadata["source_band_indices"], [2, 3, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
