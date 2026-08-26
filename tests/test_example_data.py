import re
import unittest
from pathlib import Path

import rasterio


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
        self.assertLess(max(dates["train"]), min(dates["target"]))
        cutoff = "2020-11-05"
        self.assertEqual(sum(date <= cutoff for date in dates["train"]), 84)
        self.assertEqual(sum(date > cutoff for date in dates["train"]), 22)


if __name__ == "__main__":
    unittest.main()
