"""Create the reproducibility manifest for the bundled Vegetation Fire example."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "Vegetation Fire"
OUTPUT_PATH = DATA_ROOT / "DATA_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for group in ("train", "target"):
        for path in sorted((DATA_ROOT / group).glob("*.tif")):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
            with rasterio.open(path) as dataset:
                rows.append(
                    {
                        "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                        "date": date_match.group(1) if date_match else "",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "bands": dataset.count,
                        "height": dataset.height,
                        "width": dataset.width,
                        "crs": str(dataset.crs or ""),
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
