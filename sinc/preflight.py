"""Validate a public SINC checkout before training or inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_manifest(root: Path = PROJECT_ROOT) -> CheckResult:
    manifest_path = root / "SOURCE_SYNC_MANIFEST.json"
    if not manifest_path.is_file():
        return CheckResult("source manifest", False, f"missing {manifest_path.name}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("files", payload)
    declared_count = int(payload.get("file_count", len(entries)))
    if declared_count != len(entries):
        return CheckResult(
            "source manifest",
            False,
            f"declared {declared_count} files but listed {len(entries)}",
        )
    mismatches = []
    for relative_path, expected_digest in entries.items():
        path = root / relative_path
        if not path.is_file():
            mismatches.append(f"missing: {relative_path}")
        elif _sha256(path) != expected_digest:
            mismatches.append(f"modified: {relative_path}")
    if mismatches:
        return CheckResult("source manifest", False, "; ".join(mismatches[:5]))
    return CheckResult(
        "source manifest",
        True,
        f"{len(entries)} synchronized implementation files verified",
    )


def verify_data_manifest(root: Path = PROJECT_ROOT) -> CheckResult:
    data_root = root / "data" / "Vegetation Fire"
    manifest_path = data_root / "DATA_MANIFEST.csv"
    if not manifest_path.is_file():
        return CheckResult("example data", False, "DATA_MANIFEST.csv is missing")

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    if len(entries) != 107:
        return CheckResult(
            "example data", False, f"expected 107 rasters, manifest has {len(entries)}"
        )

    failures = []
    train_dates = []
    target_dates = []
    for entry in entries:
        relative_path = entry["relative_path"]
        path = data_root / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        if int(entry["bytes"]) != path.stat().st_size:
            failures.append(f"size mismatch: {relative_path}")
            continue
        if _sha256(path) != entry["sha256"]:
            failures.append(f"checksum mismatch: {relative_path}")
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        if date_match:
            destination = train_dates if relative_path.startswith("train/") else target_dates
            destination.append(date_match.group(1))

    if failures:
        return CheckResult("example data", False, "; ".join(failures[:5]))
    if len(train_dates) != 106 or len(target_dates) != 1:
        return CheckResult(
            "example data",
            False,
            f"expected 106 training and 1 target raster; got {len(train_dates)} and {len(target_dates)}",
        )
    if max(train_dates) >= min(target_dates):
        return CheckResult(
            "example data", False, "target acquisition is not later than training data"
        )
    return CheckResult(
        "example data",
        True,
        "106 training rasters and 1 later target raster verified by SHA-256",
    )


def verify_python_dependencies(require_gpu: bool = True) -> list[CheckResult]:
    required = {
        "numpy": "NumPy",
        "scipy": "SciPy",
        "rasterio": "Rasterio",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
        "matplotlib": "Matplotlib",
        "zstandard": "zstandard",
        "torch": "PyTorch",
    }
    missing = []
    for module_name, display_name in required.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            missing.append(f"{display_name}: {exc}")
    results = [
        CheckResult(
            "Python dependencies",
            not missing,
            "all required packages import successfully"
            if not missing
            else "; ".join(missing),
        )
    ]
    if missing:
        return results

    import torch

    cuda_ok = bool(torch.cuda.is_available())
    cuda_detail = (
        f"CUDA available ({torch.cuda.get_device_name(0)})"
        if cuda_ok
        else "CUDA is not available to PyTorch"
    )
    results.append(CheckResult("CUDA", cuda_ok or not require_gpu, cuda_detail))

    try:
        importlib.import_module("tinycudann")
        tcnn_ok = True
        tcnn_detail = "tiny-cuda-nn PyTorch bindings import successfully"
    except Exception as exc:  # pragma: no cover - environment dependent
        tcnn_ok = False
        tcnn_detail = str(exc)
    results.append(CheckResult("tiny-cuda-nn", tcnn_ok or not require_gpu, tcnn_detail))
    return results


def run_checks(root: Path = PROJECT_ROOT, require_gpu: bool = True) -> list[CheckResult]:
    results = [
        CheckResult(
            "Python version",
            sys.version_info >= (3, 10),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    ]
    results.extend(verify_python_dependencies(require_gpu=require_gpu))
    results.append(verify_source_manifest(root))
    results.append(verify_data_manifest(root))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the SINC environment, synchronized source, and example data"
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Do not fail when CUDA or tiny-cuda-nn is unavailable",
    )
    args = parser.parse_args(argv)

    results = run_checks(require_gpu=not args.skip_gpu)
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
