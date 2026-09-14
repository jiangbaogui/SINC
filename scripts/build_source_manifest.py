"""Regenerate the source-integrity manifest after an intentional release update."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sinc.preflight import _source_sha256, discover_tracked_source_files


OUTPUT_PATH = PROJECT_ROOT / "SOURCE_SYNC_MANIFEST.json"


def main() -> None:
    relative_paths = discover_tracked_source_files(PROJECT_ROOT)
    payload = {
        "schema_version": 1,
        "canonical_root": ".",
        "file_count": len(relative_paths),
        "files": {
            relative_path: _source_sha256(PROJECT_ROOT / relative_path)
            for relative_path in relative_paths
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(relative_paths)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
