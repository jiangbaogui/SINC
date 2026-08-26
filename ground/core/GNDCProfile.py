"""Read and update auxiliary calibration metadata in a GNDC container."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Mapping


def read_gndc_header(path: str | Path) -> dict:
    source = Path(path)
    with source.open("rb") as stream:
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ValueError(f"Invalid GNDC header in {source}")
        header_length = struct.unpack("I", raw_length)[0]
        payload = stream.read(header_length)
        if len(payload) != header_length:
            raise ValueError(f"Truncated GNDC header in {source}")
    return json.loads(payload.decode("utf-8"))


def embed_empirical_null_profile(
    model_path: str | Path,
    profile: Mapping | object,
    output_path: str | Path | None = None,
) -> Path:
    """Copy a GNDC while adding a versioned empirical-null profile to its header.

    When ``output_path`` is omitted the source is replaced atomically. Model
    weights and any trailing payload are copied byte-for-byte.
    """

    source = Path(model_path).resolve()
    # Preserve a caller-provided short drive or junction path on Windows.
    # Resolving it here can expand the destination back beyond MAX_PATH before
    # NamedTemporaryFile creates the atomic replacement beside the GNDC file.
    destination = Path(output_path).absolute() if output_path else source
    values = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)

    with source.open("rb") as stream:
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ValueError(f"Invalid GNDC header in {source}")
        old_header_length = struct.unpack("I", raw_length)[0]
        old_header = json.loads(stream.read(old_header_length).decode("utf-8"))
        trailing_payload = stream.read()

    old_header["empirical_null_profile"] = values
    old_header["calibration_profile_type"] = "global_terminal_state_empirical_null"
    encoded_header = json.dumps(
        old_header, ensure_ascii=False, default=str, separators=(",", ":")
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=destination.parent, suffix=".gndc.tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(struct.pack("I", len(encoded_header)))
            temporary.write(encoded_header)
            temporary.write(trailing_payload)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
