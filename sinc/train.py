"""Public training entry point for SINC."""

from __future__ import annotations

import argparse

from ground.tools.trainer import train

__all__ = ["main", "train"]


def main(argv: list[str] | None = None) -> None:
    """Train a SINC model from an explicit YAML configuration."""

    parser = argparse.ArgumentParser(description="Train a SINC GNDC model")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a SINC YAML configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output path overriding paths.output_model in the YAML file",
    )
    args = parser.parse_args(argv)
    train(args.config, args.output)


if __name__ == "__main__":
    main()
