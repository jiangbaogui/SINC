"""Command-line training entry point for SINC GNDC models."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from ground.core.NDCCompressor import NDCCompressor
from ground.core.NDCConfig import NDCConfig
from ground.core.NDCUtils import group_dated_paths_by_day


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _training_temporal_domain(cfg):
    """Return the first and final observed acquisitions in the training range."""

    dated_groups = group_dated_paths_by_day(Path(cfg.input_folder).rglob("*.tif"))
    dated_groups = [
        (acquisition, paths)
        for acquisition, paths in dated_groups
        if not (
            cfg.global_start_date
            and acquisition.date() < cfg.global_start_date.date()
        )
        and not (
            cfg.global_end_date
            and acquisition.date() > cfg.global_end_date.date()
        )
    ]
    if not dated_groups:
        raise ValueError(
            "No dated training images fall inside the configured temporal range."
        )

    start = dated_groups[0][0]
    end = dated_groups[-1][0]
    if end <= start:
        raise ValueError(
            f"The final training acquisition ({end}) must be after the start ({start})."
        )
    return start, end, dated_groups


def train(config_path: str, output_model_path: str | None = None):
    """Train and export one GNDC model from a YAML configuration."""

    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)

    cfg = NDCConfig(config_path)
    print(f"Project: {cfg.project_name}")
    if not cfg.input_quality_screened:
        raise ValueError(
            "SINC expects quality-screened surface-reflectance inputs. Apply cloud, "
            "cloud-shadow, and snow masking during preprocessing and set "
            "data.input_quality_screened: true in the training configuration."
        )
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)
    print(f"Random seed: {cfg.random_seed} | Ablation mode: {cfg.ablation_mode}")

    if output_model_path:
        cfg.output_model = output_model_path
    output_path_raw = cfg.output_model
    if not output_path_raw:
        print("Error: 'output_model' not found in config.")
        sys.exit(1)

    output_dir = os.path.dirname(output_path_raw)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(cfg.input_folder):
        print(f"Error: Input folder not found: {cfg.input_folder}")
        sys.exit(1)

    temporal_start, final_observation_date, dated_groups = _training_temporal_domain(cfg)
    source_file_count = sum(len(paths) for _, paths in dated_groups)
    print("=== GeoNDC Pipeline Start ===")
    print(
        "Temporal domain: "
        f"{temporal_start.isoformat()} to {final_observation_date.isoformat()} "
        f"({len(dated_groups)} daily observations from {source_file_count} source "
        "files; same-date nanmedian compositing; no extrapolation buffer)"
    )

    comp = NDCCompressor(
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_dtype=cfg.train_dtype,
        global_start_date=temporal_start,
        global_end_date=final_observation_date,
        cache_dir=(
            cfg.cache_dir
            or os.path.join(output_dir or os.getcwd(), ".ndc_cache")
        ),
    )
    comp.load_dataset_folder(
        folder_path=cfg.input_folder,
        expected_bands=cfg.n_bands,
        source_band_indices=cfg.source_band_indices,
        source_band_names=cfg.source_band_names,
        scale_factor=cfg.data_scale,
        valid_range=cfg.input_valid_range,
        p99_removal=cfg.input_p99_removal,
        temporal_bin_days=cfg.temporal_bin_days,
        valid_threshold=cfg.valid_threshold,
    )
    comp.meta_info["last_training_observation_date"] = (
        final_observation_date.isoformat()
    )
    comp.meta_info["input_quality_screened"] = cfg.input_quality_screened
    comp.meta_info["quality_mask_description"] = cfg.quality_mask_description
    print(f"Input quality screening: {cfg.quality_mask_description}")

    comp.prepare_training_data_optimized(
        use_log=cfg.use_log, valid_threshold=cfg.valid_threshold
    )
    comp.build_kplanes_model(
        model_type="CCDC",
        xy_hash_size=cfg.xy_hash_size,
        t_hash_size=cfg.t_hash_size,
        n_levels=cfg.n_levels,
        per_level_scale=cfg.per_level_scale,
        base_resolution=cfg.base_resolution,
        features_per_level=cfg.features_per_level,
        decoder_hidden=cfg.decoder_hidden,
        n_hidden_layers=cfg.n_hidden_layers,
        n_harmonics=cfg.n_harmonics,
        ablation_mode=cfg.ablation_mode,
        anchor_dynamic_state=True,
        time_varying_uncertainty=True,
        horizon_uncertainty=True,
    )

    print("\n--- [Phase 3] Heteroscedastic NLL Training ---")
    comp.fit(
        epochs=cfg.epochs,
        lr=cfg.lr,
        lambda_reg=cfg.lambda_reg,
        lambda_dynamic_magnitude=cfg.lambda_dynamic_magnitude,
    )

    model_name = output_path_raw.replace(".gndc", "_weights.gndc")
    residual_name = output_path_raw.replace(".gndc", "_residuals.gndc")
    print("\n--- [Phase 4] Saving Dual Output ---")
    comp.save_dual_gndc(
        model_filename=model_name,
        residual_filename=residual_name,
        quant_mode=cfg.quant_mode,
        save_residuals=cfg.save_residuals,
        residual_dtype=cfg.residual_dtype,
    )

    del comp
    torch.cuda.empty_cache()
    print("\nTraining complete.")
    print(f" - Weights saved to: {model_name}")
    print(f" - Residuals saved to: {residual_name}")


def main():
    parser = argparse.ArgumentParser(description="SINC model training")
    parser.add_argument("--config", "-c", default="sensor_flood_s2.yaml")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    train(args.config, args.output)


if __name__ == "__main__":
    main()
