"""GeoCCDC network with explicit ablation modes."""

import numpy as np
import torch
import torch.nn as nn


try:
    import tinycudann as tcnn

    HAS_TCNN = True
except Exception as exc:
    print(f"[Warning] tinycudann is unavailable: {exc}")
    HAS_TCNN = False


VALID_ABLATION_MODES = {"full", "no_spatial", "no_temporal", "no_prob"}


class GeoCCDCNetwork(nn.Module):
    def __init__(
        self,
        n_bands: int = 7,
        time_span_years: float = 3.0,
        xy_hash_size: int = 19,
        n_levels: int = 16,
        features_per_level: int = 2,
        base_resolution: int = 32,
        per_level_scale: float = 1.5,
        decoder_hidden: int = 128,
        n_steps: int = 1,
        n_harmonics: int = 5,
        n_hidden_layers: int = 2,
        ablation_mode: str = "full",
        anchor_dynamic_state: bool = False,
        time_varying_uncertainty: bool = False,
        horizon_uncertainty: bool = False,
    ):
        super().__init__()
        if ablation_mode not in VALID_ABLATION_MODES:
            choices = ", ".join(sorted(VALID_ABLATION_MODES))
            raise ValueError(f"Unknown ablation_mode={ablation_mode!r}; expected one of: {choices}")

        self.ablation_mode = ablation_mode
        self.n_bands = n_bands
        self.n_harmonics = n_harmonics
        self.omega = 2 * np.pi * time_span_years
        self.uses_hash_grid = ablation_mode != "no_spatial"
        self.uses_dynamic_branch = ablation_mode != "no_temporal"
        self.uses_heteroscedasticity = ablation_mode != "no_prob"
        self.anchor_dynamic_state = bool(anchor_dynamic_state)
        self.time_varying_uncertainty = bool(time_varying_uncertainty)
        self.horizon_uncertainty = bool(horizon_uncertainty)

        self.curve_param_count = 1 + n_harmonics * 4
        uncertainty_param_count = (
            2
            if self.uses_heteroscedasticity and self.time_varying_uncertainty
            else int(self.uses_heteroscedasticity)
        )
        decay_param_count = 1
        self.params_per_static = (
            self.curve_param_count + uncertainty_param_count + decay_param_count
        )
        self.dim_static_total = n_bands * self.params_per_static

        if self.uses_hash_grid:
            if not HAS_TCNN:
                raise RuntimeError(
                    "The full/hash-grid model requires tinycudann. "
                    "Use an environment with tinycudann or select no_spatial."
                )
            network_otype = (
                "FullyFusedMLP"
                if decoder_hidden in [16, 32, 64, 128]
                else "CutlassMLP"
            )
            self.encoder = tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": xy_hash_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                },
            )
            spatial_feature_dim = self.encoder.n_output_dims
            self.decoder_static = tcnn.Network(
                n_input_dims=spatial_feature_dim,
                n_output_dims=self.dim_static_total,
                network_config={
                    "otype": network_otype,
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": decoder_hidden,
                    "n_hidden_layers": n_hidden_layers,
                },
            )
        else:
            # Hash-grid ablation: decode directly from normalized (x, y).
            self.encoder = nn.Identity()
            spatial_feature_dim = 2
            layers = []
            input_dim = spatial_feature_dim
            for _ in range(n_hidden_layers):
                layers.extend([nn.Linear(input_dim, decoder_hidden), nn.ReLU()])
                input_dim = decoder_hidden
            layers.append(nn.Linear(input_dim, self.dim_static_total))
            self.decoder_static = nn.Sequential(*layers)

        if self.uses_dynamic_branch:
            self.decoder_dynamic = nn.Sequential(
                nn.Linear(spatial_feature_dim + 1, 64),
                nn.ReLU(),
                nn.Linear(64, n_bands),
            )
        else:
            self.decoder_dynamic = None

    def _compute_curve(
        self,
        t,
        dynamic_intercept,
        static_params,
        extra_intercept_bias: float = 0.5,
    ):
        value = dynamic_intercept + extra_intercept_bias
        safe_t = torch.clamp(t, max=1.0)

        raw_slope = torch.tanh(static_params[..., 0]) * 0.2
        value = value + raw_slope * safe_t

        for k in range(self.n_harmonics):
            base_idx = 1 + k * 4
            c_base = static_params[..., base_idx]
            s_base = static_params[..., base_idx + 1]
            c_slope = torch.tanh(static_params[..., base_idx + 2]) * 0.02
            s_slope = torch.tanh(static_params[..., base_idx + 3]) * 0.02

            frequency = self.omega * (k + 1)
            amp_c = c_base + c_slope * safe_t
            amp_s = s_base + s_slope * safe_t
            value = value + amp_c * torch.cos(frequency * t)
            value = value + amp_s * torch.sin(frequency * t)

        return value

    def _forward_impl(self, x_spatial, t_time):
        spatial_features = self.encoder(x_spatial)
        static_out = self.decoder_static(spatial_features)

        if self.uses_dynamic_branch:
            safe_t = torch.clamp(t_time, min=0.0, max=1.0)
            dynamic_input = torch.cat([spatial_features.float(), safe_t.float()], dim=-1)
            dynamic_intercept = self.decoder_dynamic(dynamic_input)

            if self.anchor_dynamic_state:
                anchor_t = torch.zeros_like(safe_t)
                anchor_input = torch.cat(
                    [spatial_features.float(), anchor_t.float()], dim=-1
                )
                dynamic_anchor = self.decoder_dynamic(anchor_input)
                dynamic_intercept = dynamic_intercept - dynamic_anchor
            else:
                dynamic_anchor = 0.0

            next_t = torch.clamp(t_time + 0.01, max=1.0)
            next_input = torch.cat([spatial_features.float(), next_t.float()], dim=-1)
            next_intercept = self.decoder_dynamic(next_input)
            if self.anchor_dynamic_state:
                next_intercept = next_intercept - dynamic_anchor
            temporal_deltas = [next_intercept - dynamic_intercept]
        else:
            dynamic_intercept = torch.zeros(
                x_spatial.shape[0],
                self.n_bands,
                dtype=static_out.dtype,
                device=static_out.device,
            )
            temporal_deltas = []

        base_params = static_out.view(-1, self.n_bands, self.params_per_static)
        curve_params = base_params[..., : self.curve_param_count]
        t = t_time.expand(-1, self.n_bands)
        pred_mean = self._compute_curve(t, dynamic_intercept, curve_params)

        next_idx = self.curve_param_count
        if self.uses_heteroscedasticity:
            log_var = base_params[..., next_idx]
            next_idx += 1
            if self.time_varying_uncertainty:
                log_var_slope = torch.tanh(base_params[..., next_idx]) * 2.0
                next_idx += 1
                safe_t = torch.clamp(t, min=0.0, max=1.0)
                log_var = log_var + log_var_slope * (safe_t - 1.0)
        else:
            log_var = torch.zeros_like(pred_mean)

        learned_decay = torch.sigmoid(base_params[..., next_idx]) * 4.9 + 0.1
        if self.uses_heteroscedasticity and self.horizon_uncertainty:
            extrapolation_horizon = torch.relu(t - 1.0)
            log_var = log_var + 2.0 * torch.log1p(
                learned_decay * extrapolation_horizon
            )
        return (
            pred_mean,
            log_var,
            learned_decay,
            temporal_deltas,
            dynamic_intercept,
        )

    def forward(self, x_spatial, t_time):
        """Preserve the historical four-value inference interface."""

        return self._forward_impl(x_spatial, t_time)[:4]

    def forward_with_state(self, x_spatial, t_time):
        """Return predictions together with the learned dynamic state correction.

        The final value is the band-specific additive correction d(x, t), whereas
        ``temporal_deltas`` contains d(x, t + 0.01) - d(x, t). Exposing both
        quantities allows calibration code to distinguish a state level from its
        local temporal variation.
        """

        return self._forward_impl(x_spatial, t_time)


def create_model(
    n_bands: int = 7,
    time_span_years: float = 3.0,
    xy_hash_size: int = 19,
    n_levels: int = 16,
    base_resolution: int = 32,
    per_level_scale: float = 1.5,
    features_per_level: int = 2,
    decoder_hidden: int = 128,
    n_harmonics: int = 5,
    n_hidden_layers: int = 2,
    ablation_mode: str = "full",
    anchor_dynamic_state: bool = False,
    time_varying_uncertainty: bool = False,
    horizon_uncertainty: bool = False,
    device: str = "cuda",
) -> GeoCCDCNetwork:
    model = GeoCCDCNetwork(
        n_bands=n_bands,
        time_span_years=time_span_years,
        xy_hash_size=xy_hash_size,
        n_levels=n_levels,
        base_resolution=base_resolution,
        per_level_scale=per_level_scale,
        features_per_level=features_per_level,
        decoder_hidden=decoder_hidden,
        n_harmonics=n_harmonics,
        n_hidden_layers=n_hidden_layers,
        ablation_mode=ablation_mode,
        anchor_dynamic_state=anchor_dynamic_state,
        time_varying_uncertainty=time_varying_uncertainty,
        horizon_uncertainty=horizon_uncertainty,
    )
    return model.to(device)
