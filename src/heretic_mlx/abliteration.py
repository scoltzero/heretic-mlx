# SPDX-License-Identifier: AGPL-3.0-or-later
"""MLX implementation of Magnitude-Preserving Orthogonal Ablation (MPOA) from Heretic 2.2."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from optuna import Trial
from rich.console import Console

from .config import (
    AbliterationConfig,
    HereticMLXConfig,
    format_dataset_spec,
    load_prompts,
)
from .model import MLXModel
from .quant import LoRABypassLinear, dequantize_linear

console = Console()


@dataclass
class WeightDistribution:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


@dataclass
class AbliterationParameters:
    direction_index: float | None
    weight_distributions: dict[str, WeightDistribution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbliterationParameters:
        return cls(
            direction_index=data["direction_index"],
            weight_distributions={
                k: WeightDistribution(**v)
                for k, v in data["weight_distributions"].items()
            },
        )


def _randomized_svd_lowrank(
    M: np.ndarray,
    rank: int,
    seed: int,
    niter: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized low-rank SVD matching torch.svd_lowrank(W, q=2*r+4, niter=6)."""
    rng = np.random.default_rng(seed)
    m, n = M.shape
    q = min(2 * rank + 4, m, n)
    omega = rng.standard_normal((n, q)).astype(np.float32)
    Y = M @ omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    for _ in range(niter):
        Z = M.T @ Q
        Q_z, _ = np.linalg.qr(Z, mode="reduced")
        Y = M @ Q_z
        Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ M
    Ub, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    r = min(rank, len(S))
    return U[:, :r], S[:r], Vh[:r, :]


class AbliterationModifier:
    """Heretic 2.2 Projected + Magnitude-Preserving Orthogonal Ablation (MPOA) in MLX."""

    def __init__(self, config: HereticMLXConfig, model: MLXModel):
        self.config = config
        self.settings: AbliterationConfig = config.abliteration
        self.model = model
        self.residual_directions: mx.array | None = None
        self.lora_rank = (
            1
            if self.settings.row_normalization != "full"
            else self.settings.full_normalization_lora_rank
        )

    @property
    def name(self) -> str:
        if self.settings.orthogonalize_direction and self.settings.row_normalization == "full":
            return "Magnitude-Preserving Orthogonal Ablation (MPOA)"
        if self.settings.orthogonalize_direction:
            return "Projected Abliteration"
        return "Abliteration"

    def init(self) -> None:
        console.print(
            f"\nLoading good prompts from [bold]{format_dataset_spec(self.settings.good_prompts)}[/]..."
        )
        good_prompts = load_prompts(self.settings.good_prompts, self.config.system_prompt)
        console.print(f"* [bold]{len(good_prompts)}[/] good prompts loaded")

        console.print(
            f"Loading bad prompts from [bold]{format_dataset_spec(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = load_prompts(self.settings.bad_prompts, self.config.system_prompt)
        console.print(f"* [bold]{len(bad_prompts)}[/] bad prompts loaded")

        console.print("\nCalculating per-layer refusal directions...")
        good_means = self.model.get_residuals_mean(
            good_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
            desc="Good residuals",
        )
        bad_means = self.model.get_residuals_mean(
            bad_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
            desc="Bad residuals",
        )

        diff = bad_means - good_means
        directions = diff / mx.maximum(mx.linalg.norm(diff, axis=1, keepdims=True), 1e-12)

        if self.settings.orthogonalize_direction:
            good_dir = good_means / mx.maximum(
                mx.linalg.norm(good_means, axis=1, keepdims=True), 1e-12
            )
            proj = mx.sum(directions * good_dir, axis=1, keepdims=True)
            directions = directions - proj * good_dir
            directions = directions / mx.maximum(
                mx.linalg.norm(directions, axis=1, keepdims=True), 1e-12
            )

        mx.eval(directions)
        self.residual_directions = directions
        self.model.apply_lora_bypass(self.lora_rank)

    def suggest_parameters(self, trial: Trial) -> AbliterationParameters:
        last_layer = self.model.num_layers - 1
        direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
        direction_index = trial.suggest_float(
            "direction_index", 0.4 * last_layer, 0.9 * last_layer
        )
        if direction_scope == "per layer":
            direction_index = None

        weight_distributions: dict[str, WeightDistribution] = {}
        for comp in self.model.get_modifiable_components():
            low = -0.25 if comp == "mlp.down_proj" else 0.8
            max_w = max(0.0, trial.suggest_float(f"{comp}.max_weight", low, 1.5))
            pos = trial.suggest_float(
                f"{comp}.max_weight_position", 0.6 * last_layer, 1.0 * last_layer
            )
            min_frac = trial.suggest_float(f"{comp}.min_weight", 0.0, 1.0)
            dist = trial.suggest_float(
                f"{comp}.min_weight_distance", 1.0, max(0.6 * last_layer, 1.0)
            )
            weight_distributions[comp] = WeightDistribution(
                max_weight=max_w,
                max_weight_position=pos,
                min_weight=min_frac * max_w,
                min_weight_distance=dist,
            )
        return AbliterationParameters(
            direction_index=direction_index,
            weight_distributions=weight_distributions,
        )

    def reset_model(self) -> None:
        self.model.reset_lora()

    def modify_model(self, parameters: AbliterationParameters) -> None:
        assert self.residual_directions is not None

        global_dir: mx.array | None = None
        if parameters.direction_index is not None:
            frac, idx_f = math.modf(parameters.direction_index + 1.0)
            idx = int(idx_f)
            idx_next = min(idx + 1, self.residual_directions.shape[0] - 1)
            d = (1.0 - frac) * self.residual_directions[idx] + frac * self.residual_directions[
                idx_next
            ]
            global_dir = d / mx.maximum(mx.linalg.norm(d), 1e-12)

        for layer_idx in range(self.model.num_layers):
            for comp, refs in self.model.get_layer_targets(layer_idx).items():
                wd = parameters.weight_distributions[comp]
                distance = abs(layer_idx - wd.max_weight_position)
                if distance > wd.min_weight_distance:
                    continue
                alpha = wd.max_weight + (distance / wd.min_weight_distance) * (
                    wd.min_weight - wd.max_weight
                )
                if alpha == 0.0:
                    continue

                v = (
                    self.residual_directions[layer_idx + 1]
                    if global_dir is None
                    else global_dir
                )
                for ref in refs:
                    mod = ref.get()
                    assert isinstance(mod, LoRABypassLinear)

                    W = dequantize_linear(mod)  # float32 (out, in)
                    W_org = W
                    W_row_norms = mx.linalg.norm(W, axis=1, keepdims=True)

                    if self.settings.row_normalization != "none":
                        W = W / mx.maximum(W_row_norms, 1e-12)

                    lora_A = (v @ W)[None, :]  # (1, in)
                    lora_B = (-alpha * v)[:, None]  # (out, 1)

                    if self.settings.row_normalization == "pre":
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == "full":
                        W_new = W + (lora_B @ lora_A)
                        W_new = W_new / mx.maximum(
                            mx.linalg.norm(W_new, axis=1, keepdims=True), 1e-12
                        )
                        W_new = W_new * W_row_norms
                        W_delta = W_new - W_org
                        mx.eval(W_delta)

                        U, S, Vh = _randomized_svd_lowrank(
                            np.asarray(W_delta, dtype=np.float32),
                            rank=self.lora_rank,
                            seed=self.config.seed,
                        )
                        sqrt_s = np.sqrt(np.maximum(S, 0.0))
                        lora_B = mx.array(U * sqrt_s[None, :], dtype=mx.float32)
                        lora_A = mx.array(sqrt_s[:, None] * Vh, dtype=mx.float32)

                    mod.set_lora(lora_A, lora_B)
                    del W
            mx.clear_cache()
