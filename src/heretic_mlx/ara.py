# SPDX-License-Identifier: AGPL-3.0-or-later
"""MLX implementation of Arbitrary-Rank Ablation (ARA) from Heretic 2.2 (Weidmann 2026)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from optuna import Trial
from rich.console import Console
from scipy.optimize import minimize

from .config import ARAConfig, HereticMLXConfig, format_dataset_spec, load_prompts
from .model import MLXModel, ModuleIO
from .quant import (
    LoRABypassLinear,
    dequantize_linear,
    get_linear_bias,
)

console = Console()


@dataclass
class ARAParameters:
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ARAParameters:
        return cls(**data)


def mean_distances_to_knn(a: mx.array, b: mx.array, k: int) -> mx.array:
    """For each row in 2D array `a`, mean Euclidean distance to its `k` nearest rows in `b`."""
    # Pairwise squared Euclidean distance: ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i . b_j
    a_sq = mx.sum(a * a, axis=1, keepdims=True)
    b_sq = mx.sum(b * b, axis=1, keepdims=True).T
    sq_dists = mx.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 1e-12)
    dists = mx.sqrt(sq_dists)
    # Smallest k distances along axis=1
    nearest = mx.sort(dists, axis=1)[:, :k]
    return mx.mean(nearest, axis=1)


def ara_loss(
    good_output: mx.array,
    bad_output: mx.array,
    new_good_output: mx.array,
    new_bad_output: mx.array,
    parameters: ARAParameters,
) -> mx.array:
    """Exact Heretic 2.2 ARA objective function."""
    preserve_good = mx.mean((new_good_output - good_output) ** 2)

    steer_bad = (
        mx.mean(mean_distances_to_knn(new_bad_output, good_output, parameters.neighbor_count))
        + parameters.overcorrect_relative_weight
        * -mx.mean(
            mean_distances_to_knn(new_bad_output, bad_output, parameters.neighbor_count)
        )
    )

    return (
        parameters.preserve_good_behavior_weight * preserve_good
        + parameters.steer_bad_behavior_weight * steer_bad
    )


def _kaiming_uniform_a(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Deterministic Kaiming uniform initialization matching PEFT LoRA A (a = sqrt(5))."""
    fan_in = shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def solve_ara_module(
    W_base: mx.array,
    bias: mx.array | None,
    good_input: mx.array,
    good_output: mx.array,
    bad_input: mx.array,
    bad_output: mx.array,
    rank: int,
    parameters: ARAParameters,
    *,
    preserve_row_magnitudes: bool = True,
    n_optimization_steps: int = 5,
    max_iter: int = 20,
    history_size: int = 10,
    rng: np.random.Generator,
) -> tuple[mx.array, mx.array, mx.array | None, float]:
    """Optimize low-rank adapter (A, B) for a single linear projection using MLX autodiff + L-BFGS-B."""
    out_features, in_features = int(W_base.shape[0]), int(W_base.shape[1])
    W_row_norms = mx.linalg.norm(W_base, axis=1)  # (out_features,)

    # Precompute base products in float32 once (avoids materializing input @ W_eff.T each step)
    good_product = good_input @ W_base.T
    bad_product = bad_input @ W_base.T
    mx.eval(W_row_norms, good_product, bad_product)

    def objective_fn(A: mx.array, B: mx.array) -> mx.array:
        new_good = good_product + (good_input @ A.T) @ B.T
        new_bad = bad_product + (bad_input @ A.T) @ B.T

        if preserve_row_magnitudes:
            eff_row_norms = mx.linalg.norm(W_base + (B @ A), axis=1)
            scale = W_row_norms / mx.maximum(eff_row_norms, 1e-12)
            new_good = new_good * scale
            new_bad = new_bad * scale

        if bias is not None:
            new_good = new_good + bias
            new_bad = new_bad + bias

        return ara_loss(good_output, bad_output, new_good, new_bad, parameters)

    val_and_grad = mx.value_and_grad(objective_fn, argnums=(0, 1))

    # Initialize A with deterministic Kaiming uniform and B with zeros (matching upstream Heretic)
    A0 = _kaiming_uniform_a((rank, in_features), rng)
    B0 = np.zeros((out_features, rank), dtype=np.float32)
    a_size = rank * in_features
    x0 = np.concatenate([A0.ravel(), B0.ravel()]).astype(np.float64)

    def scipy_fun_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_f32 = x.astype(np.float32)
        A = mx.array(x_f32[:a_size].reshape(rank, in_features))
        B = mx.array(x_f32[a_size:].reshape(out_features, rank))
        loss_mx, (gA_mx, gB_mx) = val_and_grad(A, B)
        mx.eval(loss_mx, gA_mx, gB_mx)
        loss_val = float(loss_mx.item())
        grad = np.concatenate(
            [
                np.asarray(gA_mx, dtype=np.float64).ravel(),
                np.asarray(gB_mx, dtype=np.float64).ravel(),
            ]
        )
        if not np.isfinite(loss_val) or not np.all(np.isfinite(grad)):
            return 1e12, np.zeros_like(grad)
        return loss_val, grad

    total_iters = n_optimization_steps * max_iter
    result = minimize(
        scipy_fun_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": total_iters,
            "maxcor": history_size,
            "ftol": 1e-9,
            "gtol": 1e-7,
        },
    )

    x_opt = result.x.astype(np.float32)
    A_opt = mx.array(x_opt[:a_size].reshape(rank, in_features))
    B_opt = mx.array(x_opt[a_size:].reshape(out_features, rank))
    row_scale: mx.array | None = None
    if preserve_row_magnitudes:
        eff_row_norms = mx.linalg.norm(W_base + (B_opt @ A_opt), axis=1)
        row_scale = W_row_norms / mx.maximum(eff_row_norms, 1e-12)
        mx.eval(A_opt, B_opt, row_scale)
    else:
        mx.eval(A_opt, B_opt)
    return A_opt, B_opt, row_scale, float(result.fun)


class ARAModifier:
    """Heretic 2.2 Arbitrary-Rank Ablation modifier for MLX models."""

    name = "Arbitrary-Rank Ablation (ARA)"

    def __init__(self, config: HereticMLXConfig, model: MLXModel):
        self.config = config
        self.settings: ARAConfig = config.ara
        self.model = model
        self.good_module_io: ModuleIO = []
        self.bad_module_io: ModuleIO = []
        self.good_prompt_count = 0
        self.bad_prompt_count = 0

    def init(self) -> None:
        console.print(
            f"\nLoading ARA good prompts from [bold]{format_dataset_spec(self.settings.good_prompts)}[/]..."
        )
        good_prompts = load_prompts(self.settings.good_prompts, self.config.system_prompt)
        console.print(f"* [bold]{len(good_prompts)}[/] good prompts loaded")

        console.print(
            f"Loading ARA bad prompts from [bold]{format_dataset_spec(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = load_prompts(self.settings.bad_prompts, self.config.system_prompt)
        console.print(f"* [bold]{len(bad_prompts)}[/] bad prompts loaded")

        self.good_prompt_count = len(good_prompts)
        self.bad_prompt_count = len(bad_prompts)

        # Install zero-initialized LoRA bypass wrappers (also provides I/O capture hooks)
        self.model.apply_lora_bypass(self.settings.lora_rank)

        console.print("\nCapturing baseline module I/O for good prompts...")
        self.good_module_io = self.model.get_module_io(
            good_prompts, desc="ARA good I/O"
        )
        console.print("Capturing baseline module I/O for bad prompts...")
        self.bad_module_io = self.model.get_module_io(
            bad_prompts, desc="ARA bad I/O"
        )

    def suggest_parameters(self, trial: Trial) -> ARAParameters:
        layer_count = self.model.num_layers
        start_layer_index = trial.suggest_int("start_layer_index", 0, layer_count // 2)
        end_layer_index = trial.suggest_int(
            "end_layer_index", layer_count // 2 + 1, layer_count
        )
        preserve_good_behavior_weight = trial.suggest_float(
            "preserve_good_behavior_weight", 0.0, 1.0
        )
        steer_bad_behavior_weight = trial.suggest_float(
            "steer_bad_behavior_weight", 0.0001, 1.0, log=True
        )
        overcorrect_relative_weight = trial.suggest_float(
            "overcorrect_relative_weight", 0.0, 1.3
        )
        neighbor_count = trial.suggest_int(
            "neighbor_count",
            1,
            min(15, self.good_prompt_count, self.bad_prompt_count),
        )
        return ARAParameters(
            start_layer_index=start_layer_index,
            end_layer_index=end_layer_index,
            preserve_good_behavior_weight=preserve_good_behavior_weight,
            steer_bad_behavior_weight=steer_bad_behavior_weight,
            overcorrect_relative_weight=overcorrect_relative_weight,
            neighbor_count=neighbor_count,
        )

    def reset_model(self) -> None:
        self.model.reset_lora()

    def modify_model(self, parameters: ARAParameters) -> None:
        rng = np.random.default_rng(self.config.seed)

        for layer_index in range(parameters.start_layer_index, parameters.end_layer_index):
            for component, refs in self.model.get_layer_targets(layer_index).items():
                for ref in refs:
                    mod = ref.get()
                    assert isinstance(mod, LoRABypassLinear)

                    # Dequantize ONE projection layer at a time (MXFP4 / oQ / BF16 -> float32)
                    W_base = dequantize_linear(mod)
                    bias = get_linear_bias(mod)
                    good_input, good_output = self.good_module_io[layer_index][component][
                        ref.module_index
                    ]
                    bad_input, bad_output = self.bad_module_io[layer_index][component][
                        ref.module_index
                    ]

                    A_opt, B_opt, row_scale, final_loss = solve_ara_module(
                        W_base=W_base,
                        bias=bias,
                        good_input=good_input,
                        good_output=good_output,
                        bad_input=bad_input,
                        bad_output=bad_output,
                        rank=self.settings.lora_rank,
                        parameters=parameters,
                        preserve_row_magnitudes=self.settings.preserve_row_magnitudes,
                        n_optimization_steps=self.settings.n_optimization_steps,
                        max_iter=self.settings.max_iter,
                        history_size=self.settings.history_size,
                        rng=rng,
                    )
                    mod.set_lora(A_opt, B_opt, row_scale=row_scale)

                    if self.settings.print_loss:
                        console.print(
                            f"  [{layer_index}/{component}/{ref.module_index}] "
                            f"ARA loss={final_loss:.6f}"
                        )
                    # Release temporary float32 dequantized weight promptly
                    del W_base
            mx.clear_cache()
