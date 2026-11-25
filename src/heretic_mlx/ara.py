# SPDX-License-Identifier: AGPL-3.0-or-later
"""MLX implementation of Arbitrary-Rank Ablation (ARA) from Heretic 2.2 (Weidmann 2026)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np
from optuna import Trial
from rich.console import Console
from tqdm import tqdm

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
    a_sq = mx.sum(a * a, axis=1, keepdims=True)
    b_sq = mx.sum(b * b, axis=1, keepdims=True).T
    sq_dists = mx.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 1e-12)
    dists = mx.sqrt(sq_dists)
    k_eff = min(k, int(dists.shape[1]))
    nearest = -mx.topk(-dists, k=k_eff, axis=1)
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


def _cubic_interpolate(
    x1: float,
    f1: float,
    g1: float,
    x2: float,
    f2: float,
    g2: float,
    bounds: tuple[float, float] | None = None,
) -> float:
    """Cubic interpolation ported verbatim from PyTorch `torch.optim.lbfgs._cubic_interpolate`."""
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)
    if x1 == x2:
        return (xmin_bound + xmax_bound) / 2.0

    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    d2_square = d1 * d1 - g1 * g2
    if d2_square >= 0.0:
        d2 = math.sqrt(d2_square)
        denom = (g2 - g1 + 2.0 * d2) if x1 <= x2 else (g1 - g2 + 2.0 * d2)
        if abs(denom) > 1e-30:
            if x1 <= x2:
                min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / denom)
            else:
                min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / denom)
            if math.isfinite(min_pos):
                return min(max(min_pos, xmin_bound), xmax_bound)
    return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    dir_eval_fn: Callable[[mx.array, float, mx.array], tuple[float, mx.array, float]],
    x: mx.array,
    t: float,
    d: mx.array,
    d_norm: float,
    f: float,
    g: mx.array,
    gtd: float,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    tolerance_change: float = 1e-9,
    max_ls: int = 25,
) -> tuple[float, mx.array, float, int]:
    """Strong Wolfe line search ported verbatim from PyTorch `torch.optim.lbfgs._strong_wolfe`."""
    f_new, g_new, gtd_new = dir_eval_fn(x, t, d)
    ls_func_evals = 1

    t_prev, f_prev, g_prev, gtd_prev = 0.0, f, g, gtd
    done = False
    ls_iter = 0
    bracket: list[float] = [0.0, t]
    bracket_f: list[float] = [f, f_new]
    bracket_g: list[mx.array] = [g, g_new]
    bracket_gtd: list[float] = [gtd, gtd_new]

    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            bracket_gtd = [gtd_new]
            done = True
            break

        if gtd_new >= 0.0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10.0
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        t_prev = tmp
        f_prev = f_new
        g_prev = g_new
        gtd_prev = gtd_new
        f_new, g_new, gtd_new = dir_eval_fn(x, t, d)
        ls_func_evals += 1
        ls_iter += 1

    if ls_iter == max_ls and not done:
        bracket = [0.0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]

    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        b_max, b_min = max(bracket), min(bracket)
        eps = 0.1 * (b_max - b_min)
        if min(b_max - t, t - b_min) < eps:
            if insuf_progress or t >= b_max or t <= b_min:
                t = (b_max - eps) if abs(t - b_max) < abs(t - b_min) else (b_min + eps)
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new, gtd_new = dir_eval_fn(x, t, d)
        ls_func_evals += 1
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0.0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new
            bracket_gtd[low_pos] = gtd_new

    return bracket_f[low_pos], bracket_g[low_pos], bracket[low_pos], ls_func_evals


@lru_cache(maxsize=32)
def _get_compiled_ara_val_and_grad(
    rank: int,
    in_features: int,
    out_features: int,
    good_n: int,
    bad_n: int,
    preserve_row_magnitudes: bool,
    has_bias: bool,
    k_good: int,
    k_bad: int,
) -> Callable[..., tuple[mx.array, mx.array]]:
    """Compile and cache the MLX Metal forward+backward graph per shape signature."""
    a_size = rank * in_features

    def loss_fn(
        x: mx.array,
        W_base: mx.array,
        W_row_norms_sq: mx.array,
        W_row_norms: mx.array,
        good_input: mx.array,
        good_output: mx.array,
        bad_input: mx.array,
        bad_output: mx.array,
        good_product: mx.array,
        bad_product: mx.array,
        good_out_sq: mx.array,
        bad_out_sq: mx.array,
        weights_vec: mx.array,
        bias: mx.array,
    ) -> mx.array:
        A = x[:a_size].reshape(rank, in_features)
        B = x[a_size:].reshape(out_features, rank)

        new_good = good_product + (good_input @ A.T) @ B.T
        new_bad = bad_product + (bad_input @ A.T) @ B.T

        if preserve_row_magnitudes:
            # Exact low-rank row-norm identity:
            # ||W + B A||_row^2 = ||W||_row^2 + 2 sum((W A^T) * B, 1) + sum((B (A A^T)) * B, 1)
            # Avoids materializing 3x dense (out_features, in_features) float32 buffers per step.
            WA_T = W_base @ A.T
            AA_T = A @ A.T
            eff_sq = (
                W_row_norms_sq
                + 2.0 * mx.sum(B * WA_T, axis=1)
                + mx.sum(B * (B @ AA_T), axis=1)
            )
            eff_row_norms = mx.sqrt(mx.maximum(eff_sq, 1e-12))
            scale = W_row_norms / mx.maximum(eff_row_norms, 1e-12)
            new_good = new_good * scale
            new_bad = new_bad * scale

        if has_bias:
            new_good = new_good + bias
            new_bad = new_bad + bias

        preserve_good = mx.mean((new_good - good_output) ** 2)
        nb_sq = mx.sum(new_bad * new_bad, axis=1, keepdims=True)
        dists_good = mx.sqrt(
            mx.maximum(nb_sq + good_out_sq - 2.0 * (new_bad @ good_output.T), 1e-12)
        )
        dists_bad = mx.sqrt(
            mx.maximum(nb_sq + bad_out_sq - 2.0 * (new_bad @ bad_output.T), 1e-12)
        )
        knn_good = mx.mean(-mx.topk(-dists_good, k=k_good, axis=1))
        knn_bad = mx.mean(-mx.topk(-dists_bad, k=k_bad, axis=1))
        steer_bad = knn_good - weights_vec[2] * knn_bad
        return weights_vec[0] * preserve_good + weights_vec[1] * steer_bad

    return mx.compile(mx.value_and_grad(loss_fn, argnums=0))


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
    learning_rate: float = 1.0,
    max_iter: int = 20,
    history_size: int = 10,
    rng: np.random.Generator,
) -> tuple[mx.array, mx.array, mx.array | None, float]:
    """Optimize low-rank adapter (A, B) natively on MLX Metal GPU using Strong-Wolfe L-BFGS."""
    out_features, in_features = int(W_base.shape[0]), int(W_base.shape[1])
    W_row_norms_sq = mx.sum(W_base * W_base, axis=1)
    W_row_norms = mx.sqrt(W_row_norms_sq)

    # Precompute base products & output squared norms in float32 once per module
    good_product = good_input @ W_base.T
    bad_product = bad_input @ W_base.T
    good_out_sq = mx.sum(good_output * good_output, axis=1, keepdims=True).T
    bad_out_sq = mx.sum(bad_output * bad_output, axis=1, keepdims=True).T
    has_bias = bias is not None
    bias_arr = bias if bias is not None else mx.zeros((1,), dtype=mx.float32)
    weights_vec = mx.array(
        [
            parameters.preserve_good_behavior_weight,
            parameters.steer_bad_behavior_weight,
            parameters.overcorrect_relative_weight,
        ],
        dtype=mx.float32,
    )

    # Initialize A with deterministic Kaiming uniform and B with zeros (matching upstream Heretic)
    A0 = _kaiming_uniform_a((rank, in_features), rng)
    a_size = rank * in_features
    x = mx.concatenate(
        [
            mx.array(A0).reshape(-1),
            mx.zeros((out_features * rank,), dtype=mx.float32),
        ]
    )
    mx.eval(
        W_row_norms_sq,
        W_row_norms,
        good_product,
        bad_product,
        good_out_sq,
        bad_out_sq,
        weights_vec,
        bias_arr,
        x,
    )

    good_n = int(good_output.shape[0])
    bad_n = int(bad_output.shape[0])
    k_good = min(parameters.neighbor_count, good_n)
    k_bad = min(parameters.neighbor_count, bad_n)

    vg_fn = _get_compiled_ara_val_and_grad(
        rank,
        in_features,
        out_features,
        good_n,
        bad_n,
        preserve_row_magnitudes,
        has_bias,
        k_good,
        k_bad,
    )

    static_args = (
        W_base,
        W_row_norms_sq,
        W_row_norms,
        good_input,
        good_output,
        bad_input,
        bad_output,
        good_product,
        bad_product,
        good_out_sq,
        bad_out_sq,
        weights_vec,
        bias_arr,
    )

    def eval_vg(x_vec: mx.array) -> tuple[mx.array, mx.array]:
        return vg_fn(x_vec, *static_args)

    def dir_eval(
        x_base: mx.array, step_t: float, direction: mx.array
    ) -> tuple[float, mx.array, float]:
        x_try = x_base + mx.array(step_t, dtype=mx.float32) * direction
        l_mx, g_mx = eval_vg(x_try)
        gtd_mx = mx.sum(g_mx * direction)
        mx.eval(l_mx, g_mx, gtd_mx)
        lv = float(l_mx.item())
        gv = float(gtd_mx.item())
        if not math.isfinite(lv) or not math.isfinite(gv):
            return 1e12, mx.zeros_like(x_base), 0.0
        return lv, g_mx, gv

    # Exact port of PyTorch `torch.optim.LBFGS(..., lr=learning_rate, max_iter=max_iter,
    # history_size=history_size, line_search_fn="strong_wolfe")` across n_optimization_steps.
    max_eval = max(1, max_iter * 5 // 4)
    tol_grad = 1e-7
    tol_change = 1e-9

    total_n_iter = 0
    old_dirs: list[mx.array] = []
    old_stps: list[mx.array] = []
    ro: list[mx.array] = []
    H_diag = mx.array(1.0, dtype=mx.float32)
    prev_flat_grad: mx.array | None = None
    d = mx.zeros_like(x)
    t = 0.0
    final_loss = 0.0

    for _outer_step in range(n_optimization_steps):
        l_mx, flat_grad = eval_vg(x)
        g_inf_mx = mx.max(mx.abs(flat_grad))
        g_l1_mx = mx.sum(mx.abs(flat_grad))
        mx.eval(l_mx, flat_grad, g_inf_mx, g_l1_mx)

        loss = float(l_mx.item())
        if not math.isfinite(loss):
            break
        final_loss = loss
        if float(g_inf_mx.item()) <= tol_grad:
            break

        current_evals = 1
        n_iter = 0
        g_l1_first = float(g_l1_mx.item())

        while n_iter < max_iter:
            n_iter += 1
            total_n_iter += 1

            if total_n_iter == 1:
                d = -flat_grad
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = mx.array(1.0, dtype=mx.float32)
            else:
                assert prev_flat_grad is not None
                y = flat_grad - prev_flat_grad
                s = d * mx.array(t, dtype=mx.float32)
                ys_mx = mx.sum(y * s)
                yy_mx = mx.sum(y * y)
                mx.eval(y, s, ys_mx, yy_mx)
                ys = float(ys_mx.item())
                if ys > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(mx.array(1.0 / ys, dtype=mx.float32))
                    H_diag = mx.array(
                        ys / max(float(yy_mx.item()), 1e-30), dtype=mx.float32
                    )

                num_old = len(old_dirs)
                q = -flat_grad
                al: list[mx.array] = [mx.array(0.0, dtype=mx.float32)] * num_old
                for i in range(num_old - 1, -1, -1):
                    al_i = mx.sum(old_stps[i] * q) * ro[i]
                    al[i] = al_i
                    q = q - al_i * old_dirs[i]
                r = q * H_diag
                for i in range(num_old):
                    be_i = mx.sum(old_dirs[i] * r) * ro[i]
                    r = r + (al[i] - be_i) * old_stps[i]
                d = r

            prev_flat_grad = flat_grad
            prev_loss = loss

            if total_n_iter == 1:
                t = min(1.0, 1.0 / max(g_l1_first, 1e-30)) * learning_rate
            else:
                t = learning_rate

            gtd_mx = mx.sum(flat_grad * d)
            d_norm_mx = mx.max(mx.abs(d))
            mx.eval(d, gtd_mx, d_norm_mx)
            gtd = float(gtd_mx.item())
            d_norm = float(d_norm_mx.item())

            if gtd > -tol_change:
                break

            x_init = x
            loss, flat_grad, t, ls_evals = _strong_wolfe(
                dir_eval,
                x_init,
                t,
                d,
                d_norm,
                loss,
                flat_grad,
                gtd,
                tolerance_change=tol_change,
                max_ls=max(1, max_eval - current_evals),
            )
            x = x_init + mx.array(t, dtype=mx.float32) * d
            g_inf_mx = mx.max(mx.abs(flat_grad))
            mx.eval(x, g_inf_mx)
            final_loss = loss
            opt_cond = float(g_inf_mx.item()) <= tol_grad

            current_evals += ls_evals
            if n_iter == max_iter or current_evals >= max_eval or opt_cond:
                break
            if abs(t) * d_norm <= tol_change:
                break
            if abs(loss - prev_loss) < tol_change:
                break

    A_opt = x[:a_size].reshape(rank, in_features)
    B_opt = x[a_size:].reshape(out_features, rank)
    row_scale: mx.array | None = None
    if preserve_row_magnitudes:
        WA_T_opt = W_base @ A_opt.T
        AA_T_opt = A_opt @ A_opt.T
        eff_sq_opt = (
            W_row_norms_sq
            + 2.0 * mx.sum(B_opt * WA_T_opt, axis=1)
            + mx.sum(B_opt * (B_opt @ AA_T_opt), axis=1)
        )
        eff_row_norms = mx.sqrt(mx.maximum(eff_sq_opt, 1e-12))
        row_scale = W_row_norms / mx.maximum(eff_row_norms, 1e-12)
        mx.eval(A_opt, B_opt, row_scale)
    else:
        mx.eval(A_opt, B_opt)
    return A_opt, B_opt, row_scale, final_loss


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

        work_items: list[tuple[int, str, Any]] = []
        for layer_index in range(parameters.start_layer_index, parameters.end_layer_index):
            for component, refs in self.model.get_layer_targets(layer_index).items():
                for ref in refs:
                    work_items.append((layer_index, component, ref))

        pbar = tqdm(
            work_items,
            desc="ARA solving",
            unit="mod",
            leave=False,
            disable=len(work_items) <= 4,
        )
        last_layer = -1
        for layer_index, component, ref in pbar:
            if last_layer != -1 and layer_index != last_layer:
                mx.clear_cache()
            last_layer = layer_index

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
                learning_rate=self.settings.learning_rate,
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
            del W_base

        pbar.close()
        mx.clear_cache()
