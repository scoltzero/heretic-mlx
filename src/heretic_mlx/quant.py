# SPDX-License-Identifier: AGPL-3.0-or-later
"""Universal MLX quantization helpers (MXFP4, MXFP8, oQ/affine, BF16) and LoRA bypass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import ExportModifiedMode


@dataclass(frozen=True)
class QuantSpec:
    """Quantization parameters for an MLX QuantizedLinear layer."""

    group_size: int
    bits: int
    mode: str  # "affine" (oQ / MLX default), "mxfp4", "mxfp8", "nvfp4"

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "bits": self.bits,
            "mode": self.mode,
        }


def is_supported_linear(module: Any) -> bool:
    """Return True for 2D MLX Linear / QuantizedLinear layers (excludes 3D SwitchLinear)."""
    if isinstance(module, LoRABypassLinear):
        return True
    return isinstance(module, (nn.Linear, nn.QuantizedLinear))


def get_quant_spec(module: nn.Module) -> QuantSpec | None:
    """Extract QuantSpec from a QuantizedLinear or LoRABypassLinear, or None for unquantized."""
    if isinstance(module, LoRABypassLinear):
        module = module.base
    if isinstance(module, nn.QuantizedLinear):
        return QuantSpec(
            group_size=int(module.group_size),
            bits=int(module.bits),
            mode=str(module.mode),
        )
    return None


def dequantize_linear(module: nn.Module) -> mx.array:
    """Dequantize a Linear or QuantizedLinear (MXFP4, MXFP8, NVFP4, oQ/affine) to float32."""
    if isinstance(module, LoRABypassLinear):
        module = module.base

    if isinstance(module, nn.QuantizedLinear):
        biases = getattr(module, "biases", None)
        weight_f32 = mx.dequantize(
            module.weight,
            module.scales,
            biases,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
            dtype=mx.float32,
        )
        return weight_f32

    if isinstance(module, nn.Linear):
        return module.weight.astype(mx.float32)

    raise TypeError(f"Unsupported linear module type: {type(module).__name__}")


def get_linear_dims(module: nn.Module) -> tuple[int, int]:
    """Return (out_features, in_features) for Linear or QuantizedLinear without full dequantization."""
    if isinstance(module, LoRABypassLinear):
        module = module.base

    if isinstance(module, nn.QuantizedLinear):
        out_features = int(module.weight.shape[0])
        packed_in = int(module.weight.shape[1])
        # MLX packs 32 // bits elements into each uint32 word along axis 1.
        in_features = packed_in * (32 // int(module.bits))
        return out_features, in_features

    if isinstance(module, nn.Linear):
        return int(module.weight.shape[0]), int(module.weight.shape[1])

    raise TypeError(f"Unsupported linear module type: {type(module).__name__}")


def get_linear_bias(module: nn.Module) -> mx.array | None:
    """Return float32 bias tensor if present, else None."""
    if isinstance(module, LoRABypassLinear):
        module = module.base
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    return bias.astype(mx.float32)


def get_linear_compute_dtype(module: nn.Module) -> mx.Dtype:
    """Infer the preferred activation/adapter dtype (usually bfloat16) for a linear layer."""
    if isinstance(module, LoRABypassLinear):
        module = module.base
    if isinstance(module, nn.QuantizedLinear):
        biases = getattr(module, "biases", None)
        if biases is not None:
            return biases.dtype
        if module.scales.dtype in (mx.bfloat16, mx.float16, mx.float32):
            return module.scales.dtype
        return mx.bfloat16
    return module.weight.dtype


def build_quantized_linear(
    weight_f32: mx.array,
    bias_f32: mx.array | None,
    spec: QuantSpec,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> nn.QuantizedLinear:
    """Pack a float32 weight matrix into an MLX QuantizedLinear with the requested spec."""
    out_features, in_features = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    ql = nn.QuantizedLinear(
        input_dims=in_features,
        output_dims=out_features,
        bias=bias_f32 is not None,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    # Cast to compute_dtype (bfloat16) before quantizing so affine scales/biases
    # match the model's native dtype (mxfp4/mxfp8 always emit uint8 E8M0 scales).
    w_cast = weight_f32.astype(compute_dtype)
    q_result = mx.quantize(
        w_cast,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    if len(q_result) == 3:
        ql.weight, ql.scales, ql.biases = q_result
    elif len(q_result) == 2:
        ql.weight, ql.scales = q_result
        ql.biases = None
    else:
        raise ValueError(f"Unexpected mx.quantize output length: {len(q_result)}")

    if bias_f32 is not None:
        ql.bias = bias_f32.astype(compute_dtype)
    return ql


class LoRABypassLinear(nn.Module):
    """Wraps an MLX Linear or QuantizedLinear with an in-memory low-rank delta (B @ A).

    During optimization trials:
    - Base quantized weights (MXFP4 / oQ / BF16) are never mutated or duplicated.
    - Forward pass computes: y = base(x) + ((x @ lora_A.T) @ lora_B.T)
    - Optional I/O capture hooks extract the final token (x[:, -1, :], y[:, -1, :])
      with zero architecture-specific model rewriting.
    """

    def __init__(self, base: nn.Module, rank: int = 50):
        super().__init__()
        if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
            raise TypeError(f"LoRABypassLinear cannot wrap {type(base).__name__}")
        self.base = base
        self.rank = rank
        self.out_features, self.in_features = get_linear_dims(base)
        self.compute_dtype = get_linear_compute_dtype(base)

        # Kept as non-parameter attributes or standard arrays; initialized to zero = identity.
        self.lora_A = mx.zeros((rank, self.in_features), dtype=self.compute_dtype)
        self.lora_B = mx.zeros((self.out_features, rank), dtype=self.compute_dtype)
        self.row_scale: mx.array | None = None
        self._active = False

        # I/O capture state (used once during ARA initialization).
        self._capture_enabled = False
        self._captured_inputs: list[mx.array] = []
        self._captured_outputs: list[mx.array] = []

    def set_lora(
        self,
        lora_A: mx.array,
        lora_B: mx.array,
        row_scale: mx.array | None = None,
    ) -> None:
        """Install new low-rank adapter matrices A (rank, in), B (out, rank), and optional row scale."""
        self.lora_A = lora_A.astype(self.compute_dtype)
        self.lora_B = lora_B.astype(self.compute_dtype)
        self.row_scale = (
            row_scale.astype(self.compute_dtype) if row_scale is not None else None
        )
        self._active = True
        if self.row_scale is not None:
            mx.eval(self.lora_A, self.lora_B, self.row_scale)
        else:
            mx.eval(self.lora_A, self.lora_B)

    def zero_lora(self) -> None:
        """Reset adapter to exact identity without touching base weights."""
        if self._active:
            self.lora_A = mx.zeros(
                (self.rank, self.in_features), dtype=self.compute_dtype
            )
            self.lora_B = mx.zeros(
                (self.out_features, self.rank), dtype=self.compute_dtype
            )
            self.row_scale = None
            self._active = False
            mx.eval(self.lora_A, self.lora_B)

    @property
    def is_modified(self) -> bool:
        return self._active

    def enable_capture(self) -> None:
        self._captured_inputs = []
        self._captured_outputs = []
        self._capture_enabled = True

    def disable_capture(self) -> tuple[mx.array, mx.array]:
        self._capture_enabled = False
        if not self._captured_inputs:
            return mx.zeros((0, self.in_features), dtype=mx.float32), mx.zeros(
                (0, self.out_features), dtype=mx.float32
            )
        inputs = mx.concatenate(self._captured_inputs, axis=0)
        outputs = mx.concatenate(self._captured_outputs, axis=0)
        self._captured_inputs = []
        self._captured_outputs = []
        mx.eval(inputs, outputs)
        return inputs, outputs

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if self._active:
            delta = (x.astype(self.compute_dtype) @ self.lora_A.T) @ self.lora_B.T
            if self.row_scale is not None:
                bias = getattr(self.base, "bias", None)
                if bias is not None:
                    y = (y - bias + delta.astype(y.dtype)) * self.row_scale.astype(y.dtype) + bias
                else:
                    y = (y + delta.astype(y.dtype)) * self.row_scale.astype(y.dtype)
            else:
                y = y + delta.astype(y.dtype)
        if self._capture_enabled:
            # Capture last prompt token (shape: (batch, dim)) in float32.
            x_last = x[:, -1, :].astype(mx.float32) if x.ndim == 3 else x.astype(mx.float32)
            y_last = y[:, -1, :].astype(mx.float32) if y.ndim == 3 else y.astype(mx.float32)
            mx.eval(x_last, y_last)
            self._captured_inputs.append(x_last)
            self._captured_outputs.append(y_last)
        return y

    def effective_lora_matrices(self, seed: int = 42) -> tuple[mx.array, mx.array]:
        """Return (lora_A, lora_B) with any active row_scale folded in via low-rank SVD.

        When `row_scale` is None, returns the exact `(self.lora_A, self.lora_B)` directly.
        When `row_scale` is active (ARA with preserve_row_magnitudes=True), computes the
        exact merged delta `W_eff - W_base = (W_base + B @ A) * row_scale[:, None] - W_base`
        and compresses it to rank `self.rank` via randomized SVD so standard `mlx-lm`
        `LoRALinear` reproduces the row-normalized output natively.
        """
        if self.row_scale is None:
            return self.lora_A, self.lora_B

        W_base = dequantize_linear(self.base)
        delta = self.lora_B.astype(mx.float32) @ self.lora_A.astype(mx.float32)
        W_eff = (W_base + delta) * self.row_scale.astype(mx.float32)[:, None]
        W_delta = W_eff - W_base
        mx.eval(W_delta)

        M = np.asarray(W_delta, dtype=np.float32)
        m, n = M.shape
        r = min(self.rank, m, n)
        rng = np.random.default_rng(seed)
        q = min(r + 8, m, n)
        omega = rng.standard_normal((n, q)).astype(np.float32)
        Q, _ = np.linalg.qr(M @ omega, mode="reduced")
        for _ in range(4):
            Q_z, _ = np.linalg.qr(M.T @ Q, mode="reduced")
            Q, _ = np.linalg.qr(M @ Q_z, mode="reduced")
        Ub, S, Vh = np.linalg.svd(Q.T @ M, full_matrices=False)
        U = (Q @ Ub)[:, :r]
        S = np.maximum(S[:r], 0.0)
        Vh = Vh[:r, :]
        sqrt_s = np.sqrt(S)
        lora_B_eff = mx.array((U * sqrt_s[None, :]).astype(np.float32)).astype(self.compute_dtype)
        lora_A_eff = mx.array((sqrt_s[:, None] * Vh).astype(np.float32)).astype(self.compute_dtype)
        mx.eval(lora_A_eff, lora_B_eff)
        return lora_A_eff, lora_B_eff

    def merge_to_module(
        self,
        export_mode: ExportModifiedMode,
    ) -> tuple[nn.Module, dict[str, Any] | bool | None]:
        """Merge active LoRA delta (and optional row_scale) into base module."""
        orig_spec = get_quant_spec(self.base)

        if not self._active:
            # Unmodified layer: return exact original base module byte-for-byte!
            return self.base, None

        W_base = dequantize_linear(self.base)  # float32 (out, in)
        delta = self.lora_B.astype(mx.float32) @ self.lora_A.astype(mx.float32)
        W_merged = W_base + delta

        if self.row_scale is not None:
            W_merged = W_merged * self.row_scale.astype(mx.float32)[:, None]

        bias_f32 = get_linear_bias(self.base)

        # Resolve AUTO mode:
        # - MXFP4 (4-bit float) -> upgrade modified layers to 6-bit affine (empirically
        #   measured on 27B MXFP4: KL drift 0.0029 vs 0.0020 for affine8, at half the size delta).
        # - oQ / affine (4/5/6/8-bit) -> preserve original per-layer spec (+0.00 GB).
        # - Unquantized BF16 -> keep BF16.
        effective_mode = export_mode
        if effective_mode == ExportModifiedMode.AUTO:
            if orig_spec is None:
                effective_mode = ExportModifiedMode.BF16
            elif orig_spec.mode in ("mxfp4", "nvfp4") or (orig_spec.bits <= 4 and orig_spec.mode != "affine"):
                effective_mode = ExportModifiedMode.AFFINE6
            else:
                effective_mode = ExportModifiedMode.PRESERVE

        if effective_mode == ExportModifiedMode.BF16 or (
            effective_mode == ExportModifiedMode.PRESERVE and orig_spec is None
        ):
            linear = nn.Linear(
                self.in_features, self.out_features, bias=bias_f32 is not None
            )
            linear.weight = W_merged.astype(self.compute_dtype)
            if bias_f32 is not None:
                linear.bias = bias_f32.astype(self.compute_dtype)
            mx.eval(linear.parameters())
            # If the model globally uses quantization, mark this path False so mlx_lm keeps it BF16.
            override: dict[str, Any] | bool | None = False if orig_spec is not None else None
            return linear, override

        if effective_mode == ExportModifiedMode.AFFINE5:
            target_spec = QuantSpec(group_size=64, bits=5, mode="affine")
        elif effective_mode == ExportModifiedMode.AFFINE6:
            target_spec = QuantSpec(group_size=64, bits=6, mode="affine")
        elif effective_mode == ExportModifiedMode.AFFINE8:
            target_spec = QuantSpec(group_size=64, bits=8, mode="affine")
        elif effective_mode == ExportModifiedMode.MXFP8:
            target_spec = QuantSpec(group_size=32, bits=8, mode="mxfp8")
        elif effective_mode == ExportModifiedMode.PRESERVE:
            assert orig_spec is not None
            target_spec = orig_spec
        else:
            raise ValueError(f"Unsupported merge export mode: {export_mode}")

        q_module = build_quantized_linear(
            W_merged,
            bias_f32,
            target_spec,
            compute_dtype=self.compute_dtype,
        )
        mx.eval(q_module.parameters())
        override = (
            target_spec.to_config_dict()
            if (orig_spec is None or target_spec != orig_spec)
            else None
        )
        return q_module, override


class LayerResidualCapture(nn.Module):
    """Transparent wrapper around a decoder layer to capture residual streams for Abliteration.

    Handles standard 3D hidden states (B, S, H), 4D Hyper-Connection streams (B, S, C, H),
    and flattened Hyper-Connection streams (B, S, hc_count * H) like Qwen4-Exp / Qwen3.8-Flash-Next.
    """

    def __init__(self, layer: nn.Module, hidden_size: int | None = None, hc_count: int = 1):
        super().__init__()
        self.layer = layer
        self._hidden_size = hidden_size
        self._hc_count = hc_count
        self._capture_enabled = False
        self._captured_inputs: list[mx.array] = []
        self._captured_outputs: list[mx.array] = []

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute lookups (is_linear, self_attn, linear_attn, mlp, etc.)
        # directly to the wrapped decoder layer without recursing during __init__.
        if (value := self.get(name, None)) is not None:
            return value
        if name != "layer" and "layer" in self:
            return getattr(self["layer"], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def enable_capture(self) -> None:
        self._captured_inputs = []
        self._captured_outputs = []
        self._capture_enabled = True

    def disable_capture(self) -> tuple[list[mx.array], list[mx.array]]:
        self._capture_enabled = False
        ins, outs = self._captured_inputs, self._captured_outputs
        self._captured_inputs = []
        self._captured_outputs = []
        return ins, outs

    def _collapse_last_token(self, h: mx.array) -> mx.array:
        # Handle 4D hyper-connections (B, S, streams, H)
        if h.ndim == 4:
            h = mx.mean(h, axis=2)
        # Handle flattened hyper-connections (B, S, hc_count * H) e.g. Qwen4-Exp
        if (
            self._hidden_size is not None
            and self._hc_count > 1
            and h.shape[-1] == self._hidden_size * self._hc_count
        ):
            h = h.reshape(*h.shape[:-1], self._hc_count, self._hidden_size)
            h = mx.mean(h, axis=-2)
        last = h[:, -1, :] if h.ndim == 3 else h
        return last.astype(mx.float32)

    def __call__(self, x: mx.array, *args: Any, **kwargs: Any) -> Any:
        if self._capture_enabled:
            x_last = self._collapse_last_token(x)
            mx.eval(x_last)
            self._captured_inputs.append(x_last)
        out = self.layer(x, *args, **kwargs)
        if self._capture_enabled:
            h_out = out[0] if isinstance(out, tuple) else out
            y_last = self._collapse_last_token(h_out)
            mx.eval(y_last)
            self._captured_outputs.append(y_last)
        return out
