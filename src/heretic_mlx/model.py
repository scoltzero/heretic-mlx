# SPDX-License-Identifier: AGPL-3.0-or-later
"""Architecture-agnostic MLX model wrapper supporting MXFP4, oQ, hybrid GDN, and MoE."""

from __future__ import annotations

import copy
import glob
import importlib
import json
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from os.path import commonprefix
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_lm_load
from mlx_lm.tokenizer_utils import load as mlx_load_tokenizer
from mlx_lm.utils import load_model as mlx_load_model
from mlx_lm.utils import save_config, save_model
from rich.console import Console
from tqdm import tqdm

from .config import ExportModifiedMode, HereticMLXConfig, Prompt
from .quant import (
    LayerResidualCapture,
    LoRABypassLinear,
    get_quant_spec,
    is_supported_linear,
)

console = Console()


def _strftime_fixed_date(fmt: str) -> str:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(fmt)


def chat_template_supports_system_role(tokenizer: Any) -> bool:
    sentinel = "HERETIC_SYSTEM_ROLE_PROBE"
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sentinel},
                {"role": "user", "content": "Hi"},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return False
    return sentinel in rendered


@dataclass
class TargetModuleRef:
    """Reference to a single modifiable projection module inside a decoder layer."""

    layer_index: int
    component: str  # "attn.o_proj" or "mlp.down_proj"
    module_index: int
    parent: Any  # parent nn.Module or list owning the attribute/index
    attr_or_idx: str | int  # attribute name on parent, or int index if parent is list
    weight_path: str  # canonical MLX dotted path (matches config['quantization'] keys)

    def get(self) -> nn.Module:
        if isinstance(self.attr_or_idx, int):
            return self.parent[self.attr_or_idx]
        return getattr(self.parent, self.attr_or_idx)

    def set(self, module: nn.Module) -> None:
        if isinstance(self.attr_or_idx, int):
            self.parent[self.attr_or_idx] = module
        else:
            setattr(self.parent, self.attr_or_idx, module)


ModuleIO = list[dict[str, dict[int, tuple[mx.array, mx.array]]]]


def _resolve_model_dir(model_spec: str) -> Path:
    """Resolve a local path or Hugging Face repo ID to a local directory."""
    local = Path(model_spec).expanduser()
    if local.exists():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_spec))


def _discover_layers_container(model: nn.Module) -> tuple[list[nn.Module], str]:
    """Return (layers_list, dotted_prefix) for any mlx-lm model layout."""
    candidates = [
        (("language_model", "model", "layers"), "language_model.model.layers"),
        (("model", "language_model", "layers"), "model.language_model.layers"),
        (("model", "layers"), "model.layers"),
        (("language_model", "layers"), "language_model.layers"),
        (("layers",), "layers"),
    ]
    for attrs, prefix in candidates:
        cur: Any = model
        ok = True
        for attr in attrs:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and len(cur) > 0:
            return cur, prefix

    raise RuntimeError(
        f"Cannot locate decoder layers list on MLX model {type(model).__name__}."
    )


def _discover_layer_targets(
    layer: nn.Module,
    layer_index: int,
    layer_prefix: str,
) -> list[TargetModuleRef]:
    """Discover modifiable attention & MLP output projections on a single layer.

    Mirrors upstream p-e-w/heretic Model.get_layer_modules() while safely skipping
    3D fused SwitchLinear / QuantizedSwitchLinear expert arrays in MLX MoE models.
    """
    # Unwrap residual capture wrapper if installed.
    raw_layer = layer.layer if isinstance(layer, LayerResidualCapture) else layer

    refs: list[TargetModuleRef] = []
    counts: dict[str, int] = {}

    def try_add(component: str, parent: Any, attr_or_idx: str | int, rel_path: str) -> None:
        try:
            mod = (
                parent[attr_or_idx]
                if isinstance(attr_or_idx, int)
                else getattr(parent, attr_or_idx)
            )
        except Exception:
            return
        if not is_supported_linear(mod):
            return
        idx = counts.get(component, 0)
        counts[component] = idx + 1
        refs.append(
            TargetModuleRef(
                layer_index=layer_index,
                component=component,
                module_index=idx,
                parent=parent,
                attr_or_idx=attr_or_idx,
                weight_path=f"{layer_prefix}.{rel_path}",
            )
        )

    # Attention / linear-attention / conv output projections
    with suppress(Exception):
        try_add("attn.o_proj", raw_layer.self_attn, "o_proj", "self_attn.o_proj")
    with suppress(Exception):
        try_add("attn.o_proj", raw_layer.linear_attn, "out_proj", "linear_attn.out_proj")
    with suppress(Exception):
        try_add("attn.o_proj", raw_layer.self_attn, "o_b_proj", "self_attn.o_b_proj")
    with suppress(Exception):
        try_add("attn.o_proj", raw_layer.self_attn, "out_proj", "self_attn.out_proj")
    with suppress(Exception):
        try_add("attn.o_proj", raw_layer.conv, "out_proj", "conv.out_proj")

    # Dense MLP output projection
    with suppress(Exception):
        try_add("mlp.down_proj", raw_layer.mlp, "down_proj", "mlp.down_proj")

    # Shared experts (DeepSeek / GLM plural & Qwen3.5-MoE / Qwen4-Exp / Qwen3.8-Flash-Next singular)
    with suppress(Exception):
        try_add(
            "mlp.down_proj",
            raw_layer.mlp.shared_experts,
            "down_proj",
            "mlp.shared_experts.down_proj",
        )
    with suppress(Exception):
        try_add(
            "mlp.down_proj",
            raw_layer.mlp.shared_expert,
            "down_proj",
            "mlp.shared_expert.down_proj",
        )

    # Per-expert ModuleList MoE (only when experts are individual Linear/QuantizedLinear modules)
    with suppress(Exception):
        experts = raw_layer.mlp.experts
        if isinstance(experts, list):
            for i, expert in enumerate(experts):
                try_add("mlp.down_proj", expert, "down_proj", f"mlp.experts.{i}.down_proj")

    with suppress(Exception):
        experts = raw_layer.block_sparse_moe.experts
        if isinstance(experts, list):
            for i, expert in enumerate(experts):
                try_add("mlp.down_proj", expert, "w2", f"block_sparse_moe.experts.{i}.w2")

    # Liquid LFM feed_forward blocks
    with suppress(Exception):
        try_add("mlp.down_proj", raw_layer.feed_forward, "w2", "feed_forward.w2")
    with suppress(Exception):
        experts = raw_layer.feed_forward.experts
        if isinstance(experts, list):
            for i, expert in enumerate(experts):
                try_add("mlp.down_proj", expert, "w2", f"feed_forward.experts.{i}.w2")

    # IBM Granite MoE Hybrid
    with suppress(Exception):
        try_add(
            "mlp.down_proj",
            raw_layer.shared_mlp,
            "output_linear",
            "shared_mlp.output_linear",
        )
    with suppress(Exception):
        experts = raw_layer.moe.experts
        if isinstance(experts, list):
            for i, expert in enumerate(experts):
                try_add(
                    "mlp.down_proj",
                    expert,
                    "output_linear",
                    f"moe.experts.{i}.output_linear",
                )

    return refs


class MLXModel:
    """High-level wrapper around an mlx-lm model for Heretic 2.2 modification."""

    def __init__(
        self,
        config: HereticMLXConfig,
        *,
        preloaded: tuple[nn.Module, Any, dict[str, Any], Path | None] | None = None,
    ):
        self.config = config
        self.response_prefix: str = config.response_prefix or ""
        self._lora_applied = False
        self._lora_rank = 0
        self._residual_wrappers_installed = False

        if preloaded is not None:
            self.model, self.tokenizer, self.model_config, self.model_dir = preloaded
            self.tokenizer_dir = self.model_dir
        else:
            console.print(f"\nLoading MLX model [bold]{config.model}[/]...")
            self.model_dir = _resolve_model_dir(config.model)
            tokenizer_config = (
                {"trust_remote_code": True} if config.trust_remote_code else {}
            )
            if config.tokenizer is not None:
                self.tokenizer_dir = _resolve_model_dir(config.tokenizer)
                self.model, self.model_config = mlx_load_model(
                    self.model_dir,
                    trust_remote_code=config.trust_remote_code,
                )
                self.tokenizer = mlx_load_tokenizer(
                    self.tokenizer_dir,
                    tokenizer_config_extra=tokenizer_config,
                    eos_token_ids=self.model_config.get("eos_token_id", None),
                )
            else:
                self.tokenizer_dir = self.model_dir
                self.model, self.tokenizer, self.model_config = mlx_lm_load(
                    str(self.model_dir),
                    tokenizer_config=tokenizer_config,
                    return_config=True,
                    trust_remote_code=config.trust_remote_code,
                )

        self.supports_system_role = chat_template_supports_system_role(self.tokenizer)
        self._layers, self._layers_prefix = _discover_layers_container(self.model)
        self.num_layers = len(self._layers)

        # Discover target modules per layer
        self._targets_by_layer: list[dict[str, list[TargetModuleRef]]] = []
        self._all_targets: list[TargetModuleRef] = []
        quant_summary: dict[str, int] = {}
        comp_summary: dict[str, int] = {}
        allowed_components = (
            set(config.target_components) if config.target_components else None
        )

        for i, layer in enumerate(self._layers):
            refs = _discover_layer_targets(layer, i, f"{self._layers_prefix}.{i}")
            if allowed_components is not None:
                refs = [r for r in refs if r.component in allowed_components]
            if not refs:
                if allowed_components is not None:
                    self._targets_by_layer.append({})
                    continue
                raise RuntimeError(f"No modifiable linear modules found in layer {i}.")
            layer_map: dict[str, list[TargetModuleRef]] = {}
            for ref in refs:
                layer_map.setdefault(ref.component, []).append(ref)
                self._all_targets.append(ref)
                comp_summary[ref.component] = comp_summary.get(ref.component, 0) + 1
                spec = get_quant_spec(ref.get())
                qlabel = (
                    f"{spec.mode} {spec.bits}-bit (g={spec.group_size})"
                    if spec is not None
                    else "unquantized"
                )
                quant_summary[qlabel] = quant_summary.get(qlabel, 0) + 1
            self._targets_by_layer.append(layer_map)

        if not self._all_targets:
            raise RuntimeError(
                f"No modifiable modules matched target_components={config.target_components}."
            )

        console.print(
            f"* Loaded MLX model with [bold]{self.num_layers}[/] layers "
            f"(prefix: [cyan]{self._layers_prefix}[/])"
        )
        console.print("* Modifiable components:")
        for comp, count in sorted(comp_summary.items()):
            console.print(f"  * [bold]{comp}[/]: [bold]{count}[/] modules")
        console.print("* Detected target-layer quantization formats:")
        for qlabel, count in sorted(quant_summary.items()):
            console.print(f"  * [bold]{qlabel}[/]: [bold]{count}[/] modules")

    def get_modifiable_components(self) -> list[str]:
        comps: set[str] = set()
        for layer_map in self._targets_by_layer:
            comps.update(layer_map.keys())
        return sorted(comps)

    def get_layer_targets(self, layer_index: int) -> dict[str, list[TargetModuleRef]]:
        return self._targets_by_layer[layer_index]

    def apply_lora_bypass(self, rank: int) -> None:
        """Wrap every modifiable target module with LoRABypassLinear (zero-copy on base weights)."""
        if self._lora_applied and self._lora_rank == rank:
            self.reset_lora()
            return
        if self._lora_applied:
            self.remove_lora_bypass()

        for ref in self._all_targets:
            base = ref.get()
            if isinstance(base, LoRABypassLinear):
                base = base.base
            ref.set(LoRABypassLinear(base, rank=rank))

        self._lora_applied = True
        self._lora_rank = rank

    def remove_lora_bypass(self) -> None:
        """Restore original base modules (discards any unmerged LoRA adapters)."""
        if not self._lora_applied:
            return
        for ref in self._all_targets:
            mod = ref.get()
            if isinstance(mod, LoRABypassLinear):
                ref.set(mod.base)
        self._lora_applied = False
        self._lora_rank = 0

    def reset_lora(self) -> None:
        """Zero out all LoRA bypass weights in O(1) without touching quantized base weights."""
        for ref in self._all_targets:
            mod = ref.get()
            if isinstance(mod, LoRABypassLinear):
                mod.zero_lora()
        mx.clear_cache()

    def _get_text_config_dict(self) -> dict[str, Any]:
        tc = self.model_config.get("text_config")
        if isinstance(tc, dict):
            return tc
        return self.model_config

    def _install_residual_wrappers(self) -> None:
        if self._residual_wrappers_installed:
            return
        tc = self._get_text_config_dict()
        hidden_size = tc.get("hidden_size")
        hc_count = int(tc.get("hc_count", 1))
        for i in range(self.num_layers):
            self._layers[i] = LayerResidualCapture(
                self._layers[i],
                hidden_size=hidden_size,
                hc_count=hc_count,
            )
        self._residual_wrappers_installed = True

    def _remove_residual_wrappers(self) -> None:
        if not self._residual_wrappers_installed:
            return
        for i in range(self.num_layers):
            if isinstance(self._layers[i], LayerResidualCapture):
                self._layers[i] = self._layers[i].layer
        self._residual_wrappers_installed = False

    def _apply_system_role(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.supports_system_role:
            return messages
        result: list[dict[str, str]] = []
        pending: list[str] = []
        for msg in messages:
            if msg["role"] == "system":
                if msg["content"]:
                    pending.append(msg["content"])
                continue
            if pending and msg["role"] == "user":
                msg = dict(msg)
                msg["content"] = "\n\n".join([*pending, msg["content"]])
                pending = []
            result.append(msg)
        if pending:
            result.append({"role": "user", "content": "\n\n".join(pending)})
        return result

    def render_prompt(self, prompt: Prompt, *, include_prefix: bool = True) -> str:
        messages = self._apply_system_role(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        )
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                strftime_now=_strftime_fixed_date,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        if include_prefix and self.response_prefix:
            rendered += self.response_prefix
        return rendered

    def encode_prompt(self, prompt: Prompt, *, include_prefix: bool = True) -> list[int]:
        text = self.render_prompt(prompt, include_prefix=include_prefix)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def detect_response_prefix(self, probe_prompts: list[Prompt] | None = None) -> str:
        """Port of Heretic 2.2 automatic Chain-of-Thought (<think>) prefix detection."""
        if self.config.response_prefix is not None:
            self.response_prefix = self.config.response_prefix
            return self.response_prefix

        console.print("\nChecking chat template for Chain-of-Thought block...")
        dummy_messages: list[dict[str, str]] = []
        if self.supports_system_role:
            dummy_messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        dummy_messages.append({"role": "user", "content": "This is a dummy prompt."})
        dummy_prompt = self.tokenizer.apply_chat_template(
            dummy_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        cot_skip_applied = False
        prefix: str | None = None

        for cot_init, cot_fin in self.config.chain_of_thought_blocks:
            pattern_closed = rf"{re.escape(cot_init)}.*{re.escape(cot_fin)}"
            if re.search(pattern_closed, dummy_prompt, flags=re.DOTALL):
                prefix = ""
                console.print(
                    f"* Chat template already closes CoT block [bold]{cot_init + cot_fin!r}[/]"
                )
                cot_skip_applied = True
                break
            if cot_init in dummy_prompt:
                prefix = cot_fin
                console.print(
                    f"* Chat template opens CoT initializer [bold]{cot_init!r}[/]; "
                    f"setting response prefix to [bold]{cot_fin!r}[/]"
                )
                cot_skip_applied = True
                break

        if prefix is None and probe_prompts:
            console.print("* Probing model outputs for common response / CoT prefix...")
            sample = probe_prompts[: min(16, len(probe_prompts))]
            responses = self.get_responses(sample, max_tokens=24)
            prefix = commonprefix(responses).rstrip(" ")
            for cot_init, cot_fin in self.config.chain_of_thought_blocks:
                if prefix.startswith(cot_init):
                    prefix = cot_init + cot_fin
                    cot_skip_applied = True
                    console.print(f"* Closed CoT block: [bold]{prefix!r}[/]")
                    break

        self.response_prefix = prefix or ""

        if cot_skip_applied and self.response_prefix and probe_prompts:
            sample = probe_prompts[: min(16, len(probe_prompts))]
            responses = self.get_responses(sample, max_tokens=8)
            extra = commonprefix(responses).rstrip(" ")
            if extra:
                self.response_prefix += extra
                console.print(f"* Extended response prefix: [bold]{self.response_prefix!r}[/]")

        return self.response_prefix

    def _forward_last_logits(self, token_ids: list[int]) -> mx.array:
        """Single-prompt prefill forward pass returning float32 last-token logits (vocab,)."""
        ids = mx.array([token_ids], dtype=mx.int32)
        logits = self.model(ids)
        last_logits = logits[0, -1, :].astype(mx.float32)
        mx.eval(last_logits)
        return last_logits

    def get_logits(self, prompts: list[Prompt], desc: str = "Computing logits") -> mx.array:
        """Compute first-token raw logits of shape (len(prompts), vocab) in float32.

        Groups identical-length prompts into small zero-padding batches so SSM/GDN layers
        never see pad tokens while identical-length prompts run in parallel on Metal GPU.
        """
        encoded = [self.encode_prompt(p) for p in prompts]
        n = len(encoded)
        if n == 0:
            return mx.zeros((0, 0), dtype=mx.float32)

        buckets: dict[int, list[int]] = {}
        for idx, toks in enumerate(encoded):
            buckets.setdefault(len(toks), []).append(idx)

        rows: list[mx.array | None] = [None] * n
        max_bs = max(1, min(self.config.eval_batch_size, 8))
        pbar = tqdm(total=n, desc=desc, leave=False)

        for indices in buckets.values():
            for start in range(0, len(indices), max_bs):
                chunk = indices[start : start + max_bs]
                ids = mx.array([encoded[i] for i in chunk], dtype=mx.int32)
                logits = self.model(ids)[:, -1, :].astype(mx.float32)
                mx.eval(logits)
                for local_i, orig_i in enumerate(chunk):
                    rows[orig_i] = logits[local_i]
                pbar.update(len(chunk))

        pbar.close()
        result = mx.stack([r for r in rows if r is not None], axis=0)
        mx.eval(result)
        mx.clear_cache()
        return result

    def get_module_io(
        self,
        prompts: list[Prompt],
        desc: str = "Capturing module I/O",
    ) -> ModuleIO:
        """Capture exact (input, output) at final prompt token for every modifiable module.

        Requires apply_lora_bypass() to have wrapped target modules (with LoRA at zero).
        Uses single-prompt forward passes to avoid any left-padding distortion in SSM/GDN layers.
        """
        if not self._lora_applied:
            raise RuntimeError("apply_lora_bypass() must be called before get_module_io().")

        for ref in self._all_targets:
            mod = ref.get()
            assert isinstance(mod, LoRABypassLinear)
            mod.enable_capture()

        for prompt in tqdm(prompts, desc=desc, leave=False):
            token_ids = self.encode_prompt(prompt)
            ids = mx.array([token_ids], dtype=mx.int32)
            out = self.model(ids)
            mx.eval(out)

        module_io: ModuleIO = [{} for _ in range(self.num_layers)]
        for ref in self._all_targets:
            mod = ref.get()
            assert isinstance(mod, LoRABypassLinear)
            inp, out = mod.disable_capture()
            module_io[ref.layer_index].setdefault(ref.component, {})[ref.module_index] = (
                inp,
                out,
            )

        mx.clear_cache()
        return module_io

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
        desc: str = "Capturing residual means",
    ) -> mx.array:
        """Compute mean per-layer residual stream vector of shape (num_layers + 1, hidden)."""
        import numpy as np

        self._install_residual_wrappers()
        try:
            running_sum: np.ndarray | None = None
            count = 0

            for prompt in tqdm(prompts, desc=desc, leave=False):
                for wrapper in self._layers:
                    assert isinstance(wrapper, LayerResidualCapture)
                    wrapper.enable_capture()

                token_ids = self.encode_prompt(prompt)
                ids = mx.array([token_ids], dtype=mx.int32)
                out = self.model(ids)
                mx.eval(out)

                # Collect [input_to_layer_0 (embedding), output_of_layer_0, ..., output_of_last]
                layer_vecs: list[mx.array] = []
                for idx, wrapper in enumerate(self._layers):
                    assert isinstance(wrapper, LayerResidualCapture)
                    ins, outs = wrapper.disable_capture()
                    if idx == 0:
                        layer_vecs.append(ins[0][0])
                    layer_vecs.append(outs[0][0])

                stacked = mx.stack(layer_vecs, axis=0)  # (num_layers + 1, hidden)
                mx.eval(stacked)
                vec_np = np.asarray(stacked, dtype=np.float64)

                if 0.0 <= winsorization_quantile < 1.0:
                    thresholds = np.quantile(
                        np.abs(vec_np),
                        winsorization_quantile,
                        axis=1,
                        keepdims=True,
                    )
                    vec_np = np.clip(vec_np, -thresholds, thresholds)

                if running_sum is None:
                    running_sum = vec_np
                else:
                    running_sum += vec_np
                count += 1

            assert running_sum is not None and count > 0
            mean_np = (running_sum / count).astype(np.float32)
            mx.clear_cache()
            return mx.array(mean_np, dtype=mx.float32)
        finally:
            self._remove_residual_wrappers()

    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        max_tokens: int | None = None,
        desc: str = "Generating responses",
    ) -> list[str]:
        """Greedy-decode responses using mlx_lm.generate.batch_generate with safe fallback."""
        gen_mod = importlib.import_module("mlx_lm.generate")
        max_new = max_tokens if max_tokens is not None else self.config.max_response_length
        encoded_prompts = [self.encode_prompt(p) for p in prompts]
        greedy_sampler = lambda x: mx.argmax(x, axis=-1)

        # Fast path: mlx_lm batch_generate
        if self.config.eval_batch_size > 1 and len(encoded_prompts) > 1:
            try:
                resp = gen_mod.batch_generate(
                    self.model,
                    self.tokenizer,
                    prompts=encoded_prompts,
                    max_tokens=max_new,
                    verbose=False,
                    completion_batch_size=self.config.eval_batch_size,
                    prefill_batch_size=min(8, self.config.eval_batch_size),
                    sampler=greedy_sampler,
                )
                mx.clear_cache()
                return list(resp.texts)
            except Exception as err:
                console.print(
                    f"[yellow]batch_generate fallback to sequential generate_step ({err})[/]"
                )

        # Universal fallback: generate_step (handles any custom cache / SSM architecture)
        eos_ids: set[int] = set()
        if hasattr(self.tokenizer, "eos_token_ids") and self.tokenizer.eos_token_ids:
            eos_ids.update(self.tokenizer.eos_token_ids)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_ids.add(int(self.tokenizer.eos_token_id))

        texts: list[str] = []
        for token_ids in tqdm(encoded_prompts, desc=desc, leave=False):
            prompt_arr = mx.array(token_ids, dtype=mx.int32)
            out_tokens: list[int] = []
            for tok, _ in gen_mod.generate_step(
                prompt_arr,
                self.model,
                max_tokens=max_new,
                sampler=greedy_sampler,
            ):
                tok_int = int(tok)
                if tok_int in eos_ids:
                    break
                out_tokens.append(tok_int)
            texts.append(self.tokenizer.decode(out_tokens))

        mx.clear_cache()
        return texts

    def stream_chat_response(
        self,
        chat: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        on_token: Any | None = None,
    ) -> str:
        """Stream a greedy chat response for interactive trial testing."""
        gen_mod = importlib.import_module("mlx_lm.generate")
        messages = self._apply_system_role(chat)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        token_ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        eos_ids: set[int] = set()
        if hasattr(self.tokenizer, "eos_token_ids") and self.tokenizer.eos_token_ids:
            eos_ids.update(self.tokenizer.eos_token_ids)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_ids.add(int(self.tokenizer.eos_token_id))

        greedy_sampler = lambda x: mx.argmax(x, axis=-1)
        out_tokens: list[int] = []
        prev_text = ""

        for tok, _ in gen_mod.generate_step(
            mx.array(token_ids, dtype=mx.int32),
            self.model,
            max_tokens=max_tokens,
            sampler=greedy_sampler,
        ):
            tok_int = int(tok)
            if tok_int in eos_ids:
                break
            out_tokens.append(tok_int)
            cur_text = self.tokenizer.decode(out_tokens)
            delta = cur_text[len(prev_text) :]
            if delta:
                if on_token is not None:
                    on_token(delta)
                else:
                    print(delta, end="", flush=True)
                prev_text = cur_text

        if on_token is None:
            print()
        mx.clear_cache()
        return prev_text

    def export(
        self,
        output_dir: str | Path,
        export_mode: ExportModifiedMode | None = None,
        *,
        keep_lora_active: bool = True,
    ) -> Path:
        """Export modified model (or LoRA adapter only) in mlx-lm / omlx compatible format.

        When `keep_lora_active=True` (default), LoRABypassLinear wrappers are preserved on
        `self.model` after saving so the user can immediately chat or switch to another trial
        without reloading the base model from disk.
        """
        out_path = Path(output_dir).expanduser()
        out_path.mkdir(parents=True, exist_ok=True)
        mode = export_mode or self.config.export_modified_mode

        if mode == ExportModifiedMode.ADAPTER:
            return self._export_adapter_only(out_path)

        console.print(
            f"\nMerging modified layers into [bold]{out_path}[/] (mode=[cyan]{mode.value}[/])..."
        )
        export_config = copy.deepcopy(self.model_config)
        quant_cfg = export_config.get("quantization")
        if quant_cfg is None and isinstance(export_config.get("quantization_config"), dict):
            qc = export_config["quantization_config"]
            if "group_size" in qc and "bits" in qc:
                quant_cfg = copy.deepcopy(qc)
            elif qc.get("quant_method") == "mxfp4":
                quant_cfg = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            elif qc.get("quant_method") == "mxfp8":
                quant_cfg = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        if quant_cfg is not None:
            export_config["quantization"] = copy.deepcopy(quant_cfg)

        modified_count = 0
        upgraded_count = 0
        saved_wrappers: list[tuple[TargetModuleRef, LoRABypassLinear]] = []

        for ref in self._all_targets:
            mod = ref.get()
            if isinstance(mod, LoRABypassLinear):
                was_modified = mod.is_modified
                merged_mod, quant_override = mod.merge_to_module(mode)
                if keep_lora_active:
                    saved_wrappers.append((ref, mod))
                ref.set(merged_mod)
                if was_modified:
                    modified_count += 1
                    if quant_override is not None:
                        upgraded_count += 1
                        if "quantization" not in export_config:
                            export_config["quantization"] = {}
                        export_config["quantization"][ref.weight_path] = quant_override

        try:
            # Preserve untouched extra tensors from source safetensors that mlx-lm does not
            # load into the text model (e.g. vision_tower.*, model.visual.*, language_model.mtp.*, mtp.*).
            extra_weights: dict[str, mx.array] = {}
            keep_vision = False
            if self.config.should_preserve_extra_weights() and self.model_dir is not None:
                exported_param_keys = set(dict(tree_flatten(self.model.parameters())).keys())
                # Track base module prefixes owned by the active text model so we never copy back
                # stale .scales/.biases from a layer whose quantization format changed.
                owned_module_prefixes = {
                    k.rsplit(".", 1)[0] for k in exported_param_keys if "." in k
                }
                preserved_groups: dict[str, int] = {}

                for wf in sorted(glob.glob(str(self.model_dir / "*.safetensors"))):
                    loaded = mx.load(wf)
                    for k, v in loaded.items():
                        if k in exported_param_keys:
                            continue
                        mod_prefix = k.rsplit(".", 1)[0] if "." in k else ""
                        if mod_prefix in owned_module_prefixes:
                            continue
                        # Preserve vision towers, MTP heads, and other non-loaded auxiliary weights
                        if k.startswith(
                            (
                                "vision_tower.",
                                "model.visual.",
                                "visual.",
                                "language_model.mtp.",
                                "model.mtp.",
                                "mtp.",
                            )
                        ):
                            extra_weights[k] = v
                            top = ".".join(k.split(".")[:2]) if k.startswith("language_model.") else k.split(".")[0]
                            preserved_groups[top] = preserved_groups.get(top, 0) + 1

                if extra_weights:
                    keep_vision = any(
                        k.startswith(("vision_tower.", "model.visual.", "visual."))
                        for k in extra_weights
                    )
                    summary = ", ".join(f"{g}: {n}" for g, n in sorted(preserved_groups.items()))
                    console.print(
                        f"* Preserving [bold]{len(extra_weights)}[/] untouched auxiliary tensors ({summary})"
                    )

            if extra_weights:
                self._save_model_with_extra_weights(out_path, extra_weights)
            else:
                save_model(out_path, self.model, donate_model=False)
        finally:
            if keep_lora_active:
                for ref, wrapper in saved_wrappers:
                    ref.set(wrapper)
            else:
                self._lora_applied = False
                self._lora_rank = 0
            mx.clear_cache()

        save_config(export_config, out_path / "config.json", keep_vision_config=keep_vision)
        self._copy_tokenizer_and_aux_files(out_path)

        console.print(
            f"* Exported [bold]{modified_count}[/] modified modules "
            f"([bold]{upgraded_count}[/] with per-layer precision override) to [green]{out_path}[/]"
        )
        return out_path

    def _save_model_with_extra_weights(
        self,
        out_path: Path,
        extra_weights: dict[str, mx.array],
    ) -> None:
        """Save model parameters plus untouched extra (e.g. vision) tensors in sharded safetensors."""
        from mlx_lm.utils import make_shards

        weights = dict(tree_flatten(self.model.parameters()))
        weights.update(extra_weights)
        shards = make_shards(weights)
        shards_count = len(shards)
        shard_fmt = (
            "model-{:05d}-of-{:05d}.safetensors"
            if shards_count > 1
            else "model.safetensors"
        )
        total_size = sum(v.nbytes for v in weights.values())
        index_data: dict[str, Any] = {
            "metadata": {"total_size": total_size},
            "weight_map": {},
        }
        for i, shard in enumerate(shards):
            shard_name = shard_fmt.format(i + 1, shards_count)
            mx.save_safetensors(
                str(out_path / shard_name),
                shard,
                metadata={"format": "mlx"},
            )
            for k in shard:
                index_data["weight_map"][k] = shard_name
        index_data["weight_map"] = dict(sorted(index_data["weight_map"].items()))
        with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4)

    def _export_adapter_only(self, out_path: Path) -> Path:
        """Save mlx-lm compatible LoRA adapters (with ARA row_scale folded via SVD)."""
        tensors: dict[str, mx.array] = {}
        modified_paths: list[str] = []
        actual_rank = self._lora_rank

        for ref in self._all_targets:
            mod = ref.get()
            if isinstance(mod, LoRABypassLinear) and mod.is_modified:
                eff_A, eff_B = mod.effective_lora_matrices(seed=self.config.seed)
                actual_rank = int(eff_A.shape[0])
                # mlx_lm.tuner.lora.LoRALinear expects lora_a (in, r) and lora_b (r, out)
                tensors[f"{ref.weight_path}.lora_a"] = eff_A.T
                tensors[f"{ref.weight_path}.lora_b"] = eff_B.T
                modified_paths.append(ref.weight_path)

        mx.save_safetensors(str(out_path / "adapters.safetensors"), tensors)
        adapter_cfg = {
            "base_model": self.config.model,
            "fine_tune_type": "lora",
            "num_layers": 0,
            "lora_parameters": {
                "rank": actual_rank,
                "scale": 1.0,
                "dropout": 0.0,
                "keys": modified_paths,
            },
        }
        with open(out_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_cfg, f, indent=2)
        size_mb = (out_path / "adapters.safetensors").stat().st_size / (1024**2)
        console.print(
            f"* Exported [bold]{len(modified_paths)}[/] mlx-lm LoRA adapters "
            f"([bold]{size_mb:.1f} MB[/]) to [green]{out_path}[/]"
        )
        return out_path

    def verify_exported_model(
        self,
        export_dir: str | Path,
        prompts: list[Prompt],
        *,
        mode: ExportModifiedMode,
        baseline_logprobs: mx.array | None = None,
    ) -> dict[str, float]:
        """Reload an exported checkpoint (or adapter) and measure quantization KL drift & size."""
        from mlx_lm.tuner.utils import load_adapters

        from .scorers import compute_kl_divergence, log_softmax

        exp_path = Path(export_dir).expanduser()
        pre_logits = self.get_logits(prompts, desc="Pre-export logits")
        pre_logprobs = log_softmax(pre_logits)
        mx.eval(pre_logprobs)

        if mode == ExportModifiedMode.ADAPTER:
            assert self.model_dir is not None
            reloaded_m, _ = mlx_load_model(
                self.model_dir, trust_remote_code=self.config.trust_remote_code
            )
            reloaded_m = load_adapters(reloaded_m, str(exp_path))
            export_bytes = (exp_path / "adapters.safetensors").stat().st_size
        else:
            reloaded_m, _ = mlx_load_model(
                exp_path, trust_remote_code=self.config.trust_remote_code
            )
            export_bytes = sum(f.stat().st_size for f in exp_path.glob("*.safetensors"))

        orig_bytes = (
            sum(f.stat().st_size for f in self.model_dir.glob("*.safetensors"))
            if self.model_dir is not None
            else 0
        )

        # Compute logits on reloaded model
        rows: list[mx.array] = []
        for p in prompts:
            toks = self.encode_prompt(p)
            ids = mx.array([toks], dtype=mx.int32)
            out = reloaded_m(ids)[0, -1, :].astype(mx.float32)
            mx.eval(out)
            rows.append(out)
        post_logprobs = log_softmax(mx.stack(rows, axis=0))
        mx.eval(post_logprobs)
        del reloaded_m
        mx.clear_cache()

        quant_kl_drift = compute_kl_divergence(post_logprobs, pre_logprobs)
        metrics: dict[str, float] = {
            "orig_gb": orig_bytes / (1024**3),
            "export_gb": export_bytes / (1024**3),
            "delta_gb": (export_bytes - orig_bytes) / (1024**3),
            "quant_kl_drift": quant_kl_drift,
        }
        if baseline_logprobs is not None:
            metrics["kl_vs_baseline_pre"] = compute_kl_divergence(pre_logprobs, baseline_logprobs)
            metrics["kl_vs_baseline_post"] = compute_kl_divergence(post_logprobs, baseline_logprobs)

        console.print(
            f"* Export verification ([cyan]{mode.value}[/]): "
            f"size=[bold]{metrics['export_gb']:.2f} GB[/] "
            f"({metrics['delta_gb']:+.2f} GB vs orig) | "
            f"re-quant KL drift=[bold]{quant_kl_drift:.5f}[/]"
        )
        if baseline_logprobs is not None:
            console.print(
                f"  * KL vs baseline: pre-export=[bold]{metrics['kl_vs_baseline_pre']:.4f}[/] "
                f"-> post-export=[bold]{metrics['kl_vs_baseline_post']:.4f}[/]"
            )
        return metrics

    def _copy_tokenizer_and_aux_files(self, out_path: Path) -> None:
        search_dirs = [d for d in (self.tokenizer_dir, self.model_dir) if d is not None]
        if not search_dirs:
            return
        aux_names = [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
        ]
        for name in aux_names:
            for src_dir in search_dirs:
                src = src_dir / name
                if src.is_file():
                    shutil.copy2(src, out_path / name)
                    break
