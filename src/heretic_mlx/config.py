# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration models and prompt dataset loading for heretic-mlx."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAlias

import tomllib
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ModifierKind(str, Enum):
    ARA = "ara"
    ABLITERATION = "abliteration"


class ExportModifiedMode(str, Enum):
    """How modified projection layers are written when exporting quantized models."""

    AUTO = "auto"  # MXFP4 -> affine6; oQ/affine -> preserve; BF16 -> bf16
    AFFINE5 = "affine5"  # Upgrade modified layers to 5-bit affine (group_size=64, ~+0.7 GB on 27B)
    AFFINE6 = "affine6"  # Upgrade modified layers to 6-bit affine (group_size=64, ~+1.5 GB on 27B)
    AFFINE8 = "affine8"  # Upgrade modified layers to 8-bit affine (group_size=64)
    MXFP8 = "mxfp8"  # Upgrade modified layers to 8-bit mxfp8 (group_size=32)
    BF16 = "bf16"  # Store modified layers uncompressed in bfloat16
    PRESERVE = "preserve"  # Re-quantize modified layers back to their original format (+0 GB)
    ADAPTER = "adapter"  # Export mlx-lm compatible LoRA adapter only (~100-250 MB, 0 base rewrite)


@dataclass
class Prompt:
    system: str
    user: str


class DatasetSpec(BaseModel):
    """Specification for loading prompts from Hugging Face or local .txt/.jsonl files."""

    dataset: str
    commit: str | None = None
    split: str | None = None
    column: str | None = "text"
    config: str | None = None
    prefix: str = ""
    suffix: str = ""
    system_prompt: str | None = None


DatasetSpecOrList: TypeAlias = DatasetSpec | list[DatasetSpec]


class ScorerPluginConfig(BaseModel):
    """Configuration for a single scorer plugin entry (upstream Heretic 2.2 compatible)."""

    plugin: str
    optimization: Literal["minimize", "maximize", "none"] = "minimize"
    instance_name: str | None = None


class ARAConfig(BaseModel):
    good_prompts: DatasetSpecOrList = Field(
        default_factory=lambda: DatasetSpec(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        )
    )
    bad_prompts: DatasetSpecOrList = Field(
        default_factory=lambda: DatasetSpec(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        )
    )
    preserve_row_magnitudes: bool = True
    lora_rank: PositiveInt = 50
    n_optimization_steps: PositiveInt = 5
    learning_rate: float = 1.0
    max_iter: PositiveInt = 20
    history_size: PositiveInt = 10
    print_loss: bool = False


class AbliterationConfig(BaseModel):
    good_prompts: DatasetSpecOrList = Field(
        default_factory=lambda: DatasetSpec(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        )
    )
    bad_prompts: DatasetSpecOrList = Field(
        default_factory=lambda: DatasetSpec(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        )
    )
    orthogonalize_direction: bool = True
    row_normalization: str = "full"  # "none" | "pre" | "full"
    full_normalization_lora_rank: PositiveInt = 3
    winsorization_quantile: float = 1.0


class EvalConfig(BaseModel):
    refusal_prompts: DatasetSpecOrList = Field(
        default_factory=lambda: DatasetSpec(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        )
    )
    kl_prompts: DatasetSpecOrList = Field(
        default_factory=lambda: DatasetSpec(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        )
    )
    include_zh_markers: bool = False
    extra_markers: list[str] = Field(default_factory=list)
    print_responses: bool = False


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


class HereticMLXConfig(BaseModel):
    """Main configuration for heretic-mlx.

    Compatible with both native heretic-mlx TOML files and upstream Heretic 2.2
    presets (config.noslop.toml, config.nosyco.toml, config.terse.toml, etc.).
    """

    model_config = ConfigDict(extra="allow")

    model: str
    tokenizer: str | None = None
    output: str | None = None
    trust_remote_code: bool = False
    modifier: ModifierKind = ModifierKind.ARA
    export_modified_mode: ExportModifiedMode = ExportModifiedMode.AUTO
    target_components: list[str] | None = None  # e.g. ["attn.o_proj"] to leave MLPs untouched
    preserve_extra_weights: bool = True  # Keep vision_tower.*, language_model.mtp.*, etc.
    preserve_vision_weights: bool | None = None  # Backward-compatible alias
    verify_export: bool = False  # Reload exported checkpoint & measure quantization KL drift

    modified_model_type: str = "decensored"
    upload_tags: list[str] = Field(
        default_factory=lambda: ["uncensored", "decensored", "abliterated"]
    )

    system_prompt: str = "You are a helpful assistant."
    response_prefix: str | None = None
    chain_of_thought_blocks: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("<think>", "</think>"),
            (
                "<|channel|>analysis<|message|>",
                "<|end|><|start|>assistant<|channel|>final<|message|>",
            ),
            ("<thought>", "</thought>"),
            ("[THINK]", "[/THINK]"),
        ]
    )

    max_response_length: PositiveInt = 100
    eval_batch_size: PositiveInt = 16
    n_trials: PositiveInt = 100
    n_startup_trials: PositiveInt = 30
    seed: int = 42
    checkpoint_dir: str = "checkpoints_mlx"
    auto_select: bool = False
    max_kl_for_auto_select: float | None = None

    # Optional Heretic 2.2 style scorer plugin list.
    # When None, defaults to [KeywordRate (minimize), KLDivergence (minimize)].
    scorers: list[ScorerPluginConfig] | None = None

    ara: ARAConfig = Field(default_factory=ARAConfig)
    abliteration: AbliterationConfig = Field(default_factory=AbliterationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def should_preserve_extra_weights(self) -> bool:
        if self.preserve_vision_weights is not None:
            return self.preserve_vision_weights and self.preserve_extra_weights
        return self.preserve_extra_weights

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> HereticMLXConfig:
        with open(path, "rb") as f:
            raw: dict[str, Any] = tomllib.load(f)

        # Bridge upstream Heretic 2.2 TOML tables ([modifier.ARA], [modifier.Abliteration],
        # `modifiers = [...]`, `study_checkpoint_dir`) transparently into heretic-mlx.
        if "study_checkpoint_dir" in raw and "checkpoint_dir" not in raw:
            raw["checkpoint_dir"] = raw["study_checkpoint_dir"]

        modifiers_list = raw.get("modifiers")
        if isinstance(modifiers_list, list) and modifiers_list and "modifier" not in raw:
            first_plugin = str(modifiers_list[0].get("plugin", "")).lower()
            if "abliteration" in first_plugin:
                raw["modifier"] = ModifierKind.ABLITERATION.value
            else:
                raw["modifier"] = ModifierKind.ARA.value

        mod_table = raw.get("modifier")
        if isinstance(mod_table, dict):
            if "ARA" in mod_table:
                raw["ara"] = deep_merge_dicts(raw.get("ara", {}), mod_table["ARA"])
            if "Abliteration" in mod_table:
                raw["abliteration"] = deep_merge_dicts(
                    raw.get("abliteration", {}), mod_table["Abliteration"]
                )
            # Remove dict so it doesn't collide with ModifierKind field `modifier`
            raw.pop("modifier", None)

        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    raw[k] = v

        return cls.model_validate(raw)


def _apply_slice(items: list[str], split: str | None) -> list[str]:
    if not split or "[" not in split or "]" not in split:
        return items
    slice_spec = split[split.index("[") + 1 : split.index("]")]
    parts = slice_spec.split(":")
    if len(parts) == 1:
        idx = int(parts[0])
        return [items[idx]]
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return items[start:end:step]


def _load_single_dataset(spec: DatasetSpec, default_system_prompt: str) -> list[Prompt]:
    system_prompt = (
        spec.system_prompt if spec.system_prompt is not None else default_system_prompt
    )
    local_path = Path(spec.dataset)
    texts: list[str] = []

    if local_path.is_file():
        suffix = local_path.suffix.lower()
        if suffix == ".jsonl":
            col = spec.column or "prompt"
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        texts.append(obj)
                    elif col in obj:
                        texts.append(str(obj[col]))
                    elif "prompt" in obj:
                        texts.append(str(obj["prompt"]))
                    elif "text" in obj:
                        texts.append(str(obj["text"]))
                    else:
                        raise KeyError(
                            f"Cannot find prompt column {col!r} in {local_path}: keys={list(obj.keys())}"
                        )
        else:
            with open(local_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        texts = _apply_slice(texts, spec.split)
    else:
        ds_kwargs: dict[str, Any] = {"split": spec.split or "train"}
        if spec.commit is not None:
            ds_kwargs["revision"] = spec.commit
        ds = load_dataset(spec.dataset, name=spec.config, **ds_kwargs)
        col = spec.column or "text"
        texts = [str(row[col]) for row in ds]

    return [
        Prompt(system=system_prompt, user=f"{spec.prefix}{t}{spec.suffix}")
        for t in texts
    ]


def load_prompts(spec: DatasetSpecOrList, default_system_prompt: str) -> list[Prompt]:
    """Load prompts from one or more local .txt/.jsonl files or Hugging Face datasets."""
    if isinstance(spec, list):
        prompts: list[Prompt] = []
        for s in spec:
            prompts.extend(_load_single_dataset(s, default_system_prompt))
        return prompts
    return _load_single_dataset(spec, default_system_prompt)


def format_dataset_spec(spec: DatasetSpecOrList) -> str:
    if isinstance(spec, list):
        return ", ".join(s.dataset for s in spec)
    return spec.dataset
