# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pluggable scorer contract & dynamic loader (compatible with upstream Heretic 2.2)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_type_hints

import mlx.core as mx
from pydantic import BaseModel

from .config import (
    DatasetSpecOrList,
    HereticMLXConfig,
    Prompt,
    deep_merge_dicts,
    load_prompts,
)

if TYPE_CHECKING:
    from .model import MLXModel

T = TypeVar("T")

# Map upstream Heretic 2.2 built-in scorer paths & short names to heretic-mlx equivalents.
BUILTIN_SCORER_ALIASES: dict[str, str] = {
    "KeywordRate": "heretic_mlx.scorers.KeywordRateScorer",
    "KLDivergence": "heretic_mlx.scorers.KLDivergenceScorer",
    "ResponseLength": "heretic_mlx.scorers.ResponseLengthScorer",
    "BERTClassifier": "heretic_mlx.scorers.BERTClassifierScorer",
    "BenchmarkScore": "heretic_mlx.scorers.BenchmarkScorer",
    "heretic.scorers.keyword_rate.KeywordRate": "heretic_mlx.scorers.KeywordRateScorer",
    "heretic.scorers.kl_divergence.KLDivergence": "heretic_mlx.scorers.KLDivergenceScorer",
    "heretic.scorers.response_length.ResponseLength": "heretic_mlx.scorers.ResponseLengthScorer",
    "heretic.scorers.bert_classifier.BERTClassifier": "heretic_mlx.scorers.BERTClassifierScorer",
    "heretic.scorers.benchmark_score.BenchmarkScore": "heretic_mlx.scorers.BenchmarkScorer",
}


@dataclass
class Score:
    """Result of evaluating a single scorer (matches upstream Heretic 2.2 Score)."""

    value: float
    rich_display: str
    md_display: str


class ScorerContext:
    """Runtime context passed to scorers with within-trial response caching."""

    def __init__(self, config: HereticMLXConfig, model: MLXModel) -> None:
        self._config = config
        self._model = model
        self._responses_cache: dict[tuple[tuple[str, str], ...], list[str]] = {}
        self._logits_cache: dict[tuple[tuple[str, str], ...], mx.array] = {}

    @staticmethod
    def _cache_key(prompts: list[Prompt]) -> tuple[tuple[str, str], ...]:
        return tuple((p.system, p.user) for p in prompts)

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        key = self._cache_key(prompts)
        if key not in self._responses_cache:
            self._responses_cache[key] = self._model.get_responses(prompts)
        return self._responses_cache[key]

    def get_logits(self, prompts: list[Prompt]) -> mx.array:
        key = self._cache_key(prompts)
        if key not in self._logits_cache:
            self._logits_cache[key] = self._model.get_logits(prompts)
        return self._logits_cache[key]

    def get_model(self) -> MLXModel:
        return self._model

    def load_prompts(self, spec: DatasetSpecOrList) -> list[Prompt]:
        return load_prompts(spec, self._config.system_prompt)


class Scorer(ABC):
    """Abstract base class for heretic-mlx scorer plugins."""

    # Optional class attribute to declare config table alias (e.g. "KeywordRate" for KeywordRateScorer)
    config_name: str | None = None

    def __init__(
        self,
        *,
        heretic_config: HereticMLXConfig,
        settings: BaseModel | None = None,
    ) -> None:
        self.heretic_config = heretic_config
        self.settings = settings

    @property
    def score_name(self) -> str:
        return self.__class__.__name__

    def init(self, ctx: ScorerContext) -> None:
        """One-time setup before optimization (load prompts, compute baseline state)."""
        return

    @abstractmethod
    def get_score(self, ctx: ScorerContext) -> Score:
        """Evaluate current model and return a Score."""

    def get_baseline_score(self, ctx: ScorerContext) -> Score:
        return self.get_score(ctx)

    @classmethod
    def get_settings_model(cls) -> type[BaseModel] | None:
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = getattr(cls, "__annotations__", {})
        ann = hints.get("settings")
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return ann
        return None

    @classmethod
    def resolve_settings(
        cls,
        heretic_config: HereticMLXConfig,
        instance_name: str | None = None,
        fallback_defaults: dict[str, Any] | None = None,
    ) -> BaseModel | None:
        settings_model = cls.get_settings_model()
        if settings_model is None:
            return None

        extra = heretic_config.model_extra or {}
        scorer_tables = extra.get("scorer", {})
        if not isinstance(scorer_tables, dict):
            scorer_tables = {}

        names_to_try = [cls.__name__]
        if cls.config_name and cls.config_name not in names_to_try:
            names_to_try.insert(0, cls.config_name)

        merged: dict[str, Any] = dict(fallback_defaults or {})
        allowed = set(settings_model.model_fields.keys())

        for base_name in names_to_try:
            table = scorer_tables.get(base_name)
            if isinstance(table, dict):
                merged = deep_merge_dicts(
                    merged, {k: v for k, v in table.items() if k in allowed}
                )
            if instance_name:
                inst_table = scorer_tables.get(f"{base_name}_{instance_name}")
                if isinstance(inst_table, dict):
                    merged = deep_merge_dicts(
                        merged, {k: v for k, v in inst_table.items() if k in allowed}
                    )

        return settings_model.model_validate(merged)


def load_plugin_class(name: str, base_class: type[T]) -> type[T]:
    """Load a plugin class from a built-in alias, dotted import path, or `file.py:ClassName`."""
    resolved = BUILTIN_SCORER_ALIASES.get(name, name)

    if ":" in resolved:
        file_path, class_name = resolved.rsplit(":", 1)
        plugin_path = Path(file_path).expanduser()
        if not plugin_path.is_absolute():
            plugin_path = Path.cwd() / plugin_path
        plugin_path = plugin_path.resolve()
        if not plugin_path.is_file():
            raise ImportError(f"Plugin file '{plugin_path}' does not exist")

        mod_name = f"heretic_mlx_plugin_{abs(hash(str(plugin_path)))}"
        module = sys.modules.get(mod_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(mod_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for '{plugin_path}'")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(mod_name, None)
                raise
        obj = getattr(module, class_name, None)
    else:
        if "." not in resolved:
            raise ValueError(
                f"Unknown scorer '{name}'. Use a built-in name, 'module.Class', or 'file.py:Class'."
            )
        mod_name, class_name = resolved.rsplit(".", 1)
        module = importlib.import_module(mod_name)
        obj = getattr(module, class_name, None)

    if not inspect.isclass(obj) or not issubclass(obj, base_class):
        raise TypeError(f"Plugin '{name}' must be a subclass of {base_class.__name__}")
    return obj
