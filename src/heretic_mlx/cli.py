# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI entry point for heretic-mlx: Optuna TPE optimization, interactive chat, reproduce, evaluate, export."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import huggingface_hub
import optuna
import questionary
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, TrialState
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .abliteration import AbliterationModifier, AbliterationParameters
from .ara import ARAModifier, ARAParameters
from .config import (
    DatasetSpec,
    ExportModifiedMode,
    HereticMLXConfig,
    ModifierKind,
    Prompt,
)
from .model import MLXModel
from .reproduce import (
    load_reproduction_manifest,
    save_reproduction_manifest,
    verify_exported_hashes,
)
from .scorers import Evaluator
from .upload import upload_export_to_hf

console = Console()


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="heretic-mlx",
        description="MLX-native Heretic 2.2 (ARA + MPOA | MXFP4 / oQ | Pluggable Scorers)",
    )
    p.add_argument("model", nargs="?", help="Local MLX/MXFP4/oQ model directory or HF repo ID")
    p.add_argument("--tokenizer", type=str, default=None, help="Optional separate tokenizer directory/repo (for weights-only MXFP4 folders)")
    p.add_argument("--config", type=str, default=None, help="Path to TOML config file (supports upstream Heretic 2.2 presets)")
    p.add_argument("--output", "-o", type=str, default=None, help="Output directory for modified model")
    p.add_argument(
        "--modifier",
        choices=[m.value for m in ModifierKind],
        default=None,
        help="Modification algorithm: ara (default, Heretic 2.2) or abliteration (MPOA)",
    )
    p.add_argument(
        "--export-modified-mode",
        choices=[m.value for m in ExportModifiedMode],
        default=None,
        help="Precision for modified layers on export: auto, preserve (+0GB), affine5, affine6, affine8, mxfp8, bf16, adapter",
    )
    p.add_argument(
        "--components",
        type=str,
        default=None,
        help="Comma-separated target components to modify (e.g. 'attn.o_proj' to leave large MLPs untouched)",
    )
    p.add_argument(
        "--verify-export",
        action="store_true",
        help="Reload exported model/adapter after saving and report quantization KL drift & size delta",
    )
    p.add_argument("--evaluate-model", type=str, default=None, help="Evaluate an already-modified MLX model against the base model's baseline")
    p.add_argument("--reproduce", type=str, default=None, help="Reproduce a modified model directly from reproduce.json (skips Optuna search)")
    p.add_argument("--n-trials", type=int, default=None, help="Number of Optuna TPE trials")
    p.add_argument("--n-startup-trials", type=int, default=None, help="Random exploration trials before TPE")
    p.add_argument("--lora-rank", type=int, default=None, help="LoRA rank for ARA (default: 50)")
    p.add_argument("--max-response-length", type=int, default=None, help="Max tokens per refusal eval prompt")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Batch size for refusal eval generation")
    p.add_argument("--response-prefix", type=str, default=None, help="Override auto-detected CoT/response prefix")
    p.add_argument("--good-prompts", type=str, default=None, help="Override good prompts dataset/file")
    p.add_argument("--bad-prompts", type=str, default=None, help="Override bad prompts dataset/file")
    p.add_argument("--eval-refusal-prompts", type=str, default=None, help="Override refusal eval dataset/file")
    p.add_argument("--eval-kl-prompts", type=str, default=None, help="Override KL eval dataset/file")
    p.add_argument("--zh-refusals", action="store_true", help="Include Chinese refusal markers")
    p.add_argument("--trust-remote-code", action="store_true", help="Allow custom model_file / remote tokenizer code")
    p.add_argument("--no-preserve-extra-weights", "--no-preserve-vision", dest="no_preserve_extra_weights", action="store_true", help="Do not copy untouched vision_tower / mtp tensors from source checkpoint")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for Optuna journal checkpoints")
    p.add_argument("--auto-select", action="store_true", help="Non-interactively pick best Pareto trial and export")
    p.add_argument("--max-kl", type=float, default=None, help="Upper KL bound when using --auto-select")
    p.add_argument("--smoke", action="store_true", help="Quick 3-trial smoke test with small prompt slices")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    return p


def _select_auto_trial(
    pareto_trials: list[FrozenTrial],
    directions: list[str],
    max_kl: float | None = None,
) -> FrozenTrial:
    """Pick best Pareto trial (primary objective first, tie-breaking by secondary)."""
    candidates = pareto_trials
    if max_kl is not None and len(directions) >= 2:
        within = [t for t in pareto_trials if t.values[1] <= max_kl]
        if within:
            candidates = within

    def sort_key(t: FrozenTrial) -> tuple[float, ...]:
        return tuple(
            v if d == "minimize" else -v
            for v, d in zip(t.values, directions, strict=True)
        )

    return min(candidates, key=sort_key)


def _format_trial_label(trial: FrozenTrial) -> str:
    scores = trial.user_attrs.get("scores", [])
    parts = [
        f"{s['name']}: {Text.from_markup(s['score']['rich_display']).plain}"
        for s in scores
    ]
    return f"Trial {trial.number + 1:>3} | " + ", ".join(parts)


def _run_interactive_chat(model: MLXModel, system_prompt: str) -> None:
    console.print(
        "\n[bold cyan]Interactive chat[/] (type [bold]/clear[/] to reset history, [bold]/exit[/] to return):"
    )
    chat: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            break
        if user_input.lower() == "/clear":
            chat = [{"role": "system", "content": system_prompt}]
            console.print("[dim]Chat history cleared.[/]")
            continue
        chat.append({"role": "user", "content": user_input})
        console.print("[bold green]Model >[/] ", end="")
        reply = model.stream_chat_response(chat)
        chat.append({"role": "assistant", "content": reply})


def _make_modifier(
    config: HereticMLXConfig,
    model: MLXModel,
) -> ARAModifier | AbliterationModifier:
    if config.modifier == ModifierKind.ARA:
        return ARAModifier(config, model)
    return AbliterationModifier(config, model)


def _apply_raw_params(
    config: HereticMLXConfig,
    modifier: ARAModifier | AbliterationModifier,
    raw_params: dict[str, Any],
) -> None:
    modifier.reset_model()
    if config.modifier == ModifierKind.ARA:
        assert isinstance(modifier, ARAModifier)
        modifier.modify_model(ARAParameters.from_dict(raw_params))
    else:
        assert isinstance(modifier, AbliterationModifier)
        modifier.modify_model(AbliterationParameters.from_dict(raw_params))


def _maybe_verify_export(
    model: MLXModel,
    evaluator: Evaluator,
    export_path: Path,
    mode: ExportModifiedMode,
) -> None:
    """Find KL scorer prompts (or fallback prompts) and run post-export quantization drift check."""
    from .scorers import KLDivergenceScorer

    prompts: list[Prompt] | None = None
    baseline_logprobs = None
    for entry in evaluator.entries:
        if isinstance(entry.scorer, KLDivergenceScorer):
            prompts = entry.scorer.prompts
            baseline_logprobs = entry.scorer.baseline_logprobs
            break
    if not prompts:
        prompts = [
            Prompt(system=model.config.system_prompt, user="What is the capital of France?"),
            Prompt(system=model.config.system_prompt, user="Explain recursion in one sentence."),
        ]
    model.verify_exported_model(
        export_path,
        prompts,
        mode=mode,
        baseline_logprobs=baseline_logprobs,
    )


def evaluate_existing_model(
    config: HereticMLXConfig,
    target_model_spec: str,
) -> list[dict[str, Any]]:
    """Compute baseline on `config.model`, then evaluate `target_model_spec` on identical scorers."""
    console.print(
        f"[bold cyan]heretic-mlx evaluate mode[/]: baseline=[bold]{config.model}[/] -> target=[bold]{target_model_spec}[/]"
    )
    base_model = MLXModel(config)
    probe_prompts = [
        Prompt(system=config.system_prompt, user="What is the capital of France?"),
        Prompt(system=config.system_prompt, user="Tell me how to pick a lock."),
    ]
    base_model.detect_response_prefix(probe_prompts)
    evaluator = Evaluator(config, base_model)

    # Free base model before loading target model
    saved_prefix = base_model.response_prefix
    del base_model

    target_cfg = config.model_copy(
        update={"model": target_model_spec, "response_prefix": saved_prefix}
    )
    target_model = MLXModel(target_cfg)
    evaluator.model = target_model

    console.print(f"\nEvaluating [bold]{target_model_spec}[/]...")
    scores = evaluator.get_scores()
    records = evaluator.get_paired_records(scores)

    table = Table(title=f"Evaluation: {Path(target_model_spec).name}")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Target Model", justify="right")
    for rec in records:
        table.add_row(
            rec["name"],
            rec["baseline"]["rich_display"],
            rec["score"]["rich_display"],
        )
    console.print()
    console.print(table)
    return records


def reproduce_from_manifest(
    manifest_path: str | Path,
    *,
    output_override: str | None = None,
    model_override: str | None = None,
    export_mode_override: ExportModifiedMode | None = None,
) -> Path:
    """Reproduce a modified model directly from reproduce.json without running Optuna."""
    manifest = load_reproduction_manifest(manifest_path)
    cfg = HereticMLXConfig.model_validate(manifest["config"])
    if model_override:
        cfg.model = model_override
    if export_mode_override is not None:
        cfg.export_modified_mode = export_mode_override
    cfg.response_prefix = manifest.get("response_prefix", cfg.response_prefix)

    console.print(
        f"[bold cyan]heretic-mlx reproduce mode[/] from [bold]{manifest_path}[/]"
    )
    model = MLXModel(cfg)
    modifier = _make_modifier(cfg, model)
    modifier.init()
    _apply_raw_params(cfg, modifier, manifest["parameters"])

    out_dir = output_override or cfg.output or f"{Path(cfg.model).name}-reproduced-mlx"
    export_path = model.export(out_dir, cfg.export_modified_mode)
    console.print("\nVerifying exported file hashes against manifest:")
    verify_exported_hashes(export_path, manifest)
    return export_path


def run_heretic_mlx(config: HereticMLXConfig) -> Path | None:
    """Execute full Heretic-MLX optimization, interactive testing, and export pipeline."""
    console.print(
        "[bold cyan]heretic-mlx[/] — MLX-native Heretic 2.2 (ARA + MPOA | MXFP4 / oQ)"
    )

    model = MLXModel(config)
    modifier = _make_modifier(config, model)

    probe_prompts = [
        Prompt(system=config.system_prompt, user="What is the capital of France?"),
        Prompt(system=config.system_prompt, user="Explain photosynthesis in one sentence."),
        Prompt(system=config.system_prompt, user="Write a Python function to add two numbers."),
        Prompt(system=config.system_prompt, user="Tell me how to pick a lock."),
    ]
    model.detect_response_prefix(probe_prompts)

    evaluator = Evaluator(config, model)
    directions = evaluator.get_objective_directions()
    if not directions:
        raise ValueError(
            "No optimization objectives configured. Set at least one scorer to 'minimize' or 'maximize'."
        )

    modifier.init()

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    study_key = hashlib.sha256(
        f"{config.model}:{config.modifier.value}:{'/'.join(evaluator.get_objective_names())}".encode()
    ).hexdigest()[:12]
    journal_path = ckpt_dir / f"study_{study_key}.jsonl"
    lock_obj = JournalFileOpenLock(str(journal_path))
    storage = JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock_obj))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="heretic_mlx",
        storage=storage,
        load_if_exists=True,
        directions=directions,
        sampler=TPESampler(
            n_startup_trials=config.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            seed=config.seed,
        ),
    )

    completed_before = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    if completed_before > 0:
        console.print(
            f"\n[yellow]Resuming checkpoint[/] ([bold]{completed_before}[/]/{config.n_trials} trials completed) from {journal_path}"
        )

    def _run_trials(n_to_run: int) -> None:
        if n_to_run <= 0:
            return
        start_time = time.perf_counter()
        done_session = 0
        target_total = len([t for t in study.trials if t.state == TrialState.COMPLETE]) + n_to_run

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            nonlocal done_session
            params = modifier.suggest_parameters(trial)
            trial.set_user_attr("parameters", params.to_dict())

            console.print(
                f"\n[magenta]Trial {trial.number + 1}/{target_total}[/] ({modifier.name})"
            )
            for k, v in params.to_dict().items():
                if isinstance(v, float):
                    console.print(f"  * {k} = [bold]{v:.4f}[/]")
                else:
                    console.print(f"  * {k} = [bold]{v}[/]")

            modifier.reset_model()
            t0 = time.perf_counter()
            modifier.modify_model(params)
            t_mod = time.perf_counter() - t0

            scores = evaluator.get_scores()
            obj_vals = evaluator.get_objective_values(scores)
            records = evaluator.get_paired_records(scores)
            trial.set_user_attr("scores", records)

            t_total = time.perf_counter() - t0
            done_session += 1
            elapsed = time.perf_counter() - start_time
            eta = (elapsed / done_session) * max(0, n_to_run - done_session)

            score_str = " | ".join(f"{name}: [green]{s.rich_display}[/]" for name, s in scores)
            console.print(
                f"  => {score_str} | mod={t_mod:.1f}s, total={t_total:.1f}s | ETA={_format_duration(eta)}"
            )
            return obj_vals

        try:
            study.optimize(objective, n_trials=n_to_run)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user — proceeding to Pareto selection.[/]")

    _run_trials(max(0, config.n_trials - completed_before))

    last_export: Path | None = None

    while True:
        pareto_trials = sorted(
            study.best_trials,
            key=lambda t: tuple(
                v if d == "minimize" else -v
                for v, d in zip(t.values, directions, strict=True)
            ),
        )
        if not pareto_trials:
            console.print("[red]No completed trials available.[/]")
            return last_export

        table = Table(title="Pareto-Optimal Trials")
        table.add_column("Trial", justify="right")
        for obj_name in evaluator.get_objective_names():
            table.add_column(obj_name, justify="right")
        table.add_column("Parameters")
        for t in pareto_trials:
            by_name = {s["name"]: s["score"]["rich_display"] for s in t.user_attrs.get("scores", [])}
            row_vals = [by_name.get(n, f"{v:.4f}") for n, v in zip(evaluator.get_objective_names(), t.values, strict=True)]
            table.add_row(
                str(t.number + 1),
                *row_vals,
                json.dumps(t.user_attrs.get("parameters", {}), ensure_ascii=False)[:80],
            )
        console.print()
        console.print(table)

        if config.auto_select:
            chosen = _select_auto_trial(
                pareto_trials, directions, config.max_kl_for_auto_select
            )
            console.print(f"\nAuto-selected [bold]{_format_trial_label(chosen)}[/]")
            raw_params = chosen.user_attrs["parameters"]
            _apply_raw_params(config, modifier, raw_params)
            out_dir = config.output or f"{Path(config.model).name or 'model'}-heretic-mlx"
            last_export = model.export(out_dir, config.export_modified_mode)
            if config.verify_export:
                _maybe_verify_export(model, evaluator, last_export, config.export_modified_mode)
            save_reproduction_manifest(
                last_export,
                config=config,
                modifier_name=modifier.name,
                parameters=raw_params,
                scores=chosen.user_attrs.get("scores", []),
                response_prefix=model.response_prefix,
                selected_trial=chosen.number + 1,
            )
            return last_export

        choices: list[questionary.Choice] = [
            questionary.Choice(title=_format_trial_label(t), value=t)
            for t in pareto_trials
        ]
        choices.append(questionary.Choice(title="Run additional trials", value="continue"))
        choices.append(questionary.Choice(title="Exit program", value=None))

        chosen = questionary.select(
            "Which Pareto trial do you want to inspect / export?",
            choices=choices,
        ).ask()

        if chosen is None:
            return last_export

        if chosen == "continue":
            raw_n = questionary.text("How many additional trials?", default="20").ask()
            if raw_n and raw_n.strip().isdigit() and int(raw_n.strip()) > 0:
                _run_trials(int(raw_n.strip()))
            continue

        raw_params = chosen.user_attrs["parameters"]
        scores_rec = chosen.user_attrs.get("scores", [])
        console.print(f"\nApplying [bold]Trial {chosen.number + 1}[/]...")
        _apply_raw_params(config, modifier, raw_params)

        # Action submenu on the active trial (zero reload needed when switching trials!)
        while True:
            action = questionary.select(
                f"What do you want to do with Trial {chosen.number + 1} ({config.modified_model_type})?",
                choices=[
                    questionary.Choice("Chat with the modified model", value="chat"),
                    questionary.Choice("Save the model to a local folder", value="save"),
                    questionary.Choice("Upload the model to Hugging Face", value="upload"),
                    questionary.Choice("Return to trial selection menu", value="back"),
                ],
            ).ask()

            if action is None or action == "back":
                break

            if action == "chat":
                _run_interactive_chat(model, config.system_prompt)

            elif action == "save":
                default_out = config.output or f"{Path(config.model).name or 'model'}-heretic-mlx"
                out_dir = questionary.text("Output folder path:", default=default_out).ask()
                if not out_dir:
                    continue
                mode_choice = questionary.select(
                    "Export precision for modified layers:",
                    choices=[
                        questionary.Choice("auto (MXFP4 -> affine6, oQ -> preserve)", ExportModifiedMode.AUTO),
                        questionary.Choice("preserve (re-pack to original format, +0 GB)", ExportModifiedMode.PRESERVE),
                        questionary.Choice("affine5 (5-bit affine, g=64, compact)", ExportModifiedMode.AFFINE5),
                        questionary.Choice("affine6 (6-bit affine, g=64, balanced)", ExportModifiedMode.AFFINE6),
                        questionary.Choice("affine8 (8-bit affine, g=64, high precision)", ExportModifiedMode.AFFINE8),
                        questionary.Choice("mxfp8 (8-bit mxfp8, g=32)", ExportModifiedMode.MXFP8),
                        questionary.Choice("bf16 (uncompressed modified layers)", ExportModifiedMode.BF16),
                        questionary.Choice("adapter (mlx-lm LoRA adapter only, ~100-250 MB)", ExportModifiedMode.ADAPTER),
                    ],
                    default=config.export_modified_mode,
                ).ask()
                if mode_choice is None:
                    continue
                last_export = model.export(out_dir, mode_choice, keep_lora_active=True)
                if config.verify_export:
                    _maybe_verify_export(model, evaluator, last_export, mode_choice)
                save_reproduction_manifest(
                    last_export,
                    config=config.model_copy(update={"export_modified_mode": mode_choice}),
                    modifier_name=modifier.name,
                    parameters=raw_params,
                    scores=scores_rec,
                    response_prefix=model.response_prefix,
                    selected_trial=chosen.number + 1,
                )

            elif action == "upload":
                token = huggingface_hub.get_token() or questionary.password(
                    "Hugging Face access token:"
                ).ask()
                if not token:
                    continue
                user = huggingface_hub.whoami(token)
                default_repo = f"{user['name']}/{Path(config.model).name}-heretic-mlx"
                repo_id = questionary.text("Hugging Face repo ID:", default=default_repo).ask()
                if not repo_id:
                    continue
                private = questionary.confirm("Private repository?", default=False).ask()
                with tempfile.TemporaryDirectory(prefix="heretic_mlx_upload_") as tmp_dir:
                    exp_path = model.export(tmp_dir, config.export_modified_mode, keep_lora_active=True)
                    save_reproduction_manifest(
                        exp_path,
                        config=config,
                        modifier_name=modifier.name,
                        parameters=raw_params,
                        scores=scores_rec,
                        response_prefix=model.response_prefix,
                        selected_trial=chosen.number + 1,
                    )
                    upload_export_to_hf(
                        exp_path,
                        repo_id=repo_id,
                        token=token,
                        private=bool(private),
                        config=config,
                        modifier_name=modifier.name,
                        scores=scores_rec,
                        parameters=raw_params,
                    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.reproduce:
        reproduce_from_manifest(
            args.reproduce,
            output_override=args.output,
            model_override=args.model,
            export_mode_override=(
                ExportModifiedMode(args.export_modified_mode)
                if args.export_modified_mode
                else None
            ),
        )
        return

    if args.config:
        overrides: dict[str, Any] = {}
        if args.model:
            overrides["model"] = args.model
        if args.output:
            overrides["output"] = args.output
        config = HereticMLXConfig.from_toml(args.config, overrides=overrides)
    else:
        if not args.model:
            parser.error("Provide a model path/repo ID, --config <file.toml>, or --reproduce <reproduce.json>.")
        config = HereticMLXConfig(model=args.model)

    if args.output:
        config.output = args.output
    if args.tokenizer:
        config.tokenizer = args.tokenizer
    if args.modifier:
        config.modifier = ModifierKind(args.modifier)
    if args.export_modified_mode:
        config.export_modified_mode = ExportModifiedMode(args.export_modified_mode)
    if args.components:
        config.target_components = [
            c.strip() for c in args.components.split(",") if c.strip()
        ]
    if args.verify_export:
        config.verify_export = True
    if args.n_trials is not None:
        config.n_trials = args.n_trials
    if args.n_startup_trials is not None:
        config.n_startup_trials = args.n_startup_trials
    if args.lora_rank is not None:
        config.ara.lora_rank = args.lora_rank
    if args.max_response_length is not None:
        config.max_response_length = args.max_response_length
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size
    if args.response_prefix is not None:
        config.response_prefix = args.response_prefix
    if args.zh_refusals:
        config.eval.include_zh_markers = True
    if args.trust_remote_code:
        config.trust_remote_code = True
    if args.no_preserve_extra_weights:
        config.preserve_extra_weights = False
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.auto_select:
        config.auto_select = True
    if args.max_kl is not None:
        config.max_kl_for_auto_select = args.max_kl
    if args.seed is not None:
        config.seed = args.seed

    if args.good_prompts:
        spec = DatasetSpec(dataset=args.good_prompts)
        config.ara.good_prompts = spec
        config.abliteration.good_prompts = spec
    if args.bad_prompts:
        spec = DatasetSpec(dataset=args.bad_prompts)
        config.ara.bad_prompts = spec
        config.abliteration.bad_prompts = spec
    if args.eval_refusal_prompts:
        config.eval.refusal_prompts = DatasetSpec(dataset=args.eval_refusal_prompts)
    if args.eval_kl_prompts:
        config.eval.kl_prompts = DatasetSpec(dataset=args.eval_kl_prompts)

    if args.smoke:
        config.n_trials = 3
        config.n_startup_trials = 2
        config.max_response_length = 40
        config.ara.n_optimization_steps = 2
        config.ara.max_iter = 10
        if not args.good_prompts and isinstance(config.ara.good_prompts, DatasetSpec):
            config.ara.good_prompts.split = "train[:32]"
        if not args.good_prompts and isinstance(config.abliteration.good_prompts, DatasetSpec):
            config.abliteration.good_prompts.split = "train[:32]"
        if not args.bad_prompts and isinstance(config.ara.bad_prompts, DatasetSpec):
            config.ara.bad_prompts.split = "train[:32]"
        if not args.bad_prompts and isinstance(config.abliteration.bad_prompts, DatasetSpec):
            config.abliteration.bad_prompts.split = "train[:32]"
        if not args.eval_refusal_prompts and isinstance(config.eval.refusal_prompts, DatasetSpec):
            config.eval.refusal_prompts.split = "test[:16]"
        if not args.eval_kl_prompts and isinstance(config.eval.kl_prompts, DatasetSpec):
            config.eval.kl_prompts.split = "test[:16]"

    if args.evaluate_model:
        evaluate_existing_model(config, args.evaluate_model)
        return

    run_heretic_mlx(config)


if __name__ == "__main__":
    main()
