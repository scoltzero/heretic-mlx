# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hugging Face Hub upload & Model Card generator for heretic-mlx."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard, ModelCardData
from rich.console import Console

from .config import HereticMLXConfig

console = Console()


def build_model_card(
    *,
    config: HereticMLXConfig,
    modifier_name: str,
    scores: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> ModelCard:
    base_model = config.model
    tags = list(dict.fromkeys([*config.upload_tags, "heretic", "mlx"]))
    card_data = ModelCardData(
        base_model=base_model if "/" in base_model and not Path(base_model).exists() else None,
        tags=tags,
    )

    lines: list[str] = [
        f"# {Path(base_model).name} ({config.modified_model_type}, MLX)",
        "",
        (
            f"This model is a **{config.modified_model_type}** MLX variant of `{base_model}`, "
            f"created with **heretic-mlx** using **{modifier_name}** "
            f"(export mode: `{config.export_modified_mode.value}`)."
        ),
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Modified |",
        "| :--- | :---: | :---: |",
    ]
    for rec in scores:
        b_disp = rec["baseline"]["md_display"]
        s_disp = rec["score"]["md_display"]
        lines.append(f"| {rec['name']} | {b_disp} | {s_disp} |")

    lines.extend(
        [
            "",
            "## Parameters",
            "",
            "```json",
        ]
    )
    import json

    lines.append(json.dumps(parameters, indent=2, ensure_ascii=False))
    lines.append("```")
    return ModelCard.from_template(card_data, template_str="{{ card_data }}\n" + "\n".join(lines))


def upload_export_to_hf(
    export_dir: Path,
    *,
    repo_id: str,
    token: str,
    private: bool,
    config: HereticMLXConfig,
    modifier_name: str,
    scores: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> str:
    """Write README.md model card into export_dir and upload folder to Hugging Face Hub."""
    card = build_model_card(
        config=config,
        modifier_name=modifier_name,
        scores=scores,
        parameters=parameters,
    )
    card.save(export_dir / "README.md")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    console.print(f"Uploading [bold]{export_dir}[/] to [cyan]https://huggingface.co/{repo_id}[/]...")
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}"
    console.print(f"* Upload complete: [green]{url}[/]")
    return url
