# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproducibility manifest (reproduce.json & SHA256SUMS.txt) for heretic-mlx."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import HereticMLXConfig

console = Console()


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment_info() -> dict[str, Any]:
    chip = None
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        chip = None

    return {
        "os": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "chip": chip,
        "python": platform.python_version(),
        "packages": {
            pkg: _pkg_version(pkg)
            for pkg in ("heretic-mlx", "mlx", "mlx-lm", "optuna", "scipy", "numpy", "transformers")
        },
    }


def save_reproduction_manifest(
    export_dir: Path,
    *,
    config: HereticMLXConfig,
    modifier_name: str,
    parameters: dict[str, Any],
    scores: list[dict[str, Any]],
    response_prefix: str,
    selected_trial: int | None = None,
) -> Path:
    """Write reproduce/reproduce.json and reproduce/SHA256SUMS.txt inside export_dir."""
    repro_dir = export_dir / "reproduce"
    repro_dir.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for file_path in sorted(export_dir.iterdir()):
        if file_path.is_file() and (
            file_path.suffix == ".safetensors" or file_path.name == "config.json"
        ):
            hashes[file_path.name] = compute_file_sha256(file_path)

    manifest: dict[str, Any] = {
        "format_version": "heretic-mlx-1",
        "model": config.model,
        "modifier": config.modifier.value,
        "modifier_name": modifier_name,
        "export_modified_mode": config.export_modified_mode.value,
        "target_components": config.target_components,
        "preserve_extra_weights": config.should_preserve_extra_weights(),
        "seed": config.seed,
        "response_prefix": response_prefix,
        "selected_trial": selected_trial,
        "parameters": parameters,
        "scores": scores,
        "config": config.model_dump(mode="json"),
        "environment": collect_environment_info(),
        "hashes": hashes,
    }

    manifest_path = repro_dir / "reproduce.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    checksums_path = repro_dir / "SHA256SUMS.txt"
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.writelines(f"{digest}  {filename}\n" for filename, digest in sorted(hashes.items()))

    console.print(f"* Saved reproduction manifest to [cyan]{manifest_path}[/]")
    return manifest_path


def load_reproduction_manifest(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        data = json.load(f)
    if "parameters" not in data or "config" not in data:
        raise ValueError(f"Invalid heretic-mlx reproduction manifest: {path}")
    return data


def verify_exported_hashes(export_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Verify exported safetensors/config hashes against manifest. Returns {filename: status}."""
    expected: dict[str, str] = manifest.get("hashes", {})
    results: dict[str, str] = {}
    for filename, expected_sha in expected.items():
        fp = export_dir / filename
        if not fp.is_file():
            results[filename] = "missing"
            console.print(f"  * [bold]{filename}[/]: [red]missing[/]")
            continue
        actual = compute_file_sha256(fp)
        if actual.lower() == expected_sha.lower():
            results[filename] = "match"
            console.print(f"  * [bold]{filename}[/]: [green]SHA256 matches[/]")
        else:
            results[filename] = "mismatch"
            console.print(f"  * [bold]{filename}[/]: [yellow]SHA256 differs (expected across different HW/SW)[/]")
    return results
