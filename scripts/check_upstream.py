#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detect new releases and commits in upstream p-e-w/heretic and map porting targets for heretic-mlx."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_META_PATH = REPO_ROOT / "UPSTREAM.json"
DEFAULT_LOCAL_UPSTREAM = REPO_ROOT.parent / "heretic"

# Map upstream p-e-w/heretic paths to corresponding heretic-mlx modules
PORTING_MAP: list[tuple[str, str]] = [
    ("src/heretic/modifiers/ara.py", "src/heretic_mlx/ara.py"),
    ("src/heretic/modifiers/abliteration.py", "src/heretic_mlx/abliteration.py"),
    ("src/heretic/model.py", "src/heretic_mlx/model.py + src/heretic_mlx/quant.py"),
    ("src/heretic/scorers/", "src/heretic_mlx/scorers.py"),
    ("src/heretic/plugin.py", "src/heretic_mlx/plugins.py"),
    ("src/heretic/evaluator.py", "src/heretic_mlx/scorers.py (Evaluator)"),
    ("src/heretic/config.py", "src/heretic_mlx/config.py"),
    ("src/heretic/reproduce.py", "src/heretic_mlx/reproduce.py"),
    ("src/heretic/main.py", "src/heretic_mlx/cli.py + src/heretic_mlx/upload.py"),
    ("config.", "configs/"),
]


def _gh_api(endpoint: str) -> Any:
    url = f"https://api.github.com/{endpoint.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "heretic-mlx-upstream-watcher",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def map_upstream_file_to_mlx(path: str) -> str | None:
    for prefix, target in PORTING_MAP:
        if path.startswith(prefix):
            return target
    return None


def pull_local_upstream(local_upstream: Path) -> None:
    if not (local_upstream / ".git").is_dir():
        raise FileNotFoundError(f"Local upstream git repo not found at {local_upstream}")
    subprocess.run(
        ["git", "-C", str(local_upstream), "fetch", "--tags", "origin"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(local_upstream), "pull", "--ff-only", "origin", "master"],
        check=True,
    )


def check_upstream(meta: dict[str, Any]) -> dict[str, Any]:
    repo = meta["upstream_repo"]
    base_commit = meta["synced_commit"]
    base_tag = meta["synced_tag"]

    # 1. Check latest release / tag
    releases = _gh_api(f"repos/{repo}/releases?per_page=5")
    latest_release = releases[0]["tag_name"] if releases else None
    new_releases = [
        {
            "tag": r["tag_name"],
            "name": r.get("name") or r["tag_name"],
            "published_at": r.get("published_at"),
            "url": r.get("html_url"),
        }
        for r in releases
        if r["tag_name"] != base_tag
    ]
    if latest_release == base_tag:
        new_releases = []
    else:
        # Keep only releases newer than base_tag
        trimmed = []
        for r in releases:
            if r["tag_name"] == base_tag:
                break
            trimmed.append(
                {
                    "tag": r["tag_name"],
                    "name": r.get("name") or r["tag_name"],
                    "published_at": r.get("published_at"),
                    "url": r.get("html_url"),
                }
            )
        new_releases = trimmed

    # 2. Compare base_commit...master
    cmp = _gh_api(f"repos/{repo}/compare/{base_commit}...master")
    ahead_by = int(cmp.get("ahead_by", 0))
    commits = [
        {
            "sha": c["sha"][:7],
            "full_sha": c["sha"],
            "date": c["commit"]["committer"]["date"],
            "message": c["commit"]["message"].splitlines()[0],
            "url": c["html_url"],
        }
        for c in cmp.get("commits", [])
    ]
    changed_files = [
        {
            "filename": f["filename"],
            "status": f["status"],
            "additions": f["additions"],
            "deletions": f["deletions"],
            "mlx_target": map_upstream_file_to_mlx(f["filename"]),
        }
        for f in cmp.get("files", [])
    ]
    actionable_files = [f for f in changed_files if f["mlx_target"] is not None]

    head_sha = commits[-1]["full_sha"] if commits else base_commit
    return {
        "upstream_repo": repo,
        "synced_tag": base_tag,
        "synced_commit": base_commit[:7],
        "latest_release": latest_release,
        "latest_commit": head_sha,
        "latest_commit_short": head_sha[:7],
        "has_updates": ahead_by > 0 or len(new_releases) > 0,
        "ahead_by": ahead_by,
        "new_releases": new_releases,
        "commits": commits,
        "changed_files": changed_files,
        "actionable_files": actionable_files,
    }


def render_markdown(report: dict[str, Any]) -> str:
    if not report["has_updates"]:
        return (
            f"### Upstream Sync Status: Up to date\n\n"
            f"`{report['upstream_repo']}` has no new commits or releases since "
            f"`{report['synced_tag']}` (`{report['synced_commit']}`)."
        )

    lines = [
        f"## Upstream Updates Detected in `{report['upstream_repo']}`",
        "",
        f"- **Synced baseline**: `{report['synced_tag']}` (`{report['synced_commit']}`)",
        f"- **Latest release**: `{report['latest_release']}`",
        f"- **Latest `master` commit**: `{report['latest_commit_short']}` (`+{report['ahead_by']}` commits)",
        "",
    ]

    if report["new_releases"]:
        lines.append("### New Releases")
        for r in report["new_releases"]:
            lines.append(f"- [{r['tag']}]({r['url']}) ({r['published_at']})")
        lines.append("")

    if report["commits"]:
        lines.append("### New Upstream Commits")
        for c in report["commits"]:
            lines.append(f"- [`{c['sha']}`]({c['url']}) {c['message']}")
        lines.append("")

    if report["actionable_files"]:
        lines.append("### Porting Impact Map (`heretic` -> `heretic-mlx`)")
        lines.append("")
        lines.append("| Upstream File | Change | Target in `heretic-mlx` |")
        lines.append("| :--- | :---: | :--- |")
        for f in report["actionable_files"]:
            lines.append(
                f"| `{f['filename']}` | `+{f['additions']}/-{f['deletions']}` | `{f['mlx_target']}` |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check upstream p-e-w/heretic for new releases and commits to port."
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON report")
    parser.add_argument("--markdown", action="store_true", help="Output Markdown report")
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Fast-forward pull local parallel ../heretic clone before reporting",
    )
    parser.add_argument(
        "--local-upstream",
        type=Path,
        default=DEFAULT_LOCAL_UPSTREAM,
        help="Path to local parallel p-e-w/heretic clone",
    )
    parser.add_argument(
        "--mark-synced",
        action="store_true",
        help="Update UPSTREAM.json to the latest upstream release & master commit",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with code 2 when upstream updates are found (0 when up to date)",
    )
    args = parser.parse_args()

    meta = json.loads(UPSTREAM_META_PATH.read_text(encoding="utf-8"))
    if args.pull:
        pull_local_upstream(args.local_upstream)

    report = check_upstream(meta)

    if args.mark_synced:
        meta["synced_commit"] = report["latest_commit"]
        meta["synced_commit_short"] = report["latest_commit_short"]
        if report["latest_release"]:
            meta["synced_tag"] = report["latest_release"]
        meta["synced_date"] = date.today().isoformat()
        UPSTREAM_META_PATH.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"Updated {UPSTREAM_META_PATH} -> {meta['synced_tag']} ({meta['synced_commit_short']})"
        )
        return

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(report))

    if args.exit_code and report["has_updates"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
