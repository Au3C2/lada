#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Generate the changelog section of a Lada draft release.

Writes a markdown summary of everything that changed between the previous
release tag and HEAD: a short overview grouped by change type, followed by
one entry per commit with its subject (what changed) and the first line of
its body (what effect it has). With no previous tag the whole history is
used (first release).

The workflow appends the asset listing after this script's output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

# (heading label, (singular, plural) display names) per conventional-commit
# prefix. Matching is case-insensitive; anything else falls under "other".
CATEGORY_LABELS = {
    "feat": ("new feature", "new features"),
    "fix": ("bug fix", "bug fixes"),
    "perf": ("performance improvement", "performance improvements"),
    "refactor": ("refactoring change", "refactoring changes"),
    "docs": ("documentation change", "documentation changes"),
    "ci": ("CI change", "CI changes"),
    "packaging": ("packaging change", "packaging changes"),
    "build": ("build change", "build changes"),
    "test": ("test change", "test changes"),
    "gui": ("GUI change", "GUI changes"),
    "cli": ("CLI change", "CLI changes"),
    "translation": ("translation change", "translation changes"),
    "training": ("training change", "training changes"),
    "update": ("update", "updates"),
    "chore": ("chore", "chores"),
}
CATEGORY_ORDER = ["feat", "fix", "perf", "refactor", "docs", "ci", "packaging", "build", "test", "gui", "cli", "translation", "training", "update", "chore"]
TYPE_RE = None  # compiled lazily below


def category_of(subject: str) -> str:
    global TYPE_RE
    if TYPE_RE is None:
        import re

        TYPE_RE = re.compile(r"^([a-zA-Z]+)(\([^)]*\))?: ")
    match = TYPE_RE.match(subject)
    if match:
        key = match.group(1).lower()
        if key in CATEGORY_LABELS:
            return key
    return "other"


def get_commits(repo_dir: Path, previous_tag: str | None) -> list[tuple[str, str, str]]:
    args = ["git", "log", "--no-merges", "--format=%H%x1f%s%x1f%b%x1e"]
    args.append(f"{previous_tag}..HEAD" if previous_tag else "--root")
    output = subprocess.run(args, cwd=repo_dir, capture_output=True, text=True, check=True).stdout
    commits = []
    for record in output.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x1f")
        sha = parts[0]
        subject = parts[1] if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""
        if subject:
            commits.append((sha, subject, body))
    return commits


def render_overview(commits: list[tuple[str, str, str]], previous_tag: str | None) -> str:
    counts = Counter(category_of(subject) for _, subject, _ in commits)
    summary_parts = []
    for key in CATEGORY_ORDER:
        if counts[key]:
            singular, plural = CATEGORY_LABELS[key]
            n = counts[key]
            summary_parts.append(f"{n} {plural if n != 1 else singular}")
    if counts["other"]:
        n = counts["other"]
        summary_parts.append(f"{n} other change{'' if n == 1 else 's'}")
    if not summary_parts:
        return "No commits."
    listed = ", ".join(summary_parts[:-1]) + " and " + summary_parts[-1] if len(summary_parts) > 1 else summary_parts[0]
    if previous_tag:
        return f"Since **{previous_tag}** this release contains {listed}."
    return f"This is the first release; it contains {listed} since the beginning of the project."


def render_commits(commits: list[tuple[str, str, str]]) -> str:
    lines = []
    for sha, subject, body in commits:
        first_body_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
        lines.append(f"- `{sha[:7]}` {subject}")
        if first_body_line and first_body_line not in subject:
            lines.append(f"  > {first_body_line}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("."), help="Path to the git repository (default: current directory)")
    parser.add_argument("--current-tag", required=True, help="Tag being released, e.g. v0.11.1")
    parser.add_argument("--previous-tag", default=None, help="Previous release tag; omit or pass empty for the first release")
    args = parser.parse_args()

    previous_tag = args.previous_tag or None
    commits = get_commits(args.repo_dir, previous_tag)

    print(f"## Lada {args.current_tag}")
    print()
    print(render_overview(commits, previous_tag))
    print()
    print("### Commits (newest first)")
    print()
    print(render_commits(commits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
