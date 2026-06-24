from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from odile.recipes import REPO_ROOT, SERVE_RECIPES, TRAIN_RECIPES


DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_title(path: Path) -> str:
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
    except UnicodeDecodeError:
        return path.name
    return path.stem


def _extract_date(path: Path) -> str | None:
    match = DATE_PATTERN.search(path.name)
    return match.group(1) if match else None


def _scan_markdown_tree(root: Path, relative_to: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(relative_to)
        items.append(
            {
                "path": str(rel),
                "title": _extract_title(path),
                "date": _extract_date(path),
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            }
        )
    return items


def _serialize_recipe_map(recipes: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for recipe in recipes.values():
        payload = asdict(recipe) if is_dataclass(recipe) else dict(recipe)
        items.append(payload)
    return items


def build_catalogs() -> list[Path]:
    docs_root = REPO_ROOT / "docs"
    agentcontext_root = REPO_ROOT / "agentcontext"
    output_root = docs_root / "catalog"
    output_root.mkdir(parents=True, exist_ok=True)

    docs_manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "docs": _scan_markdown_tree(docs_root, REPO_ROOT),
        "agentcontext": _scan_markdown_tree(agentcontext_root, REPO_ROOT),
    }
    recipes_manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "train": _serialize_recipe_map(TRAIN_RECIPES),
        "serve": _serialize_recipe_map(SERVE_RECIPES),
    }

    docs_path = output_root / "docs_manifest.json"
    recipes_path = output_root / "recipes_manifest.json"
    docs_path.write_text(json.dumps(docs_manifest, indent=2))
    recipes_path.write_text(json.dumps(recipes_manifest, indent=2))
    return [docs_path, recipes_path]
