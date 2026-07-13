"""Reproducible experiment artifact management."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_run_directory(root: str | Path, label: str = "benchmark") -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = Path(root) / f"{timestamp}-{label}"
    suffix = 1
    while target.exists():
        target = Path(root) / f"{timestamp}-{label}-{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    return target


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_run_manifest(
    run_dir: Path, *, config: dict[str, object], data_path: Path, status: str = "running"
) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            "data_file": data_path.name,
            "data_sha256": sha256_file(data_path),
            "config": config,
        },
    )


def write_table(path: str | Path, table: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
