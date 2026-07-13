"""Post-run verification for auditable experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def verify_smoke_artifact(root: Path) -> Path:
    """Return the newest valid smoke run or raise on a missing contract element."""
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No run manifest found below {root}")
    manifest_path = manifests[-1]
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Smoke run did not complete")
    if len(str(manifest.get("data_sha256", ""))) != 64:
        raise RuntimeError("Manifest does not contain a SHA-256 data identity")

    metrics = pd.read_csv(run_dir / "metrics.csv")
    expected_models = {"seasonal_naive", "arima", "lstm", "efs"}
    missing_models = expected_models - set(metrics["model"])
    if missing_models:
        unavailable_path = run_dir / "unavailable.csv"
        diagnostic = ""
        if unavailable_path.exists():
            diagnostic = f"; unavailable records: {unavailable_path.read_text(encoding='utf-8')}"
        raise RuntimeError(f"Missing smoke models: {sorted(missing_models)}{diagnostic}")
    efs_variants = set(metrics.loc[metrics["model"] == "efs", "variant"])
    if efs_variants != {"sales_only", "target_weather"}:
        raise RuntimeError(f"Unexpected EFS variants: {sorted(efs_variants)}")

    efs_root = run_dir / "efs"
    required = ["model.mat", "predictions.csv", "rules.csv", "activations.csv"]
    for name in required:
        if not list(efs_root.glob(f"**/{name}")):
            raise RuntimeError(f"No EFS {name} was produced")
    return run_dir
