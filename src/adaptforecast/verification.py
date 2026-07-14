"""Post-run verification for auditable experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    for activation_path in efs_root.glob("**/activations.csv"):
        activation = pd.read_csv(activation_path).drop(columns=["date"], errors="ignore")
        values = activation.to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid rule activation values: {activation_path}")
        if values.min() < -1e-12 or values.max() > 1 + 1e-12:
            raise RuntimeError(f"Rule activation is outside [0,1]: {activation_path}")
        if np.max(np.std(values, axis=0)) <= 1e-12:
            raise RuntimeError(f"Rule activation does not respond to inputs: {activation_path}")
        rules = pd.read_csv(activation_path.with_name("rules.csv"))
        if (rules["support"] < 0).any() or (rules["support"] > len(values)).any():
            raise RuntimeError(f"Rule support is outside the sample count: {activation_path}")
        summary = json.loads(
            activation_path.with_name("training_summary.json").read_text(encoding="utf-8")
        )
        if summary.get("upper_parameters_locked") is not True:
            raise RuntimeError(f"Type-2 upper parameters were not locked: {activation_path}")
    return run_dir
