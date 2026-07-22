"""Artifact plots generated from committed tabular evidence."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def save_forecast_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    if os.name == "nt":
        os.environ.setdefault("WINDIR", os.environ.get("SystemRoot", "C:\\Windows"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    if "backend" not in predictions:
        predictions["backend"] = "unspecified"
    for category, group in predictions.groupby("category"):
        figure, axis = plt.subplots(figsize=(10, 5))
        actual = group.groupby("date")["actual"].first().sort_index()
        axis.plot(actual.index, actual.values, color="black", linewidth=2, label="actual")
        averaged = (
            group.groupby(["date", "model", "variant", "backend"])["predicted"].mean().reset_index()
        )
        for (model, variant, backend), line in averaged.groupby(["model", "variant", "backend"]):
            line = line.sort_values("date")
            axis.plot(
                line["date"],
                line["predicted"],
                label=f"{model}:{variant}:{backend}",
            )
        axis.set_title(f"Next-day demand forecast - {category}")
        axis.set_xlabel("Target date")
        axis.set_ylabel("Sales")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output_dir / f"forecast_{_safe_name(str(category))}.png", dpi=150)
        plt.close(figure)


def save_rule_activation_plots(run_dir: Path) -> None:
    if os.name == "nt":
        os.environ.setdefault("WINDIR", os.environ.get("SystemRoot", "C:\\Windows"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for activation_path in run_dir.glob("efs/**/activations.csv"):
        activation = pd.read_csv(activation_path)
        values = activation.drop(columns=["date"], errors="ignore").to_numpy(dtype=float)
        if values.size == 0:
            continue
        figure, axis = plt.subplots(figsize=(10, 5))
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_title("Rule activation intensity (association evidence)")
        axis.set_xlabel("Rule index")
        axis.set_ylabel("Sample index")
        figure.colorbar(image, ax=axis, label="Geometric mean firing strength")
        figure.tight_layout()
        figure.savefig(activation_path.with_name("rule_activation.png"), dpi=150)
        plt.close(figure)
