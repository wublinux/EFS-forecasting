"""Command-line interface for validation, preparation, benchmarking, and prediction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

from .benchmark import run_benchmark
from .config import load_config
from .data import wide_to_long
from .matlab_bridge import MatlabBatchRunner
from .schema import validate_dataframe

app = typer.Typer(
    name="adaptforecast",
    help="Auditable small-data supply-chain demand forecasting.",
    no_args_is_help=True,
)


@app.command("validate-data")
def validate_data(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    legacy_wide: bool = typer.Option(False, help="Validate the thesis 27-column wide format."),
) -> None:
    """Validate a dataset and print a machine-readable report."""
    frame = pd.read_csv(input_path)
    if legacy_wide:
        frame = wide_to_long(frame)
    report = validate_dataframe(frame)
    typer.echo(json.dumps(report.to_dict(), indent=2, default=str))


@app.command()
def prepare(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_path: Path = typer.Argument(...),
    legacy_wide: bool = typer.Option(True, help="Convert from the thesis wide format."),
) -> None:
    """Convert input data into the canonical long-table contract."""
    frame = pd.read_csv(input_path)
    prepared = wide_to_long(frame) if legacy_wide else frame
    validate_dataframe(prepared)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    typer.echo(f"Wrote {len(prepared)} rows to {output_path}")


@app.command()
def benchmark(
    config_path: Path = typer.Option(Path("configs/benchmark.yaml"), "--config", "-c"),
    repository_root: Path = typer.Option(Path("."), "--root"),
) -> None:
    """Run all configured baselines and EFS ablations."""
    config = load_config(config_path)
    run_dir = run_benchmark(config, repository_root)
    typer.echo(str(run_dir))


@app.command()
def predict(
    model: Path = typer.Option(..., exists=True, readable=True),
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_path: Path = typer.Option(Path("predictions.csv"), "--output"),
    features: str = typer.Option(..., help="Comma-separated normalized feature columns."),
    matlab: str = typer.Option("matlab", help="MATLAB executable or absolute path."),
    repository_root: Path = typer.Option(Path("."), "--root"),
) -> None:
    """Predict with a saved MATLAB EFS model and prepared normalized features."""
    feature_columns = [value.strip() for value in features.split(",") if value.strip()]
    if not feature_columns:
        raise typer.BadParameter("At least one feature column is required")
    runner = MatlabBatchRunner(repository_root, matlab)
    runner.predict_saved_model(model, input_path, output_path, feature_columns)
    typer.echo(str(output_path))


@app.command("app")
def app_ui(repository_root: Path = typer.Option(Path("."), "--root")) -> None:
    """Start the local Streamlit application."""
    script = repository_root.resolve() / "src" / "adaptforecast" / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(script)]
    raise typer.Exit(subprocess.run(command, check=False).returncode)


@app.command("inspect-run")
def inspect_run(run_dir: Path = typer.Argument(..., exists=True)) -> None:
    """Print the manifest for an experiment artifact directory."""
    typer.echo((run_dir / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
