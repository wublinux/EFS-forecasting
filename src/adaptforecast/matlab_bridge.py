"""File-based MATLAB batch bridge used by the CLI and Streamlit app."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .artifacts import write_json, write_table
from .config import EFSConfig


class MatlabUnavailableError(RuntimeError):
    """Raised when a licensed MATLAB executable cannot be found."""


def matlab_available(executable: str = "matlab") -> bool:
    return Path(executable).is_file() or shutil.which(executable) is not None


def _matlab_literal(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


class MatlabBatchRunner:
    def __init__(self, repository_root: str | Path, executable: str = "matlab") -> None:
        self.repository_root = Path(repository_root).resolve()
        self.matlab_root = self.repository_root / "matlab"
        self.executable = executable

    def _run(self, expression: str) -> None:
        if not matlab_available(self.executable):
            raise MatlabUnavailableError(
                f"MATLAB executable {self.executable!r} was not found. "
                "Install MATLAB R2024b+ or browse precomputed results."
            )
        command = [
            self.executable,
            "-batch",
            f"addpath('{_matlab_literal(self.matlab_root)}'); {expression}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"MATLAB batch job failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

    def train_and_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        feature_columns: list[str],
        config: EFSConfig,
        seed: int,
        output_dir: Path,
    ) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs = output_dir / "inputs"
        inputs.mkdir(exist_ok=True)
        train_path, test_path = inputs / "train.csv", inputs / "test.csv"
        write_table(train_path, train[["date", *feature_columns, "target_norm"]])
        write_table(test_path, test[["date", *feature_columns, "target_norm"]])
        job = {
            "schema_version": 1,
            "train_csv": str(train_path.resolve()),
            "test_csv": str(test_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "feature_columns": feature_columns,
            "target_column": "target_norm",
            "seed": seed,
            "efs": asdict(config),
        }
        job_path = output_dir / "job.json"
        write_json(job_path, job)
        self._run(f"adaptforecast.runJob('{_matlab_literal(job_path)}')")
        result_path = output_dir / "predictions.csv"
        if not result_path.exists():
            raise RuntimeError("MATLAB completed without producing predictions.csv")
        return pd.read_csv(result_path, parse_dates=["date"])

    def predict_saved_model(
        self, model: Path, input_csv: Path, output_csv: Path, feature_columns: list[str]
    ) -> None:
        job_path = output_csv.with_suffix(".job.json")
        write_json(
            job_path,
            {
                "schema_version": 1,
                "model_file": str(model.resolve()),
                "input_csv": str(input_csv.resolve()),
                "output_csv": str(output_csv.resolve()),
                "feature_columns": feature_columns,
            },
        )
        self._run(f"adaptforecast.predictJob('{_matlab_literal(job_path)}')")
