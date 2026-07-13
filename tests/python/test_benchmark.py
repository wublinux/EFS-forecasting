from pathlib import Path

import pandas as pd

from adaptforecast.benchmark import run_benchmark
from adaptforecast.config import BenchmarkConfig


def test_seasonal_naive_smoke_benchmark_writes_auditable_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = BenchmarkConfig(
        data_path=str(root / "data/sample/synthetic_demand.csv"),
        artifact_root=str(tmp_path),
        categories=["ice cream"],
        seeds=[42],
        models=["seasonal_naive"],
        profile="smoke",
    )
    run_dir = run_benchmark(config, root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["model"].tolist() == ["seasonal_naive"]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions.csv").exists()
