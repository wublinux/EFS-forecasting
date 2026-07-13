from pathlib import Path

import pandas as pd

from adaptforecast.benchmark import _weather_ablation, run_benchmark
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
    assert (run_dir / "category_summary.csv").exists()
    assert (run_dir / "weather_ablation.csv").exists()


def test_weather_ablation_only_compares_matching_metric_scales() -> None:
    rows = []
    for variant, factor in [("sales_only", 1.0), ("target_weather", 0.8)]:
        rows.append(
            {
                "category": "water",
                "model": "efs",
                "variant": variant,
                "seed": 42,
                "rmse": 10 * factor,
                "mae": 8 * factor,
                "mase": 1 * factor,
                "smape": 20 * factor,
                "rmse_norm": 0.5 * factor,
                "mae_norm": 0.4 * factor,
            }
        )

    ablation = _weather_ablation(pd.DataFrame(rows))

    assert ablation.loc[0, "rmse_improvement_pct"] == 20
    assert ablation.loc[0, "rmse_sales_only"] == 10
    assert ablation.loc[0, "rmse_target_weather"] == 8
