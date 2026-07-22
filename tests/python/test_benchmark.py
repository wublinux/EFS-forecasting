import json
import shutil
from pathlib import Path

import pandas as pd

from adaptforecast.artifacts import sha256_file
from adaptforecast.benchmark import _weather_ablation, run_benchmark
from adaptforecast.config import BenchmarkConfig, EFSConfig


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


def test_private_data_environment_path_is_used_for_manifest_hash(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    private_dir = tmp_path / "private-data"
    private_dir.mkdir()
    private_file = private_dir / "canonical-private.csv"
    shutil.copy2(root / "data/sample/synthetic_demand.csv", private_file)
    monkeypatch.setenv("ADAPTFORECAST_DATA_DIR", str(private_dir))
    config = BenchmarkConfig(
        data_path=private_file.name,
        artifact_root=str(tmp_path / "artifacts"),
        categories=["ice cream"],
        models=["seasonal_naive"],
        profile="smoke",
    )

    run_dir = run_benchmark(config, root)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["data_file"] == private_file.name
    assert manifest["data_sha256"] == sha256_file(private_file)


def test_python_it2_smoke_benchmark_produces_both_weather_variants(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = BenchmarkConfig(
        data_path=str(root / "data/sample/synthetic_demand.csv"),
        artifact_root=str(tmp_path),
        categories=["ice cream"],
        seeds=[42],
        models=["efs"],
        profile="smoke",
        efs=EFSConfig(
            backend="python-it2",
            population_size=4,
            max_generations=1,
            pattern_max_iterations=1,
            ga_stall_generations=1,
            use_parallel=False,
        ),
    )

    run_dir = run_benchmark(config, root)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert set(metrics["variant"]) == {"sales_only", "target_weather"}
    assert set(metrics["backend"]) == {"python-it2"}
    assert len(pd.read_csv(run_dir / "weather_ablation.csv")) == 1
    assert manifest["status"] == "complete"
    assert len(list((run_dir / "efs").glob("**/model.npz"))) == 2
