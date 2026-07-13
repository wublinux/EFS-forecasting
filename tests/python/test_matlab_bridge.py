from pathlib import Path

import pandas as pd

from adaptforecast.config import EFSConfig, load_config
from adaptforecast.matlab_bridge import MatlabBatchRunner, matlab_available


def test_matlab_source_directory_is_not_an_executable(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "matlab").mkdir()
    monkeypatch.setattr("adaptforecast.matlab_bridge.shutil.which", lambda _: None)
    assert matlab_available("matlab") is False


def test_matlab_executable_can_be_supplied_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("ADAPTFORECAST_MATLAB_EXECUTABLE", "/opt/matlab/bin/matlab")
    config = load_config("configs/benchmark.smoke.yaml")
    assert config.efs.executable == "/opt/matlab/bin/matlab"


def test_hash_matching_ci_cache_reuses_file_contract_without_launching_matlab(
    monkeypatch, tmp_path: Path
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=2),
            "sales_lag_1": [0.1, 0.2],
            "target_norm": [0.2, 0.3],
        }
    )
    runner = MatlabBatchRunner(tmp_path)
    cache_job = tmp_path / "cache" / "efs" / "water" / "sales_only" / "42"
    runner.prepare_training_job(
        frame,
        frame,
        feature_columns=["sales_lag_1"],
        config=EFSConfig(use_parallel=False),
        seed=42,
        output_dir=cache_job,
    )
    pd.DataFrame({"date": frame["date"], "prediction_norm": [0.25, 0.35]}).to_csv(
        cache_job / "predictions.csv", index=False
    )
    for name in ["model.mat", "activations.csv", "rules.csv", "training_summary.json"]:
        (cache_job / name).write_text("cached", encoding="utf-8")

    monkeypatch.setenv("ADAPTFORECAST_EFS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(runner, "_run", lambda _: (_ for _ in ()).throw(AssertionError()))
    output = tmp_path / "run" / "efs" / "water" / "sales_only" / "42"
    result = runner.train_and_predict(
        frame,
        frame,
        feature_columns=["sales_lag_1"],
        config=EFSConfig(use_parallel=False),
        seed=42,
        output_dir=output,
    )

    assert result["prediction_norm"].tolist() == [0.25, 0.35]
    assert (output / "model.mat").read_text(encoding="utf-8") == "cached"
