from pathlib import Path

import numpy as np
import pandas as pd

from adaptforecast.config import EFSConfig, load_config
from adaptforecast.python_it2 import (
    PythonIT2Model,
    PythonIT2Runner,
    _type_reduce,
    load_python_it2_model,
    train_python_it2,
)


def _training_frame() -> pd.DataFrame:
    first = np.linspace(0.0, 1.0, 24)
    second = np.clip(np.sin(np.linspace(0.0, np.pi, 24)), 0.0, 1.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=24),
            "sales_lag_1": first,
            "humidity": second,
            "target_norm": np.clip(0.65 * first + 0.25 * second, 0.0, 1.0),
        }
    )


def _smoke_config() -> EFSConfig:
    return EFSConfig(
        backend="python-it2",
        population_size=6,
        max_generations=2,
        pattern_max_iterations=2,
        ga_stall_generations=2,
        use_parallel=False,
    )


def test_python_it2_example_configuration_is_explicit() -> None:
    config = load_config("configs/benchmark.python-it2.smoke.yaml")

    assert config.efs.backend == "python-it2"
    assert config.efs.use_parallel is False


def test_type_reduction_matches_type1_weighted_average() -> None:
    firing = np.asarray([[0.2, 0.8], [0.5, 0.5]])
    prediction, no_rule = _type_reduce(firing, firing, np.asarray([0.0, 1.0]), 0.4)

    assert np.allclose(prediction, [0.8, 0.5])
    assert not no_rule.any()


def test_type_reduction_uses_fallback_when_no_rule_fires() -> None:
    prediction, no_rule = _type_reduce(np.zeros((2, 1)), np.zeros((2, 1)), np.asarray([0.7]), 0.4)

    assert np.allclose(prediction, [0.4, 0.4])
    assert no_rule.all()


def test_python_it2_training_is_deterministic_and_locks_upper_parameters() -> None:
    frame = _training_frame()
    features = ["sales_lag_1", "humidity"]

    first = train_python_it2(frame, features, _smoke_config(), seed=42)
    second = train_python_it2(frame, features, _smoke_config(), seed=42)

    values = frame[features].to_numpy()
    assert np.allclose(first.model.predict(values), second.model.predict(values))
    assert len(first.model.antecedents) <= 2 ** len(features)
    assert first.summary["upper_parameters_locked"] is True
    assert first.summary["backend"] == "python-it2"
    assert np.isfinite(first.model.predict(values)).all()


def test_python_it2_runner_writes_complete_artifact_and_reloadable_model(
    tmp_path: Path,
) -> None:
    train = _training_frame()
    test = train.assign(date=pd.date_range("2017-01-01", periods=len(train)))
    features = ["sales_lag_1", "humidity"]
    output_dir = tmp_path / "efs" / "water" / "target_weather" / "42"

    result = PythonIT2Runner(tmp_path).train_and_predict(
        train,
        test,
        feature_columns=features,
        config=_smoke_config(),
        seed=42,
        output_dir=output_dir,
    )

    required = {
        "job.json",
        "model.json",
        "model.npz",
        "predictions.csv",
        "activations.csv",
        "rules.csv",
        "training_summary.json",
    }
    assert required <= {path.name for path in output_dir.iterdir() if path.is_file()}
    assert len(result) == len(test)
    model = load_python_it2_model(output_dir / "model.npz")
    assert isinstance(model, PythonIT2Model)
    assert np.allclose(
        model.predict(test[features].to_numpy()), result["prediction_norm"].to_numpy()
    )
    rules = pd.read_csv(output_dir / "rules.csv")
    assert {
        "antecedent",
        "consequent",
        "weight",
        "support",
        "mean_activation",
        "max_activation",
    } <= set(rules.columns)
