import numpy as np

from adaptforecast.metrics import evaluate_metrics, normalized_error_metrics


def test_metrics_have_expected_values() -> None:
    metrics = evaluate_metrics(
        np.array([10.0, 20.0]),
        np.array([12.0, 18.0]),
        training_series=np.arange(1.0, 20.0),
        seasonal_period=1,
    )
    assert metrics["rmse"] == 2.0
    assert metrics["mae"] == 2.0
    assert metrics["mase"] == 2.0
    assert metrics["n"] == 2


def test_normalized_metrics_are_separate() -> None:
    metrics = normalized_error_metrics(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    assert np.isclose(metrics["rmse_norm"], 0.2)
    assert np.isclose(metrics["mae_norm"], 0.2)
