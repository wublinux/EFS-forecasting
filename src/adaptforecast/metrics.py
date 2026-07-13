"""Forecast metrics on a single, explicit scale."""

from __future__ import annotations

import numpy as np


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    training_series: np.ndarray,
    seasonal_period: int = 7,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) == 0:
        raise ValueError("No finite prediction pairs are available")

    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape_terms = np.divide(
        np.abs(errors), denominator, out=np.zeros_like(errors), where=denominator > 0
    )
    smape = float(100 * np.mean(smape_terms))

    train = np.asarray(training_series, dtype=float)
    if len(train) <= seasonal_period:
        mase_scale = np.nan
    else:
        mase_scale = float(np.mean(np.abs(train[seasonal_period:] - train[:-seasonal_period])))
    mase = float(mae / mase_scale) if mase_scale and np.isfinite(mase_scale) else float("nan")
    return {"rmse": rmse, "mae": mae, "mase": mase, "smape": smape, "n": int(len(actual))}


def normalized_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    errors = predicted[mask] - actual[mask]
    return {
        "rmse_norm": float(np.sqrt(np.mean(errors**2))),
        "mae_norm": float(np.mean(np.abs(errors))),
    }
