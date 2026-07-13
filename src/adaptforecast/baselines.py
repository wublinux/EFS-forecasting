"""Transparent statistical and neural baseline implementations."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


class OptionalDependencyError(RuntimeError):
    """Raised when an optional benchmark dependency is unavailable."""


@dataclass
class Forecast:
    dates: pd.DatetimeIndex
    values: np.ndarray
    details: dict[str, object]


def seasonal_naive(test: pd.DataFrame, period: int = 7) -> Forecast:
    ordered = test.sort_values("date").reset_index(drop=True)
    dates: list[pd.Timestamp] = []
    predictions: list[float] = []
    for index in range(period, len(ordered)):
        if not bool(ordered.loc[index, "target_observed"]):
            continue
        dates.append(pd.Timestamp(ordered.loc[index, "date"]))
        predictions.append(float(ordered.loc[index - period, "sales"]))
    return Forecast(pd.DatetimeIndex(dates), np.asarray(predictions), {"period": period})


def arima_forecast(train: pd.DataFrame, test: pd.DataFrame, profile: str = "full") -> Forecast:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise OptionalDependencyError("Install adaptforecast[benchmark] to run ARIMA") from exc

    train_values = train.sort_values("date")["sales"].to_numpy(dtype=float)
    ordered_test = test.sort_values("date").reset_index(drop=True)
    upper = 1 if profile == "smoke" else 3
    best_aic = float("inf")
    best_order: tuple[int, int, int] | None = None
    best_fit = None
    for p in range(upper + 1):
        for d in range(2):
            for q in range(upper + 1):
                if p == d == q == 0:
                    continue
                try:
                    fit = ARIMA(
                        train_values,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic, best_order, best_fit = float(fit.aic), (p, d, q), fit
    if best_fit is None or best_order is None:
        raise RuntimeError("ARIMA search did not produce a valid fitted model")
    predicted = np.asarray(best_fit.forecast(steps=len(ordered_test)), dtype=float)
    mask = ordered_test["target_observed"].to_numpy(dtype=bool)
    return Forecast(
        pd.DatetimeIndex(ordered_test.loc[mask, "date"]),
        predicted[mask],
        {"order": best_order, "aic": best_aic},
    )


def lstm_forecast(
    train_supervised: pd.DataFrame,
    test_supervised: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    profile: str = "full",
) -> Forecast:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise OptionalDependencyError("Install adaptforecast[benchmark] to run LSTM") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    x = torch.tensor(train_supervised[feature_columns].to_numpy(), dtype=torch.float32)
    y = torch.tensor(train_supervised["target_norm"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    split = max(1, int(len(x) * 0.8))
    if split >= len(x):
        split = len(x) - 1
    train_x, valid_x = x[:split], x[split:]
    train_y, valid_y = y[:split], y[split:]

    class SmallLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, values):  # noqa: ANN001
            output, _ = self.lstm(values.unsqueeze(-1))
            return self.head(output[:, -1, :])

    model = SmallLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    max_epochs, patience = (5, 2) if profile == "smoke" else (300, 30)
    best_loss = float("inf")
    best_state = None
    stale = 0
    epochs = 0
    for epoch in range(max_epochs):
        epochs = epoch + 1
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(valid_x), valid_y))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_x = torch.tensor(test_supervised[feature_columns].to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        predictions = model(test_x).squeeze(1).numpy()
    return Forecast(
        pd.DatetimeIndex(test_supervised["date"]),
        predictions,
        {"hidden_units": 32, "epochs": epochs, "validation_loss": best_loss},
    )
