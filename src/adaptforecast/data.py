"""Leakage-safe data conversion and supervised feature creation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, coerce_dataframe, validate_dataframe

WEATHER_NAMES = {
    "AVG temp": "avg_temp",
    "MAX temp": "max_temp",
    "MIN temp": "min_temp",
    "Humidity": "humidity",
    "Rainfall": "rainfall",
}


@dataclass(frozen=True)
class Scale:
    minimum: float
    maximum: float

    @property
    def span(self) -> float:
        return max(self.maximum - self.minimum, 1e-12)

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.minimum) / self.span

    def inverse(self, values: pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.span + self.minimum


@dataclass
class PreparedCategory:
    category: str
    train: pd.DataFrame
    test: pd.DataFrame
    scales: dict[str, Scale]
    medians: dict[str, float]


def resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    import os

    private_root = os.getenv("ADAPTFORECAST_DATA_DIR")
    if private_root:
        private_candidate = Path(private_root) / candidate.name
        if private_candidate.exists():
            return private_candidate
    raise FileNotFoundError(
        f"Data file not found: {candidate}. Set ADAPTFORECAST_DATA_DIR for private data."
    )


def load_canonical(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(resolve_data_path(path))
    validate_dataframe(data)
    return coerce_dataframe(data)


def wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the thesis 27-column wide table to the canonical long contract."""
    if "Date" not in frame.columns:
        raise ValueError("Legacy wide data must contain a Date column")

    sales_pattern = re.compile(r"^(\d{4})\s+(.+?)\s+sales$")
    sales_columns: list[tuple[str, int, str]] = []
    for column in frame.columns:
        match = sales_pattern.match(str(column))
        if match:
            sales_columns.append((str(column), int(match.group(1)), match.group(2).strip()))
    if not sales_columns:
        raise ValueError("No '<year> <category> sales' columns were found")

    records: list[pd.DataFrame] = []
    for sales_column, year, category in sales_columns:
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    frame["Date"].astype(str) + f"/{year}", dayfirst=True, errors="coerce"
                ),
                "category": category.replace("insecticide control", "insect repellent"),
                "sales": frame[sales_column],
            }
        )
        for legacy_name, canonical_name in WEATHER_NAMES.items():
            source = f"{year} {legacy_name}"
            part[canonical_name] = frame[source] if source in frame.columns else np.nan
        records.append(part)
    result = pd.concat(records, ignore_index=True)
    return coerce_dataframe(result.loc[:, CANONICAL_COLUMNS])


def _fill_features(
    frame: pd.DataFrame, medians: dict[str, float], columns: list[str]
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].ffill().fillna(medians[column])
    return result


def prepare_category(
    data: pd.DataFrame,
    category: str,
    train_year: int,
    test_year: int,
    weather_features: list[str],
) -> PreparedCategory:
    subset = data.loc[data["category"] == category].sort_values("date").copy()
    train = subset.loc[subset["date"].dt.year == train_year].copy()
    test = subset.loc[subset["date"].dt.year == test_year].copy()
    if train.empty or test.empty:
        raise ValueError(f"{category}: both {train_year} and {test_year} data are required")

    feature_columns = ["sales", *weather_features]
    medians: dict[str, float] = {}
    for column in feature_columns:
        median = float(train[column].median(skipna=True))
        if not np.isfinite(median):
            raise ValueError(f"{category}: training column {column} has no usable values")
        medians[column] = median

    train["target_observed"] = train["sales"].notna()
    test["target_observed"] = test["sales"].notna()
    train = _fill_features(train, medians, feature_columns)
    test = _fill_features(test, medians, feature_columns)

    scales = {
        column: Scale(float(train[column].min()), float(train[column].max()))
        for column in feature_columns
    }
    return PreparedCategory(category, train, test, scales, medians)


def make_supervised(
    frame: pd.DataFrame,
    scales: dict[str, Scale],
    lags: list[int],
    weather_features: list[str],
) -> pd.DataFrame:
    """Use historical sales and target-date weather to predict target-date sales."""
    data = frame.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    max_lag = max(lags)
    for index in range(max_lag, len(data)):
        if not bool(data.loc[index, "target_observed"]):
            continue
        row: dict[str, object] = {
            "date": data.loc[index, "date"],
            "target": float(data.loc[index, "sales"]),
            "target_norm": float(scales["sales"].transform([data.loc[index, "sales"]])[0]),
        }
        for lag in lags:
            lag_value = data.loc[index - lag, "sales"]
            scaled = scales["sales"].transform([lag_value])[0]
            row[f"sales_lag_{lag}"] = float(np.clip(scaled, 0, 1))
        for feature in weather_features:
            scaled = scales[feature].transform([data.loc[index, feature]])[0]
            row[feature] = float(np.clip(scaled, 0, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def common_evaluation_dates(test: pd.DataFrame, minimum_history: int = 7) -> pd.DatetimeIndex:
    ordered = test.sort_values("date").reset_index(drop=True)
    valid = ordered.loc[minimum_history:, "date"]
    return pd.DatetimeIndex(valid)
