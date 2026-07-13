"""Canonical data contract and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

CANONICAL_COLUMNS = [
    "date",
    "category",
    "sales",
    "avg_temp",
    "max_temp",
    "min_temp",
    "humidity",
    "rainfall",
]
NUMERIC_COLUMNS = CANONICAL_COLUMNS[2:]


class DataValidationError(ValueError):
    """Raised when demand data violates the public contract."""


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    categories: list[str]
    years: list[int]
    duplicate_rows: int
    missing_by_column: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def coerce_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical copy with parsed dates and numeric feature columns."""
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")

    data = frame.loc[:, CANONICAL_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["category"] = data["category"].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(["category", "date"], kind="stable").reset_index(drop=True)


def validate_dataframe(frame: pd.DataFrame, *, strict: bool = True) -> ValidationReport:
    """Validate canonical demand data without silently repairing it."""
    data = coerce_dataframe(frame)
    errors: list[str] = []
    warnings: list[str] = []

    invalid_dates = int(data["date"].isna().sum())
    empty_categories = int(data["category"].isna().sum() + (data["category"] == "").sum())
    duplicate_rows = int(data.duplicated(["date", "category"]).sum())

    if invalid_dates:
        errors.append(f"{invalid_dates} rows have invalid dates")
    if empty_categories:
        errors.append(f"{empty_categories} rows have empty categories")
    if duplicate_rows:
        errors.append(f"{duplicate_rows} duplicate date/category rows")
    if (data["sales"].dropna() < 0).any():
        errors.append("sales values must be non-negative")
    if (data["humidity"].dropna().between(0, 100, inclusive="both") == 0).any():
        errors.append("humidity must be between 0 and 100")
    if (data["rainfall"].dropna() < 0).any():
        errors.append("rainfall must be non-negative")

    years = sorted(data.loc[data["date"].notna(), "date"].dt.year.unique().astype(int).tolist())
    if len(years) < 2:
        warnings.append("Fewer than two calendar years are available for a year-on-year holdout")
    for category, group in data.groupby("category", dropna=False):
        if len(group) < 10:
            warnings.append(f"Category {category!s} has fewer than 10 observations")

    if strict and errors:
        raise DataValidationError("; ".join(errors))
    warnings.extend(errors)
    return ValidationReport(
        rows=len(data),
        categories=sorted(data["category"].dropna().astype(str).unique().tolist()),
        years=years,
        duplicate_rows=duplicate_rows,
        missing_by_column={column: int(data[column].isna().sum()) for column in CANONICAL_COLUMNS},
        warnings=warnings,
    )
