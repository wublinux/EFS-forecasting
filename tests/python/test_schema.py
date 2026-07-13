import pandas as pd
import pytest

from adaptforecast.schema import CANONICAL_COLUMNS, DataValidationError, validate_dataframe


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2016-03-01", "water", 10, 20, 25, 15, 60, 0],
            ["2017-03-01", "water", 11, 21, 26, 16, 61, 1],
        ],
        columns=CANONICAL_COLUMNS,
    )


def test_validate_canonical_frame() -> None:
    report = validate_dataframe(valid_frame())
    assert report.rows == 2
    assert report.years == [2016, 2017]
    assert report.categories == ["water"]


def test_missing_column_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(valid_frame().drop(columns=["rainfall"]))


def test_duplicate_key_is_rejected() -> None:
    duplicate = pd.concat([valid_frame(), valid_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_dataframe(duplicate)


def test_physical_ranges_are_checked() -> None:
    frame = valid_frame()
    frame.loc[0, "humidity"] = 101
    with pytest.raises(DataValidationError, match="humidity"):
        validate_dataframe(frame)
