import numpy as np
import pandas as pd

from adaptforecast.data import make_supervised, prepare_category, wide_to_long


def test_wide_to_long_maps_year_category_and_weather() -> None:
    wide = pd.DataFrame(
        {
            "Date": ["1/3", "2/3"],
            "2016 water sales": [10, 12],
            "2017 water sales": [20, 22],
            "2016 AVG temp": [15, 16],
            "2016 MAX temp": [20, 21],
            "2016 MIN temp": [10, 11],
            "2016 Humidity": [60, 61],
            "2016 Rainfall": [0, 1],
            "2017 AVG temp": [17, 18],
            "2017 MAX temp": [22, 23],
            "2017 MIN temp": [12, 13],
            "2017 Humidity": [62, 63],
            "2017 Rainfall": [2, 3],
        }
    )
    long = wide_to_long(wide)
    assert len(long) == 4
    assert set(long["date"].dt.year) == {2016, 2017}
    assert long.loc[long["date"] == pd.Timestamp("2017-03-01"), "rainfall"].item() == 2


def canonical_fixture() -> pd.DataFrame:
    rows = []
    for year, sales in [(2016, [1, 2, 3, 4, 5]), (2017, [100, 110, 120, 130, 140])]:
        for index, value in enumerate(sales):
            rows.append(
                {
                    "date": pd.Timestamp(year, 3, index + 1),
                    "category": "water",
                    "sales": value,
                    "avg_temp": 10 + index + (100 if year == 2017 else 0),
                    "max_temp": 15 + index,
                    "min_temp": 5 + index,
                    "humidity": 50 + index,
                    "rainfall": index,
                }
            )
    return pd.DataFrame(rows)


def test_scalers_fit_training_year_only() -> None:
    prepared = prepare_category(canonical_fixture(), "water", 2016, 2017, ["avg_temp"])
    assert prepared.scales["sales"].maximum == 5
    assert prepared.scales["avg_temp"].maximum == 14
    test = make_supervised(prepared.test, prepared.scales, [1, 2, 3], ["avg_temp"])
    assert test["avg_temp"].iloc[0] == 1
    assert test["target_norm"].iloc[0] > 1


def test_target_date_weather_and_past_sales_are_aligned() -> None:
    prepared = prepare_category(canonical_fixture(), "water", 2016, 2017, ["avg_temp"])
    supervised = make_supervised(prepared.train, prepared.scales, [1, 2, 3], ["avg_temp"])
    first = supervised.iloc[0]
    assert first["date"] == pd.Timestamp("2016-03-04")
    assert np.isclose(first["sales_lag_1"], 0.5)
    assert np.isclose(first["sales_lag_3"], 0.0)
    assert np.isclose(first["avg_temp"], 0.75)


def test_missing_target_is_excluded_but_history_is_imputed() -> None:
    frame = canonical_fixture()
    frame.loc[(frame["date"] == "2016-03-02"), "sales"] = np.nan
    prepared = prepare_category(frame, "water", 2016, 2017, ["avg_temp"])
    supervised = make_supervised(prepared.train, prepared.scales, [1, 2, 3], ["avg_temp"])
    assert pd.Timestamp("2016-03-02") not in set(supervised["date"])
    assert np.isfinite(supervised.filter(like="sales_lag").to_numpy()).all()


def test_test_year_forward_fill_cannot_cross_the_year_boundary() -> None:
    frame = canonical_fixture()
    frame.loc[frame["date"] == pd.Timestamp("2017-03-01"), "sales"] = np.nan
    prepared = prepare_category(frame, "water", 2016, 2017, ["avg_temp"])

    assert prepared.test.iloc[0]["sales"] == prepared.medians["sales"]
    assert prepared.test.iloc[0]["sales"] != prepared.train.iloc[-1]["sales"]
