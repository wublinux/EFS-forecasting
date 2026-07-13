import pytest

from adaptforecast.baselines import arima_forecast, lstm_forecast, seasonal_naive
from adaptforecast.data import load_canonical, make_supervised, prepare_category


@pytest.fixture
def prepared_sample():
    data = load_canonical("data/sample/synthetic_demand.csv")
    return prepare_category(data, "ice cream", 2016, 2017, ["avg_temp", "humidity", "rainfall"])


def test_seasonal_naive_uses_weekly_lag(prepared_sample) -> None:
    forecast = seasonal_naive(prepared_sample.test, 7)
    assert len(forecast.values) == 68
    assert forecast.values[0] == prepared_sample.test.iloc[0]["sales"]


def test_arima_smoke_runs_when_statsmodels_is_installed(prepared_sample) -> None:
    pytest.importorskip("statsmodels")
    forecast = arima_forecast(prepared_sample.train, prepared_sample.test, "smoke")
    assert len(forecast.values) == 75
    assert "order" in forecast.details


def test_lstm_smoke_is_deterministic(prepared_sample) -> None:
    pytest.importorskip("torch")
    train = make_supervised(prepared_sample.train, prepared_sample.scales, [1, 2, 3], [])
    test = make_supervised(prepared_sample.test, prepared_sample.scales, [1, 2, 3], [])
    features = ["sales_lag_3", "sales_lag_2", "sales_lag_1"]
    first = lstm_forecast(train, test, features, seed=42, profile="smoke")
    second = lstm_forecast(train, test, features, seed=42, profile="smoke")
    assert (first.values == second.values).all()
