"""Generate a deterministic, fully synthetic public demonstration dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "ice cream": (4200, 115, -12, 4),
    "water": (72000, 950, -180, 75),
    "insect repellent": (3400, 35, 22, -3),
    "jelly": (10500, 125, -8, 6),
    "yogurt": (360000, 1900, -250, 90),
    "beer": (41000, 410, -50, 22),
    "dehumidifier": (4300, 18, 75, 8),
    "plant protein": (24000, 140, -20, 14),
}


def build(seed: int = 20260713) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for year in (2016, 2017):
        dates = pd.date_range(f"{year}-03-01", periods=75, freq="D")
        day = np.arange(len(dates))
        avg_temp = 10 + 0.18 * day + 4 * np.sin(day / 7) + rng.normal(0, 1.4, len(day))
        max_temp = avg_temp + 5 + rng.normal(0, 0.8, len(day))
        min_temp = avg_temp - 4 + rng.normal(0, 0.8, len(day))
        humidity = np.clip(68 + 12 * np.sin(day / 5 + 1) + rng.normal(0, 5, len(day)), 30, 98)
        rain_event = rng.random(len(day)) < 0.24
        rainfall = rain_event * rng.gamma(1.7, 5.0, len(day))
        for category_index, (category, effects) in enumerate(CATEGORIES.items()):
            base, temp_effect, humidity_effect, rain_effect = effects
            weekly = base * 0.12 * np.sin(2 * np.pi * day / 7 + category_index / 4)
            year_shift = base * 0.08 * (year - 2016)
            noise = rng.normal(0, base * 0.08, len(day))
            sales = (
                base
                + year_shift
                + weekly
                + temp_effect * (avg_temp - avg_temp.mean())
                + humidity_effect * (humidity - humidity.mean())
                + rain_effect * rainfall
                + noise
            )
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "category": category,
                        "sales": round(max(float(sales[index]), 0), 2),
                        "avg_temp": round(float(avg_temp[index]), 2),
                        "max_temp": round(float(max_temp[index]), 2),
                        "min_temp": round(float(min_temp[index]), 2),
                        "humidity": round(float(humidity[index]), 2),
                        "rainfall": round(float(rainfall[index]), 2),
                    }
                )
    return pd.DataFrame(rows).sort_values(["category", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/sample/synthetic_demand.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(args.output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
