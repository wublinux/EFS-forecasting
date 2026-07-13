"""Prepare hash-identified EFS jobs for the licensed MATLAB Actions runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptforecast.config import load_config
from adaptforecast.data import load_canonical, make_supervised, prepare_category
from adaptforecast.matlab_bridge import MatlabBatchRunner


def prepare(config_path: Path, repository_root: Path, cache_root: Path) -> int:
    root = repository_root.resolve()
    config = load_config(config_path)
    data = load_canonical(root / config.data_path)
    categories = (
        sorted(data["category"].unique()) if config.categories == "all" else list(config.categories)
    )
    runner = MatlabBatchRunner(root, config.efs.executable)
    count = 0
    for category in categories:
        prepared = prepare_category(
            data, category, config.train_year, config.test_year, config.weather_features
        )
        for variant, weather in [
            ("sales_only", []),
            ("target_weather", config.weather_features),
        ]:
            train = make_supervised(prepared.train, prepared.scales, config.lags, weather)
            test = make_supervised(prepared.test, prepared.scales, config.lags, weather)
            features = [f"sales_lag_{lag}" for lag in config.lags] + weather
            for seed in config.seeds:
                output_dir = cache_root / "efs" / category.replace(" ", "_") / variant / str(seed)
                runner.prepare_training_job(
                    train,
                    test,
                    feature_columns=features,
                    config=config.efs,
                    seed=seed,
                    output_dir=output_dir,
                )
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache", type=Path, required=True)
    arguments = parser.parse_args()
    jobs = prepare(arguments.config, arguments.root, arguments.cache.resolve())
    print(f"Prepared {jobs} MATLAB jobs")
