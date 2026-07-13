"""End-to-end audited benchmark orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import create_run_directory, write_json, write_run_manifest, write_table
from .baselines import OptionalDependencyError, arima_forecast, lstm_forecast, seasonal_naive
from .config import BenchmarkConfig
from .data import common_evaluation_dates, load_canonical, make_supervised, prepare_category
from .matlab_bridge import MatlabBatchRunner, MatlabUnavailableError
from .metrics import evaluate_metrics, normalized_error_metrics
from .plots import save_forecast_plots, save_rule_activation_plots


def _align(
    dates: pd.DatetimeIndex,
    predicted: np.ndarray,
    truth: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    forecast = pd.DataFrame({"date": pd.to_datetime(dates), "predicted": predicted})
    merged = truth[["date", "target", "target_norm"]].merge(forecast, on="date", how="inner")
    return merged.loc[merged["date"].isin(common_dates)].sort_values("date")


def _metric_row(
    aligned: pd.DataFrame,
    *,
    category: str,
    model: str,
    variant: str,
    seed: int | None,
    train_sales: np.ndarray,
    predicted_norm: np.ndarray,
    seasonal_period: int,
) -> dict[str, object]:
    metrics = evaluate_metrics(
        aligned["target"].to_numpy(),
        aligned["predicted"].to_numpy(),
        training_series=train_sales,
        seasonal_period=seasonal_period,
    )
    metrics.update(normalized_error_metrics(aligned["target_norm"], predicted_norm))
    return {
        "category": category,
        "model": model,
        "variant": variant,
        "seed": seed,
        **metrics,
    }


METRIC_COLUMNS = ["rmse", "mae", "mase", "smape", "rmse_norm", "mae_norm"]


def _weather_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    efs = metrics.loc[metrics["model"] == "efs"]
    sales_only = efs.loc[efs["variant"] == "sales_only"]
    weather = efs.loc[efs["variant"] == "target_weather"]
    paired = sales_only.merge(
        weather, on=["category", "model", "seed"], suffixes=("_sales_only", "_target_weather")
    )
    result = paired[["category", "seed"]].copy()
    for metric in METRIC_COLUMNS:
        sales_column = f"{metric}_sales_only"
        weather_column = f"{metric}_target_weather"
        result[sales_column] = paired[sales_column]
        result[weather_column] = paired[weather_column]
        result[f"{metric}_improvement_pct"] = np.where(
            paired[sales_column] != 0,
            100 * (paired[sales_column] - paired[weather_column]) / paired[sales_column],
            np.nan,
        )
    return result


def run_benchmark(config: BenchmarkConfig, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    data_path = (root / config.data_path).resolve()
    data = load_canonical(data_path)
    categories = (
        sorted(data["category"].unique().tolist())
        if config.categories == "all"
        else list(config.categories)
    )
    run_dir = create_run_directory(root / config.artifact_root)
    write_run_manifest(run_dir, config=config.to_dict(), data_path=data_path)
    runner = MatlabBatchRunner(root, config.efs.executable)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    unavailable: list[dict[str, str]] = []

    for category in categories:
        prepared = prepare_category(
            data, category, config.train_year, config.test_year, config.weather_features
        )
        train_plain = make_supervised(prepared.train, prepared.scales, config.lags, [])
        test_plain = make_supervised(prepared.test, prepared.scales, config.lags, [])
        train_weather = make_supervised(
            prepared.train, prepared.scales, config.lags, config.weather_features
        )
        test_weather = make_supervised(
            prepared.test, prepared.scales, config.lags, config.weather_features
        )
        common_dates = common_evaluation_dates(prepared.test, config.seasonal_period)
        train_sales = prepared.train["sales"].to_numpy(dtype=float)

        def record(
            aligned: pd.DataFrame,
            model: str,
            variant: str,
            seed: int | None,
            predicted_norm: np.ndarray,
            category: str = category,
            train_sales: np.ndarray = train_sales,
        ) -> None:
            metric_rows.append(
                _metric_row(
                    aligned,
                    category=category,
                    model=model,
                    variant=variant,
                    seed=seed,
                    train_sales=train_sales,
                    predicted_norm=predicted_norm,
                    seasonal_period=config.seasonal_period,
                )
            )
            for (_, row), normalized in zip(aligned.iterrows(), predicted_norm, strict=True):
                prediction_rows.append(
                    {
                        "category": category,
                        "model": model,
                        "variant": variant,
                        "seed": seed,
                        "date": row["date"],
                        "actual": row["target"],
                        "predicted": row["predicted"],
                        "actual_norm": row["target_norm"],
                        "predicted_norm": normalized,
                    }
                )

        if "seasonal_naive" in config.models:
            forecast = seasonal_naive(prepared.test, config.seasonal_period)
            aligned = _align(forecast.dates, forecast.values, test_plain, common_dates)
            pred_norm = prepared.scales["sales"].transform(aligned["predicted"])
            record(aligned, "seasonal_naive", "sales_only", None, pred_norm)

        if "arima" in config.models:
            try:
                forecast = arima_forecast(prepared.train, prepared.test, config.profile)
                aligned = _align(forecast.dates, forecast.values, test_plain, common_dates)
                pred_norm = prepared.scales["sales"].transform(aligned["predicted"])
                record(aligned, "arima", "sales_only", None, pred_norm)
            except (OptionalDependencyError, RuntimeError) as exc:
                unavailable.append({"category": category, "model": "arima", "reason": str(exc)})

        if "lstm" in config.models:
            features = [f"sales_lag_{lag}" for lag in reversed(config.lags)]
            for seed in config.seeds:
                try:
                    forecast = lstm_forecast(
                        train_plain,
                        test_plain,
                        features,
                        seed=seed,
                        profile=config.profile,
                    )
                    predicted_original = prepared.scales["sales"].inverse(forecast.values)
                    aligned = _align(forecast.dates, predicted_original, test_plain, common_dates)
                    normalized_map = pd.Series(forecast.values, index=forecast.dates)
                    pred_norm = normalized_map.reindex(aligned["date"]).to_numpy()
                    record(aligned, "lstm", "sales_only", seed, pred_norm)
                except (OptionalDependencyError, RuntimeError) as exc:
                    unavailable.append({"category": category, "model": "lstm", "reason": str(exc)})
                    break

        if "efs" in config.models:
            for variant, train_set, test_set, weather in [
                ("sales_only", train_plain, test_plain, []),
                ("target_weather", train_weather, test_weather, config.weather_features),
            ]:
                features = [f"sales_lag_{lag}" for lag in config.lags] + weather
                for seed in config.seeds:
                    job_dir = run_dir / "efs" / category.replace(" ", "_") / variant / str(seed)
                    write_json(
                        job_dir / "preprocessing.json",
                        {
                            "schema_version": 1,
                            "category": category,
                            "feature_columns": features,
                            "sales_scale": {
                                "minimum": prepared.scales["sales"].minimum,
                                "maximum": prepared.scales["sales"].maximum,
                            },
                            "scales": {
                                name: {
                                    "minimum": scale.minimum,
                                    "maximum": scale.maximum,
                                }
                                for name, scale in prepared.scales.items()
                            },
                            "training_medians": prepared.medians,
                            "input_policy": "training_minmax_then_clip_0_1",
                        },
                    )
                    try:
                        result = runner.train_and_predict(
                            train_set,
                            test_set,
                            feature_columns=features,
                            config=config.efs,
                            seed=seed,
                            output_dir=job_dir,
                        )
                    except (MatlabUnavailableError, RuntimeError) as exc:
                        unavailable.append(
                            {"category": category, "model": f"efs:{variant}", "reason": str(exc)}
                        )
                        break
                    predicted_original = prepared.scales["sales"].inverse(result["prediction_norm"])
                    aligned = _align(
                        pd.DatetimeIndex(result["date"]),
                        predicted_original,
                        test_set,
                        common_dates,
                    )
                    normalized_map = pd.Series(
                        result["prediction_norm"].to_numpy(), index=pd.DatetimeIndex(result["date"])
                    )
                    pred_norm = normalized_map.reindex(aligned["date"]).to_numpy()
                    record(aligned, "efs", variant, seed, pred_norm)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    write_table(run_dir / "metrics.csv", metrics)
    write_table(run_dir / "predictions.csv", predictions)
    write_table(
        run_dir / "unavailable.csv",
        pd.DataFrame(unavailable, columns=["category", "model", "reason"]),
    )
    if not metrics.empty:
        category_summary = (
            metrics.groupby(["category", "model", "variant"], dropna=False)[METRIC_COLUMNS]
            .agg(["mean", "std"])
            .reset_index()
        )
        category_summary.columns = [
            column if isinstance(column, str) else "_".join(part for part in column if part)
            for column in category_summary.columns
        ]
        write_table(run_dir / "category_summary.csv", category_summary)
        macro = (
            metrics.groupby(["model", "variant", "seed"], dropna=False)[METRIC_COLUMNS]
            .mean()
            .reset_index()
        )
        summary = macro.groupby(["model", "variant"], dropna=False)[METRIC_COLUMNS].agg(
            ["mean", "std"]
        )
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        write_table(run_dir / "macro_metrics.csv", macro)
        write_table(run_dir / "macro_summary.csv", summary.reset_index())
        ablation = _weather_ablation(metrics)
        write_table(run_dir / "weather_ablation.csv", ablation)
        if not ablation.empty:
            ablation_summary = (
                ablation.drop(columns=["category", "seed"]).agg(["mean", "std"]).transpose()
            )
            ablation_summary.index.name = "measure"
            write_table(run_dir / "weather_ablation_summary.csv", ablation_summary.reset_index())
    if not predictions.empty:
        save_forecast_plots(predictions, run_dir / "plots")
    save_rule_activation_plots(run_dir)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "categories_completed": categories,
            "metric_rows": len(metrics),
            "unavailable_models": len(unavailable),
        }
    )
    write_json(manifest_path, manifest)
    return run_dir
