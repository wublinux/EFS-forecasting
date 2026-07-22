"""Typed benchmark configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class EFSConfig:
    backend: str = "matlab"
    executable: str = "matlab"
    num_input_mfs: int = 2
    population_size: int = 100
    max_generations: int = 300
    crossover_fraction: float = 0.8
    pattern_max_iterations: int = 100
    ga_stall_generations: int = 30
    complexity_penalty: float = 0.002
    use_parallel: bool = True


@dataclass
class BenchmarkConfig:
    data_path: str = "data/sample/synthetic_demand.csv"
    artifact_root: str = "artifacts"
    train_year: int = 2016
    test_year: int = 2017
    lags: list[int] = field(default_factory=lambda: [1, 2, 3])
    weather_features: list[str] = field(
        default_factory=lambda: ["avg_temp", "humidity", "rainfall"]
    )
    categories: list[str] | str = "all"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    models: list[str] = field(default_factory=lambda: ["seasonal_naive", "arima", "lstm", "efs"])
    seasonal_period: int = 7
    profile: str = "full"
    efs: EFSConfig = field(default_factory=EFSConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> BenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    efs = EFSConfig(**raw.pop("efs", {}))
    config = BenchmarkConfig(**raw, efs=efs)
    if executable := os.getenv("ADAPTFORECAST_MATLAB_EXECUTABLE"):
        config.efs.executable = executable
    if backend := os.getenv("ADAPTFORECAST_EFS_BACKEND"):
        config.efs.backend = backend
    if config.profile == "smoke":
        config.seeds = config.seeds[:1]
        config.efs.population_size = min(config.efs.population_size, 10)
        config.efs.max_generations = min(config.efs.max_generations, 2)
        config.efs.pattern_max_iterations = min(config.efs.pattern_max_iterations, 5)
        config.efs.use_parallel = False
    if sorted(config.lags) != config.lags or not config.lags or min(config.lags) < 1:
        raise ValueError("lags must be a sorted, non-empty list of positive integers")
    if config.train_year == config.test_year:
        raise ValueError("train_year and test_year must differ")
    if config.efs.backend not in {"matlab", "python-it2"}:
        raise ValueError("efs.backend must be 'matlab' or 'python-it2'")
    if config.efs.num_input_mfs < 2:
        raise ValueError("efs.num_input_mfs must be at least 2")
    if config.efs.population_size < 4 or config.efs.max_generations < 1:
        raise ValueError("EFS GA requires population_size >= 4 and max_generations >= 1")
    if not 0 <= config.efs.crossover_fraction <= 1:
        raise ValueError("efs.crossover_fraction must be between 0 and 1")
    if config.efs.pattern_max_iterations < 1 or config.efs.ga_stall_generations < 1:
        raise ValueError("EFS iteration limits must be positive")
    if config.efs.complexity_penalty < 0:
        raise ValueError("efs.complexity_penalty must be non-negative")
    return config
