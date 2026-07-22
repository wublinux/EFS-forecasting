"""Auditable pure-Python interval type-2 zero-order Sugeno implementation.

This backend mirrors the three-stage research protocol used by the MATLAB core while
remaining an explicitly separate implementation.  It uses product t-norm firing,
center-of-sets type reduction, a genetic rule-selection stage, and two local pattern
search stages.  It is intentionally small-data oriented and has no optional runtime
dependency beyond NumPy and pandas.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json, write_table
from .config import EFSConfig

EPSILON = 1e-12
MODEL_SCHEMA_VERSION = 1


@dataclass
class PythonIT2Model:
    """Serializable interval type-2 zero-order Sugeno model."""

    feature_columns: list[str]
    antecedents: np.ndarray
    centers: np.ndarray
    sigmas: np.ndarray
    lower_scale: np.ndarray
    lower_lag: np.ndarray
    consequents: np.ndarray
    weights: np.ndarray
    fallback: float
    seed: int

    def membership_bounds(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        if x.ndim != 2 or x.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected a 2-D input with {len(self.feature_columns)} features, got {x.shape}"
            )
        upper_delta = (x[:, :, None] - self.centers[None, :, :]) / self.sigmas[None, :, :]
        upper = np.exp(-0.5 * np.square(upper_delta))
        lower_centers = self.centers + self.lower_lag
        lower_delta = (x[:, :, None] - lower_centers[None, :, :]) / self.sigmas[None, :, :]
        lower_raw = self.lower_scale[None, :, :] * np.exp(-0.5 * np.square(lower_delta))
        lower = np.minimum(upper, lower_raw)
        return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)

    def firing_bounds(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower_membership, upper_membership = self.membership_bounds(values)
        num_samples = lower_membership.shape[0]
        num_rules = len(self.antecedents)
        lower = np.ones((num_samples, num_rules), dtype=float)
        upper = np.ones((num_samples, num_rules), dtype=float)
        for feature_index in range(len(self.feature_columns)):
            membership_index = self.antecedents[:, feature_index]
            lower *= lower_membership[:, feature_index, membership_index]
            upper *= upper_membership[:, feature_index, membership_index]
        lower *= self.weights[None, :]
        upper *= self.weights[None, :]
        return lower, upper

    def predict_with_activations(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower, upper = self.firing_bounds(values)
        prediction, no_rule = _type_reduce(lower, upper, self.consequents, self.fallback)
        activation = np.sqrt(np.maximum(lower * upper, 0.0))
        return prediction, activation, no_rule

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.predict_with_activations(values)[0]


@dataclass(frozen=True)
class TrainingResult:
    model: PythonIT2Model
    summary: dict[str, object]
    training_activation: np.ndarray


def _type_reduce_endpoint(
    lower: np.ndarray,
    upper: np.ndarray,
    consequents: np.ndarray,
    *,
    left: bool,
) -> np.ndarray:
    midpoint = 0.5 * (lower + upper)
    denominator = midpoint.sum(axis=1)
    current = np.divide(
        midpoint @ consequents,
        denominator,
        out=np.zeros(len(lower), dtype=float),
        where=denominator > EPSILON,
    )
    rule_index = np.arange(len(consequents))[None, :]
    for _ in range(len(consequents) + 1):
        switch = np.searchsorted(consequents, current, side="right") - 1
        low_side = rule_index <= switch[:, None]
        if left:
            selected = np.where(low_side, upper, lower)
        else:
            selected = np.where(low_side, lower, upper)
        selected_sum = selected.sum(axis=1)
        updated = np.divide(
            selected @ consequents,
            selected_sum,
            out=current.copy(),
            where=selected_sum > EPSILON,
        )
        if np.allclose(updated, current, atol=1e-10, rtol=0.0):
            return updated
        current = updated
    return current


def _type_reduce(
    lower: np.ndarray,
    upper: np.ndarray,
    consequents: np.ndarray,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(consequents) == 0:
        return np.full(len(lower), fallback), np.ones(len(lower), dtype=bool)
    order = np.argsort(consequents, kind="stable")
    sorted_consequents = consequents[order]
    sorted_lower = lower[:, order]
    sorted_upper = upper[:, order]
    no_rule = sorted_upper.sum(axis=1) <= EPSILON
    left = _type_reduce_endpoint(sorted_lower, sorted_upper, sorted_consequents, left=True)
    right = _type_reduce_endpoint(sorted_lower, sorted_upper, sorted_consequents, left=False)
    prediction = np.clip(0.5 * (left + right), 0.0, 1.0)
    prediction[no_rule] = fallback
    return prediction, no_rule


def _all_antecedents(num_features: int, num_input_mfs: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product(range(num_input_mfs), repeat=num_features)), dtype=np.int16
    )


def _initial_memberships(num_features: int, num_input_mfs: int) -> tuple[np.ndarray, ...]:
    centers_1d = (np.arange(num_input_mfs, dtype=float) + 0.5) / num_input_mfs
    centers = np.tile(centers_1d, (num_features, 1))
    sigmas = np.full((num_features, num_input_mfs), 0.55 / num_input_mfs)
    lower_scale = np.full_like(centers, 0.8)
    lower_lag = np.zeros_like(centers)
    return centers, sigmas, lower_scale, lower_lag


def _base_firing(
    values: np.ndarray,
    antecedents: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    lower_scale: np.ndarray,
    lower_lag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temporary = PythonIT2Model(
        feature_columns=[f"x{index}" for index in range(values.shape[1])],
        antecedents=antecedents,
        centers=centers,
        sigmas=sigmas,
        lower_scale=lower_scale,
        lower_lag=lower_lag,
        consequents=np.zeros(len(antecedents)),
        weights=np.ones(len(antecedents)),
        fallback=0.5,
        seed=0,
    )
    return temporary.firing_bounds(values)


def _decode_genome(
    genome: np.ndarray, antecedents: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_rules = len(antecedents)
    selection = genome[:num_rules]
    active = selection >= 0.5
    if not np.any(active):
        active[np.argmax(selection)] = True
    weights = np.clip(genome[num_rules : 2 * num_rules], 0.05, 1.0)[active]
    consequents = np.clip(genome[2 * num_rules :], 0.0, 1.0)[active]
    return antecedents[active], weights, consequents, active


def _ga_rule_learning(
    values: np.ndarray,
    target: np.ndarray,
    antecedents: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    lower_scale: np.ndarray,
    lower_lag: np.ndarray,
    fallback: float,
    config: EFSConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    base_lower, base_upper = _base_firing(
        values, antecedents, centers, sigmas, lower_scale, lower_lag
    )
    activation = np.sqrt(np.maximum(base_lower * base_upper, 0.0))
    activation_sum = activation.sum(axis=0)
    initial_consequents = np.divide(
        activation.T @ target,
        activation_sum,
        out=np.full(len(antecedents), fallback, dtype=float),
        where=activation_sum > EPSILON,
    )
    initial_rule_count = min(len(antecedents), max(2, 2 * values.shape[1]))
    strongest = np.argsort(activation_sum)[-initial_rule_count:]
    initial_selection = np.full(len(antecedents), 0.2)
    initial_selection[strongest] = 0.8
    initial = np.concatenate(
        [initial_selection, np.ones(len(antecedents)), np.clip(initial_consequents, 0, 1)]
    )

    population_size = max(4, int(config.population_size))
    population = np.tile(initial, (population_size, 1))
    if population_size > 1:
        population[1:, : len(antecedents)] = rng.uniform(
            0.0, 1.0, (population_size - 1, len(antecedents))
        )
        population[1:, len(antecedents) : 2 * len(antecedents)] = rng.uniform(
            0.35, 1.0, (population_size - 1, len(antecedents))
        )
        jitter = rng.normal(0.0, 0.12, (population_size - 1, len(antecedents)))
        population[1:, 2 * len(antecedents) :] = np.clip(
            initial_consequents[None, :] + jitter, 0.0, 1.0
        )

    def fitness(genome: np.ndarray) -> float:
        _, weights, consequents, active = _decode_genome(genome, antecedents)
        lower = base_lower[:, active] * weights[None, :]
        upper = base_upper[:, active] * weights[None, :]
        prediction, _ = _type_reduce(lower, upper, consequents, fallback)
        mae = float(np.mean(np.abs(prediction - target)))
        complexity = config.complexity_penalty * float(np.mean(active))
        return mae + complexity

    scores = np.asarray([fitness(genome) for genome in population])
    best_index = int(np.argmin(scores))
    best_genome = population[best_index].copy()
    best_score = float(scores[best_index])
    history = [best_score]
    stalled = 0
    completed_generations = 0
    elite_count = max(1, population_size // 10)
    stall_limit = max(5, min(config.ga_stall_generations, config.max_generations))

    def tournament() -> np.ndarray:
        candidates = rng.integers(0, population_size, size=3)
        return population[candidates[np.argmin(scores[candidates])]]

    for generation in range(int(config.max_generations)):
        completed_generations = generation + 1
        elite = population[np.argsort(scores)[:elite_count]].copy()
        children: list[np.ndarray] = []
        progress = generation / max(int(config.max_generations) - 1, 1)
        mutation_scale = 0.18 * (1.0 - progress) + 0.025
        while len(children) < population_size - elite_count:
            first = tournament().copy()
            second = tournament().copy()
            if rng.random() < config.crossover_fraction:
                choose_first = rng.random(len(first)) < 0.5
                child = np.where(choose_first, first, second)
            else:
                child = first
            mutation = rng.random(len(child)) < 0.08
            child[mutation] += rng.normal(0.0, mutation_scale, int(mutation.sum()))
            child = np.clip(child, 0.0, 1.0)
            children.append(child)
        population = np.vstack([elite, np.asarray(children)])
        scores = np.asarray([fitness(genome) for genome in population])
        generation_index = int(np.argmin(scores))
        generation_score = float(scores[generation_index])
        if generation_score < best_score - 1e-8:
            best_score = generation_score
            best_genome = population[generation_index].copy()
            stalled = 0
        else:
            stalled += 1
        history.append(best_score)
        if stalled >= stall_limit:
            break

    learned_antecedents, learned_weights, learned_consequents, _ = _decode_genome(
        best_genome, antecedents
    )
    details: dict[str, object] = {
        "objective": "normalized_mae_plus_rule_complexity",
        "best_objective": best_score,
        "generations": completed_generations,
        "population_size": population_size,
        "candidate_rules": len(antecedents),
        "selected_rules": len(learned_antecedents),
        "history": history,
    }
    return learned_antecedents, learned_weights, learned_consequents, details


def _pattern_search(
    initial: np.ndarray,
    objective: Callable[[np.ndarray], float],
    project: Callable[[np.ndarray], np.ndarray],
    steps: np.ndarray,
    max_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    current = project(initial.copy())
    current_score = float(objective(current))
    history = [current_score]
    evaluations = 1
    completed_iterations = 0
    step = steps.astype(float).copy()
    for iteration in range(int(max_iterations)):
        completed_iterations = iteration + 1
        improved = False
        for index in rng.permutation(len(current)):
            for direction in (1.0, -1.0):
                candidate = current.copy()
                candidate[index] += direction * step[index]
                candidate = project(candidate)
                score = float(objective(candidate))
                evaluations += 1
                if score < current_score - 1e-9:
                    current, current_score = candidate, score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
        history.append(current_score)
        if float(np.max(step)) < 1e-3:
            break
    return current, {
        "objective": "normalized_mae",
        "best_objective": current_score,
        "iterations": completed_iterations,
        "evaluations": evaluations,
        "final_max_step": float(np.max(step)),
        "history": history,
    }


def _membership_center_bounds(num_input_mfs: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.arange(num_input_mfs, dtype=float) / num_input_mfs + 0.02
    upper = (np.arange(num_input_mfs, dtype=float) + 1.0) / num_input_mfs - 0.02
    return lower, upper


def train_python_it2(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: EFSConfig,
    seed: int,
) -> TrainingResult:
    """Train the explicit Python IT2 backend with the agreed three-stage protocol."""
    started = time.perf_counter()
    values = train[feature_columns].to_numpy(dtype=float)
    target = train["target_norm"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("Python IT2 training data must be finite")
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Python IT2 requires at least one training sample")
    if np.min(values) < -EPSILON or np.max(values) > 1.0 + EPSILON:
        raise ValueError("Python IT2 inputs must be normalized to [0,1]")

    rng = np.random.default_rng(seed)
    fallback = float(np.clip(np.median(target), 0.0, 1.0))
    num_features = len(feature_columns)
    num_input_mfs = int(config.num_input_mfs)
    candidate_antecedents = _all_antecedents(num_features, num_input_mfs)
    centers, sigmas, lower_scale, lower_lag = _initial_memberships(num_features, num_input_mfs)

    antecedents, weights, consequents, stage1 = _ga_rule_learning(
        values,
        target,
        candidate_antecedents,
        centers,
        sigmas,
        lower_scale,
        lower_lag,
        fallback,
        config,
        rng,
    )

    center_count = centers.size
    sigma_count = sigmas.size
    stage2_initial = np.concatenate([centers.ravel(), sigmas.ravel(), consequents])
    center_lower, center_upper = _membership_center_bounds(num_input_mfs)

    def decode_stage2(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tuned_centers = parameters[:center_count].reshape(centers.shape).copy()
        tuned_sigmas = parameters[center_count : center_count + sigma_count].reshape(sigmas.shape)
        tuned_consequents = parameters[center_count + sigma_count :]
        tuned_centers = np.clip(tuned_centers, center_lower[None, :], center_upper[None, :])
        return (
            tuned_centers,
            np.clip(tuned_sigmas, 0.04, 0.6),
            np.clip(tuned_consequents, 0.0, 1.0),
        )

    def project_stage2(parameters: np.ndarray) -> np.ndarray:
        current_centers, current_sigmas, current_consequents = decode_stage2(parameters)
        return np.concatenate(
            [current_centers.ravel(), current_sigmas.ravel(), current_consequents]
        )

    def objective_stage2(parameters: np.ndarray) -> float:
        current_centers, current_sigmas, current_consequents = decode_stage2(parameters)
        lower, upper = _base_firing(
            values,
            antecedents,
            current_centers,
            current_sigmas,
            lower_scale,
            lower_lag,
        )
        prediction, _ = _type_reduce(
            lower * weights[None, :],
            upper * weights[None, :],
            current_consequents,
            fallback,
        )
        return float(np.mean(np.abs(prediction - target)))

    stage2_steps = np.concatenate(
        [np.full(center_count, 0.06), np.full(sigma_count, 0.05), np.full(len(consequents), 0.06)]
    )
    stage2_parameters, stage2 = _pattern_search(
        stage2_initial,
        objective_stage2,
        project_stage2,
        stage2_steps,
        config.pattern_max_iterations,
        rng,
    )
    centers, sigmas, consequents = decode_stage2(stage2_parameters)
    locked_upper = np.concatenate([centers.ravel(), sigmas.ravel()]).copy()

    scale_count = lower_scale.size
    stage3_initial = np.concatenate([lower_scale.ravel(), lower_lag.ravel()])

    def decode_stage3(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_scale = parameters[:scale_count].reshape(lower_scale.shape)
        current_lag = parameters[scale_count:].reshape(lower_lag.shape)
        return np.clip(current_scale, 0.1, 1.0), np.clip(current_lag, -0.2, 0.2)

    def project_stage3(parameters: np.ndarray) -> np.ndarray:
        current_scale, current_lag = decode_stage3(parameters)
        return np.concatenate([current_scale.ravel(), current_lag.ravel()])

    def objective_stage3(parameters: np.ndarray) -> float:
        current_scale, current_lag = decode_stage3(parameters)
        lower, upper = _base_firing(
            values,
            antecedents,
            centers,
            sigmas,
            current_scale,
            current_lag,
        )
        prediction, _ = _type_reduce(
            lower * weights[None, :], upper * weights[None, :], consequents, fallback
        )
        return float(np.mean(np.abs(prediction - target)))

    stage3_steps = np.concatenate([np.full(scale_count, 0.06), np.full(lower_lag.size, 0.025)])
    stage3_parameters, stage3 = _pattern_search(
        stage3_initial,
        objective_stage3,
        project_stage3,
        stage3_steps,
        config.pattern_max_iterations,
        rng,
    )
    lower_scale, lower_lag = decode_stage3(stage3_parameters)
    upper_after_stage3 = np.concatenate([centers.ravel(), sigmas.ravel()])
    upper_parameter_max_delta = float(np.max(np.abs(upper_after_stage3 - locked_upper)))

    order = np.argsort(consequents, kind="stable")
    model = PythonIT2Model(
        feature_columns=list(feature_columns),
        antecedents=antecedents[order],
        centers=centers,
        sigmas=sigmas,
        lower_scale=lower_scale,
        lower_lag=lower_lag,
        consequents=consequents[order],
        weights=weights[order],
        fallback=fallback,
        seed=seed,
    )
    training_prediction, training_activation, no_rule = model.predict_with_activations(values)
    summary: dict[str, object] = {
        "schema_version": 1,
        "backend": "python-it2",
        "algorithm": "interval_type2_zero_order_sugeno_center_of_sets",
        "seed": seed,
        "num_inputs": num_features,
        "num_input_mfs": num_input_mfs,
        "num_rules": len(model.antecedents),
        "elapsed_seconds": time.perf_counter() - started,
        "optimization": "ga_then_patternsearch_then_type2_uncertainty",
        "training_mae_norm": float(np.mean(np.abs(training_prediction - target))),
        "training_no_rule_samples": int(no_rule.sum()),
        "upper_parameter_max_delta": upper_parameter_max_delta,
        "upper_parameters_locked": upper_parameter_max_delta <= EPSILON,
        "stage_1_rule_learning": stage1,
        "stage_2_membership_and_output_tuning": stage2,
        "stage_3_lower_uncertainty_tuning": stage3,
        "lower_membership_definition": (
            "min(upper, LowerScale * gaussian(x; center + LowerLag, sigma))"
        ),
    }
    return TrainingResult(model, summary, training_activation)


def save_python_it2_model(model: PythonIT2Model, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / "model.npz"
    np.savez_compressed(
        array_path,
        antecedents=model.antecedents,
        centers=model.centers,
        sigmas=model.sigmas,
        lower_scale=model.lower_scale,
        lower_lag=model.lower_lag,
        consequents=model.consequents,
        weights=model.weights,
    )
    metadata_path = output_dir / "model.json"
    write_json(
        metadata_path,
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "backend": "python-it2",
            "model_type": "interval_type2_zero_order_sugeno",
            "array_file": array_path.name,
            "array_sha256": sha256_file(array_path),
            "feature_columns": model.feature_columns,
            "fallback": model.fallback,
            "seed": model.seed,
        },
    )
    return array_path, metadata_path


def load_python_it2_model(path: str | Path) -> PythonIT2Model:
    supplied = Path(path).resolve()
    metadata_path = (
        supplied if supplied.suffix.lower() == ".json" else supplied.with_suffix(".json")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("backend") != "python-it2":
        raise ValueError(f"Not a Python IT2 model: {metadata_path}")
    array_path = metadata_path.parent / str(metadata["array_file"])
    if sha256_file(array_path) != metadata.get("array_sha256"):
        raise ValueError(f"Python IT2 model array hash mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        return PythonIT2Model(
            feature_columns=list(metadata["feature_columns"]),
            antecedents=arrays["antecedents"],
            centers=arrays["centers"],
            sigmas=arrays["sigmas"],
            lower_scale=arrays["lower_scale"],
            lower_lag=arrays["lower_lag"],
            consequents=arrays["consequents"],
            weights=arrays["weights"],
            fallback=float(metadata["fallback"]),
            seed=int(metadata["seed"]),
        )


def _consequent_label(value: float) -> str:
    labels = ["critical_low", "low", "medium", "high", "peak"]
    return labels[min(int(np.clip(value, 0.0, 1.0) * len(labels)), len(labels) - 1)]


def _membership_label(index: int) -> str:
    if index == 0:
        return "low"
    if index == 1:
        return "high"
    return f"mf_{index + 1}"


def explain_python_it2(model: PythonIT2Model, training_activation: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule_index, antecedent in enumerate(model.antecedents):
        clauses = [
            f"{feature} is {_membership_label(int(membership))}"
            for feature, membership in zip(model.feature_columns, antecedent, strict=True)
        ]
        activation = training_activation[:, rule_index]
        rows.append(
            {
                "rule_id": rule_index + 1,
                "antecedent": " AND ".join(clauses),
                "consequent": _consequent_label(float(model.consequents[rule_index])),
                "consequent_value_norm": float(model.consequents[rule_index]),
                "weight": float(model.weights[rule_index]),
                "support": int(np.sum(activation > 0.01)),
                "mean_activation": float(np.mean(activation)),
                "max_activation": float(np.max(activation)),
            }
        )
    return pd.DataFrame(rows)


class PythonIT2Runner:
    """Execute Python IT2 training through the same JSON/CSV artifact contract."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()

    def prepare_training_job(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        feature_columns: list[str],
        config: EFSConfig,
        seed: int,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs = output_dir / "inputs"
        inputs.mkdir(exist_ok=True)
        train_path, test_path = inputs / "train.csv", inputs / "test.csv"
        write_table(train_path, train[["date", *feature_columns, "target_norm"]])
        write_table(test_path, test[["date", *feature_columns, "target_norm"]])
        job = {
            "schema_version": 1,
            "backend": "python-it2",
            "train_csv": str(train_path.resolve()),
            "test_csv": str(test_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "feature_columns": feature_columns,
            "target_column": "target_norm",
            "seed": seed,
            "efs": asdict(config),
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
        }
        job_path = output_dir / "job.json"
        write_json(job_path, job)
        return job_path

    def train_and_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        feature_columns: list[str],
        config: EFSConfig,
        seed: int,
        output_dir: Path,
    ) -> pd.DataFrame:
        self.prepare_training_job(
            train,
            test,
            feature_columns=feature_columns,
            config=config,
            seed=seed,
            output_dir=output_dir,
        )
        result = train_python_it2(train, feature_columns, config, seed)
        save_python_it2_model(result.model, output_dir)
        values = test[feature_columns].to_numpy(dtype=float)
        prediction, activation, no_rule = result.model.predict_with_activations(values)
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(test["date"]),
                "prediction_norm": prediction,
                "no_rule_fired": no_rule,
            }
        )
        write_table(output_dir / "predictions.csv", predictions)
        activations = pd.DataFrame(
            activation,
            columns=[f"rule_{index + 1}" for index in range(activation.shape[1])],
        )
        activations.insert(0, "date", pd.to_datetime(test["date"]).to_numpy())
        write_table(output_dir / "activations.csv", activations)
        write_table(
            output_dir / "rules.csv",
            explain_python_it2(result.model, activation),
        )
        summary = dict(result.summary)
        summary["test_no_rule_samples"] = int(no_rule.sum())
        write_json(output_dir / "training_summary.json", summary)
        return predictions

    def predict_saved_model(
        self, model_path: Path, input_csv: Path, output_csv: Path, feature_columns: list[str]
    ) -> None:
        model = load_python_it2_model(model_path)
        if feature_columns != model.feature_columns:
            expected = model.feature_columns
            raise ValueError(
                f"Feature contract mismatch: model expects {expected}, got {feature_columns}"
            )
        inputs = pd.read_csv(input_csv, parse_dates=["date"])
        prediction, _, no_rule = model.predict_with_activations(
            inputs[feature_columns].to_numpy(dtype=float)
        )
        write_table(
            output_csv,
            pd.DataFrame(
                {
                    "date": inputs["date"],
                    "prediction_norm": prediction,
                    "no_rule_fired": no_rule,
                }
            ),
        )
