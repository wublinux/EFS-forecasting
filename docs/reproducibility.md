# Reproducibility

## Public smoke run

```bash
pip install -e ".[dev,benchmark,app]"
adaptforecast validate-data data/sample/synthetic_demand.csv
adaptforecast benchmark --config configs/benchmark.smoke.yaml
```

The smoke profile uses one category, one seed, two GA generations, and five Pattern Search
iterations. Its purpose is interface verification, not model comparison.

On a host without MATLAB, run the independent Python IT2 contract instead:

```bash
adaptforecast benchmark --config configs/benchmark.python-it2.smoke.yaml
```

The two smoke runs are separate backend checks and should not be merged into one performance
claim.

## Full audit

Run the full configuration with MATLAB R2024b+ and the required toolboxes:

```bash
adaptforecast benchmark --config configs/benchmark.yaml
```

The resulting artifact directory is authoritative. Preserve its `manifest.json`, data SHA-256,
configuration, per-category and macro summaries, weather-ablation tables, predictions, rules,
activations, and model files together.
The manifest status is `partial` whenever a configured model is unavailable; only runs with every
configured model present are marked `complete`.

## Private data

Do not copy unlicensed source data into this repository. Set `ADAPTFORECAST_DATA_DIR`, run
`adaptforecast prepare` when the legacy converter is needed, and reference the resulting file by
name in a local configuration. The manifest records its hash without publishing its records.
`configs/benchmark.private.example.yaml` resolves the input through that environment variable and
writes full-run artifacts to the repository's parent directory.
Use `configs/benchmark.private.python-it2.example.yaml` for the independent Python IT2 full run.
Its artifact has the same year split and metric protocol, but its EFS backend identity remains
`python-it2` throughout the evidence tables.

## Continuous integration

Python CI verifies conversion, leakage barriers, date alignment, metrics, artifact contracts,
linting, documentation, and deterministic Python IT2 inference/training. MATLAB CI separately
uses a deliberately small synthetic training job. Full three-seed optimization remains an
explicit local research run because it is computationally expensive.

The public runner uses a staged file contract because temporary MATLAB Actions licensing is
available to the official runner action rather than arbitrary child processes. Python first
writes hash-identified jobs, the official action executes them, and the normal benchmark accepts
the results only when the job configuration and input CSV hashes match exactly. Local execution
continues to use `matlab -batch` directly.

On hosts where MATLAB is not on `PATH`, set `ADAPTFORECAST_MATLAB_EXECUTABLE` to the full path
of the MATLAB executable. CI obtains this path from the official `setup-matlab` action output.
