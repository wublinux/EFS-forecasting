# Reproducibility

## Public smoke run

```bash
pip install -e ".[dev,benchmark,app]"
adaptforecast validate-data data/sample/synthetic_demand.csv
adaptforecast benchmark --config configs/benchmark.smoke.yaml
```

The smoke profile uses one category, one seed, two GA generations, and five Pattern Search
iterations. Its purpose is interface verification, not model comparison.

## Full audit

Run the full configuration with MATLAB R2024b+ and the required toolboxes:

```bash
adaptforecast benchmark --config configs/benchmark.yaml
```

The resulting artifact directory is authoritative. Preserve its `manifest.json`, data SHA-256,
configuration, metrics, predictions, rules, activations, and model files together.

## Private data

Do not copy unlicensed source data into this repository. Set `ADAPTFORECAST_DATA_DIR`, run
`adaptforecast prepare` when the legacy converter is needed, and reference the resulting file by
name in a local configuration. The manifest records its hash without publishing its records.

## Continuous integration

Python CI verifies conversion, leakage barriers, date alignment, metrics, artifact contracts,
linting, and documentation. MATLAB CI uses a deliberately small synthetic training job. Full
three-seed optimization remains an explicit local research run because it is computationally
expensive.

