# EFS-forecasting / AdaptForecast

[![Python CI](https://github.com/wublinux/EFS-forecasting/actions/workflows/python.yml/badge.svg)](https://github.com/wublinux/EFS-forecasting/actions/workflows/python.yml)
[![MATLAB CI](https://github.com/wublinux/EFS-forecasting/actions/workflows/matlab.yml/badge.svg)](https://github.com/wublinux/EFS-forecasting/actions/workflows/matlab.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AdaptForecast is an auditable research system for **next-day, category-level demand
forecasting on small datasets**. Its reference MATLAB core learns an interval type-2 Sugeno
fuzzy inference system (EFS). Python provides leakage-safe data preparation, statistical and
neural baselines, experiment artifacts, a CLI, a local Streamlit interface, and an explicitly
identified independent `python-it2` backend for hosts without MATLAB.

The project originated from a final-year report on evolutionary fuzzy systems for supply-chain
forecasting. This repository intentionally distinguishes the report's motivation from claims
that have been reproduced by the current code.

> **Evidence status:** the software and synthetic smoke workflow are testable. Results from the
> private thesis dataset are not shipped and must be regenerated locally before they are cited.
> Online evolution, event-warning lead times, causal effects, and multi-granularity planning are
> not implemented capabilities.

[中文说明](README.zh-CN.md) · [Documentation](https://wublinux.github.io/EFS-forecasting/)

## What the system does

- Uses sales at `t-1`, `t-2`, and `t-3` plus weather known or forecast for target date `t`.
- Trains on 2016 and holds out 2017; imputation and scaling are fitted on training data only.
- Compares weekly seasonal-naive, ARIMA, a 32-unit LSTM, EFS without weather, and EFS with
  target-date weather.
- Reports original-scale RMSE, MAE, MASE, and sMAPE plus normalized RMSE/MAE for thesis
  alignment.
- Exports predictions, configuration, input hashes, fuzzy rules, rule support, and activation
  intensity for every run.

The EFS training stages are: GA rule learning, Pattern Search parameter tuning, then Pattern
Search tuning of type-2 lower scale and lag while upper parameters are fixed.

## Requirements

- Python 3.10+
- MATLAB R2024b or later
- Fuzzy Logic Toolbox and Global Optimization Toolbox
- Parallel Computing Toolbox is optional but recommended for the full profile

MATLAB is not required to validate data, inspect artifacts, run Python baselines, or train the
independent Python IT2 backend. The MATLAB and Python IT2 implementations share the protocol but
are not assumed to be numerically equivalent; every new metric row records its backend.
Set `ADAPTFORECAST_MATLAB_EXECUTABLE` when MATLAB is installed outside the system `PATH`.
The repository includes one explicitly labeled synthetic smoke snapshot so this browsing mode is
useful immediately after a clean checkout.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,benchmark,app]"

adaptforecast validate-data data/sample/synthetic_demand.csv
adaptforecast benchmark --config configs/benchmark.smoke.yaml
adaptforecast app
```

The smoke configuration deliberately reduces evolutionary optimization iterations. It verifies
the pipeline; it is not research evidence. Run `configs/benchmark.yaml` for the three-seed full
audit.

Without MATLAB, run the separately identified Python IT2 smoke workflow:

```bash
adaptforecast benchmark --config configs/benchmark.python-it2.smoke.yaml
```

### Private data

The repository contains only deterministic synthetic data. To use an authorized private file,
store it outside the repository and set:

```powershell
$env:ADAPTFORECAST_DATA_DIR = "C:\path\to\private-data"
```

Convert the original 27-column thesis layout without publishing it:

```bash
adaptforecast prepare private-wide.csv canonical-private.csv --legacy-wide
```

Then run the private audit without copying data or results into the repository:

```powershell
$env:ADAPTFORECAST_DATA_DIR = "C:\path\to\private-data"
adaptforecast benchmark --config configs/benchmark.private.example.yaml
```

For a complete three-seed run using the independent Python implementation instead, use
`configs/benchmark.private.python-it2.example.yaml`. Do not combine its EFS metrics with MATLAB
EFS metrics as if they came from the same implementation.

See [data/README.md](data/README.md) for the canonical schema. Data is not licensed under the
MIT software license.

## Public interfaces

```text
adaptforecast validate-data INPUT.csv
adaptforecast prepare INPUT.csv OUTPUT.csv
adaptforecast benchmark --config configs/benchmark.yaml
adaptforecast predict --model model.mat|model.npz --input prepared.csv --features sales_lag_1,...
adaptforecast app
```

MATLAB exposes `adaptforecast.train`, `adaptforecast.predict`, `adaptforecast.evaluate`, and
`adaptforecast.explain`. Python invokes them through `matlab -batch` and versioned JSON/CSV
contracts, not MATLAB Engine. The independent Python backend uses the same file contract and
exports a hash-verified `model.npz` plus human-readable `model.json` metadata.

With MATLAB R2024b+ installed, a clean checkout can reproduce prediction from the checked-in
synthetic sales-only smoke model:

```bash
adaptforecast predict \
  --model data/precomputed/synthetic-smoke/efs/ice_cream/sales_only/42/model.mat \
  --input data/precomputed/synthetic-smoke/efs/ice_cream/sales_only/42/inputs/test.csv \
  --features sales_lag_1,sales_lag_2,sales_lag_3 \
  --output predictions-example.csv
```

This example validates model loading and the file contract; it is not a performance claim.

Each benchmark creates `artifacts/<run-id>/` with a manifest, data SHA-256, metrics,
predictions, unavailable-model reasons, backend identity, and EFS model/rule/activation files.

## Repository map

```text
src/adaptforecast/    Python data, benchmark, IT2 backend, CLI, bridge, and Streamlit code
matlab/+adaptforecast MATLAB interval type-2 EFS implementation
configs/              Smoke and full experiment protocols
data/sample/          Synthetic public demonstration data
tests/                Python and MATLAB tests
docs/                 Method, reproducibility, thesis alignment, limitations
```

## Reproducibility and responsible interpretation

- All models share an explicit evaluation window and the same year-based holdout.
- Stochastic EFS and LSTM runs use seeds 42, 43, and 44 in the full protocol.
- Rules and weather relationships are learned associations, not causal findings.
- Do not quote benchmark superiority until the artifact manifest and dataset hash for that run
  are available.

Read [Methodology](docs/methodology.md), [Reproducibility](docs/reproducibility.md), and
[Audited results](docs/results.md) before interpreting results; the smoke table is integration
evidence, not a research comparison. See [Limitations](docs/limitations.md) for scope boundaries.

## License and citation

Code is licensed under the [MIT License](LICENSE). The license does not cover private or
third-party datasets. Citation metadata is provided in [CITATION.cff](CITATION.cff).
