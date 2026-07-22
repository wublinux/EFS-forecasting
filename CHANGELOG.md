# Changelog

All notable changes to AdaptForecast are recorded here.

## 0.2.0 - 2026-07-23

- Added an independent, explicitly labeled pure-Python interval type-2 zero-order Sugeno backend.
- Added GA rule learning, Pattern Search parameter tuning, and locked-upper uncertainty tuning.
- Added hash-verified Python model persistence, saved-model prediction, rules, activations, and
  training summaries.
- Added backend identity to metrics, predictions, summaries, weather ablations, plots, and
  unavailable-model records.
- Added Python IT2 smoke and private full-run configurations plus Streamlit backend selection.
- Added deterministic unit/integration coverage and backend-aware artifact verification.
- Fixed private-data path resolution before manifest hashing.

## 0.1.0 - 2026-07-13

- Rebuilt the thesis prototype as an auditable MATLAB/Python research project.
- Added leakage-safe data preparation, baselines, MATLAB file contracts, CI, documentation,
  synthetic smoke evidence, Streamlit browsing, and GitHub Pages.
