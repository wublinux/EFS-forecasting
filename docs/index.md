# AdaptForecast

AdaptForecast is the system name used by the EFS-forecasting research project. It investigates
whether an interval type-2 fuzzy system can provide useful next-day demand forecasts on small
category datasets while exposing readable rules and rule-activation evidence.

The public repository ships a deterministic synthetic dataset, not the thesis dataset. Published
performance tables must be generated from an auditable artifact run and must identify the data
hash used.

## System flow

1. Validate or convert canonical category/date demand records.
2. Split by year before fitting imputation or normalization.
3. Generate lagged sales and target-date weather features.
4. Train baselines and EFS weather/no-weather variants.
5. Evaluate all models on the same dates and original scale.
6. Export predictions, metrics, fuzzy rules, and activation intensities.

The local Streamlit interface can run either the reference MATLAB backend or the separately
identified Python IT2 backend, and can browse precomputed artifacts without either training
environment.
