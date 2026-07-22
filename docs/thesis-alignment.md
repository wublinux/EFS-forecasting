# Thesis alignment

The final-year report supplies the research motivation: small-data supply-chain forecasting,
external weather variables, interval type-2 fuzzy uncertainty, evolutionary optimization, and
human-readable rules. The rebuilt system retains those testable ideas while correcting the
prototype's methodological gaps.

| Report or prototype statement | Repository treatment |
| --- | --- |
| 2016 training and 2017 validation | Implemented as a strict year holdout |
| Real weather as external information | Canonical target-date weather interface implemented |
| Prototype-generated random weather | Removed |
| Full-data normalization | Replaced by training-only preprocessing |
| GA plus Pattern Search | Implemented in MATLAB and independently in the labeled Python IT2 backend |
| ARIMA/LSTM comparisons | Implemented as shared-split baselines |
| Interpretable fuzzy rules and heat map | Exported as rule and activation artifacts |
| Online model evolution | Not implemented; future work |
| Typhoon robustness and 3–5 day warning | Not established by available experiments |
| Six planning granularities | Not implemented |
| Causal weather effects | Not claimed; associations only |

The original PDF is intentionally not included in the public repository. This page is the public
traceability record between its concepts and current executable evidence.

The legacy prototype files are also not treated as reproduced evidence: they used combined-year
normalization, synthetic weather, a non-independent split, and did not retain a completed fuzzy
model artifact. They remain private historical material rather than inputs to published metrics.
