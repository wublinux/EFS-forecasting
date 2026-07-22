# Methodology

## Forecast definition

For each product category, the target is demand on date `t`. Inputs are sales on `t-1`, `t-2`,
and `t-3`, plus weather known or forecast for `t`. This corrects the ambiguous alignment in the
original prototype, which used current weather to forecast a different date.

## Leakage controls

The primary protocol trains on 2016 and evaluates on 2017. Forward filling is performed within a
year, and training medians are the only fallback. Min-Max parameters come exclusively from the
training year. Test targets that were originally missing are excluded instead of imputed.

Every model is evaluated from the eighth valid test day onward so the weekly seasonal-naive
baseline and three-lag models share exactly the same target dates.

## Models

- **Seasonal naive:** demand from seven days earlier.
- **ARIMA:** `(p,d,q)` selected by training-set AIC; the full grid uses `p,q=0..3`, `d=0..1`.
  Evaluation is rolling one-step-ahead: each observed test-day sale becomes history only after
  that day's forecast has been made.
- **LSTM:** one 32-unit layer and linear output, Adam at `1e-3`, temporal 80/20 training
  validation, maximum 300 epochs, patience 30.
- **EFS sales-only:** three normalized lag inputs.
- **EFS target-weather:** the same lags plus normalized average temperature, humidity, and
  rainfall for the target date.

EFS first learns a rule base with a genetic algorithm. It then performs Pattern Search parameter
tuning, followed by a separate Pattern Search of type-2 lower-scale/lower-lag parameters while
upper parameters remain fixed. Two input membership functions give at most 8 sales-only or 64
weather-model rules.

### EFS backend identity

The MATLAB R2024b implementation is the reference implementation for thesis alignment. The
optional `python-it2` backend is an independent zero-order Sugeno implementation for environments
without MATLAB. It uses product t-norm firing, interval firing bounds, center-of-sets type
reduction, GA rule selection, Pattern Search of upper membership/consequent parameters, and a
final Pattern Search of lower uncertainty while the upper parameters are locked. Its lower
membership is explicitly defined as
`min(upper, LowerScale * gaussian(x; center + LowerLag, sigma))`.

These backends share data, splits, seeds, metrics, and artifact fields but are not presumed
numerically equivalent. The manifest, job, training summary, metrics, predictions, and ablation
tables identify the selected backend.

## Metrics and explanations

RMSE, MAE, MASE, and sMAPE are computed in original sales units. Normalized RMSE and MAE are
reported only for thesis comparison. Three stochastic seeds are summarized as mean and standard
deviation at both category and eight-category macro levels. Weather-ablation percentages compare
the weather and sales-only EFS variants using the same metric and seed; positive values mean the
weather variant reduced error.

Rules are exported with semantic antecedents/consequents, weights, empirical support, and mean
and maximum activation on the test input window. They describe associations learned from the
data and are not causal claims.
