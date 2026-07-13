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
- **LSTM:** one 32-unit layer and linear output, Adam at `1e-3`, temporal 80/20 training
  validation, maximum 300 epochs, patience 30.
- **EFS sales-only:** three normalized lag inputs.
- **EFS target-weather:** the same lags plus normalized average temperature, humidity, and
  rainfall for the target date.

EFS first learns a rule base with a genetic algorithm and Pattern Search hybrid refinement. It
then performs local parameter tuning, followed by type-2 lower-scale/lower-lag tuning with upper
parameters fixed. Two input membership functions give at most 8 sales-only or 64 weather-model
rules.

## Metrics and explanations

RMSE, MAE, MASE, and sMAPE are computed in original sales units. Normalized RMSE and MAE are
reported only for thesis comparison. Three stochastic seeds are summarized as mean and standard
deviation.

Rules are exported with semantic antecedents/consequents, weights, empirical support, and mean
and maximum activation. They describe associations learned from the data and are not causal
claims.

