function metrics = evaluate(yTrue, yPred, trainingSeries, seasonalPeriod)
%EVALUATE Calculate original-scale forecast metrics.

arguments
    yTrue (:,1) double
    yPred (:,1) double
    trainingSeries (:,1) double
    seasonalPeriod (1,1) double {mustBeInteger,mustBePositive} = 7
end

valid = isfinite(yTrue) & isfinite(yPred);
actual = yTrue(valid);
predicted = yPred(valid);
errors = predicted - actual;
metrics = struct;
metrics.rmse = sqrt(mean(errors .^ 2));
metrics.mae = mean(abs(errors));
denominator = (abs(actual) + abs(predicted)) / 2;
terms = zeros(size(errors));
nonzero = denominator > 0;
terms(nonzero) = abs(errors(nonzero)) ./ denominator(nonzero);
metrics.smape = 100 * mean(terms);
if numel(trainingSeries) > seasonalPeriod
    scale = mean(abs(trainingSeries(1 + seasonalPeriod:end) - ...
        trainingSeries(1:end - seasonalPeriod)));
    metrics.mase = metrics.mae / scale;
else
    metrics.mase = NaN;
end
metrics.n = numel(actual);
end

