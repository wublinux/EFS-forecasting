function [predictionTable, activationTable] = predict(model, inputTable, featureNames)
%PREDICT Evaluate normalized samples and expose rule activation intensity.

arguments
    model
    inputTable table
    featureNames (1,:) string
end

X = inputTable{:, cellstr(featureNames)};
options = evalfisOptions( ...
    EmptyOutputFuzzySetMessage="none", ...
    NoRuleFiredMessage="none", ...
    OutOfRangeInputValueMessage="none");
prediction = evalfis(model, X, options);
predictionTable = table(inputTable.date, prediction, ...
    VariableNames=["date", "prediction_norm"]);

numSamples = height(inputTable);
numRules = numel(model.Rules);
activation = zeros(numSamples, numRules);
for sampleIndex = 1:numSamples
    [~, ~, ~, ~, firing] = evalfis(model, X(sampleIndex, :), options);
    if size(firing, 2) == 2
        activation(sampleIndex, :) = sqrt(firing(:, 1) .* firing(:, 2));
    else
        activation(sampleIndex, :) = firing(:, 1);
    end
end
ruleNames = "rule_" + (1:numRules);
activationTable = array2table(activation, VariableNames=cellstr(ruleNames));
activationTable = addvars(activationTable, inputTable.date, Before=1, NewVariableNames="date");
end
