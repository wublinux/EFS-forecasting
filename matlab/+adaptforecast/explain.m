function ruleTable = explain(model, activationTable)
%EXPLAIN Export semantic rules and empirical activation statistics.

numRules = numel(model.Rules);
antecedentText = strings(numRules, 1);
consequentText = strings(numRules, 1);
weights = zeros(numRules, 1);
support = zeros(numRules, 1);
meanActivation = zeros(numRules, 1);
maxActivation = zeros(numRules, 1);
activation = activationTable{:, 2:end};

for ruleIndex = 1:numRules
    rule = model.Rules(ruleIndex);
    clauses = strings(0, 1);
    for inputIndex = 1:numel(rule.Antecedent)
        mfIndex = rule.Antecedent(inputIndex);
        if mfIndex == 0
            continue
        end
        prefix = "";
        if mfIndex < 0
            prefix = "NOT ";
            mfIndex = abs(mfIndex);
        end
        inputName = string(model.Inputs(inputIndex).Name);
        mfName = string(model.Inputs(inputIndex).MembershipFunctions(mfIndex).Name);
        clauses(end + 1) = prefix + inputName + " is " + mfName; %#ok<AGROW>
    end
    antecedentText(ruleIndex) = strjoin(clauses, " AND ");
    outputIndex = abs(rule.Consequent(1));
    consequentText(ruleIndex) = string( ...
        model.Outputs(1).MembershipFunctions(outputIndex).Name);
    weights(ruleIndex) = rule.Weight;
    current = activation(:, ruleIndex);
    support(ruleIndex) = sum(current > .01);
    meanActivation(ruleIndex) = mean(current);
    maxActivation(ruleIndex) = max(current);
end

ruleTable = table((1:numRules)', antecedentText, consequentText, weights, support, ...
    meanActivation, maxActivation, VariableNames=["rule_id", "antecedent", ...
    "consequent", "weight", "support", "mean_activation", "max_activation"]);
end

