function [model, trainingSummary] = train(trainTable, config)
%TRAIN Learn rules and tune an interval type-2 Sugeno demand model.

arguments
    trainTable table
    config struct
end

featureNames = string(config.feature_columns);
targetColumn = string(config.target_column);
X = trainTable{:, cellstr(featureNames)};
y = trainTable{:, char(targetColumn)};
if any(~isfinite(X), "all") || any(~isfinite(y), "all")
    error("adaptforecast:InvalidTrainingData", "Training data must be finite and normalized.");
end

rng(double(config.seed), "twister");
efsConfig = config.efs;
fisin = adaptforecast.initializeFIS(featureNames, double(efsConfig.num_input_mfs));
started = tic;

% Stage 1: global rule learning with a GA and local hybrid refinement.
learnOptions = tunefisOptions(Method="ga", OptimizationType="learning");
learnOptions.NumMaxRules = double(efsConfig.num_input_mfs) ^ numel(featureNames);
learnOptions.MethodOptions.PopulationSize = double(efsConfig.population_size);
learnOptions.MethodOptions.MaxGenerations = double(efsConfig.max_generations);
learnOptions.MethodOptions.CrossoverFraction = double(efsConfig.crossover_fraction);
learnOptions.MethodOptions.HybridFcn = @patternsearch;
learnOptions.UseParallel = logical(efsConfig.use_parallel);
fisRules = tunefis(fisin, [], X, y, learnOptions);

% Stage 2: tune upper input parameters and output consequents locally.
[inputSettings, outputSettings] = getTunableSettings(fisRules, AsymmetricLag=true);
for inputIndex = 1:numel(inputSettings)
    for mfIndex = 1:numel(inputSettings(inputIndex).MembershipFunctions)
        inputSettings(inputIndex).MembershipFunctions(mfIndex).LowerScale.Free = false;
        inputSettings(inputIndex).MembershipFunctions(mfIndex).LowerLag.Free = false;
    end
end
tuneOptions = tunefisOptions(Method="patternsearch", OptimizationType="tuning");
tuneOptions.MethodOptions.MaxIterations = double(efsConfig.pattern_max_iterations);
tuneOptions.UseParallel = logical(efsConfig.use_parallel);
fisParameters = tunefis(fisRules, [inputSettings; outputSettings], X, y, tuneOptions);

% Stage 3: freeze upper parameters and tune only interval uncertainty.
[uncertaintySettings, ~] = getTunableSettings(fisParameters, AsymmetricLag=true);
for inputIndex = 1:numel(uncertaintySettings)
    for mfIndex = 1:numel(uncertaintySettings(inputIndex).MembershipFunctions)
        uncertaintySettings(inputIndex).MembershipFunctions(mfIndex).UpperParameters.Free = false;
        uncertaintySettings(inputIndex).MembershipFunctions(mfIndex).LowerScale.Free = true;
        uncertaintySettings(inputIndex).MembershipFunctions(mfIndex).LowerLag.Free = true;
    end
end
model = tunefis(fisParameters, uncertaintySettings, X, y, tuneOptions);
model = adaptforecast.semanticizeOutputs(model);

trainingSummary = struct( ...
    schema_version=1, ...
    seed=double(config.seed), ...
    num_inputs=numel(model.Inputs), ...
    num_rules=numel(model.Rules), ...
    elapsed_seconds=toc(started), ...
    optimization="ga_then_patternsearch_then_type2_uncertainty");
end

