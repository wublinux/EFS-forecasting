classdef TestFileContract < matlab.unittest.TestCase
    methods (Test)
        function smokeJobWritesVersionedArtifacts(testCase)
            root = string(tempname);
            mkdir(root);
            cleanup = onCleanup(@() rmdir(root, "s")); %#ok<NASGU>
            dates = datetime(2016, 3, (1:12))';
            lag1 = linspace(.1, .8, 12)';
            target = min(1, lag1 + .05);
            trainData = table(dates, lag1, target, ...
                VariableNames=["date", "sales_lag_1", "target_norm"]);
            testData = trainData(9:12, :);
            trainPath = fullfile(root, "train.csv");
            testPath = fullfile(root, "test.csv");
            writetable(trainData, trainPath);
            writetable(testData, testPath);

            job = struct;
            job.schema_version = 1;
            job.train_csv = trainPath;
            job.test_csv = testPath;
            job.output_dir = fullfile(root, "output");
            job.feature_columns = "sales_lag_1";
            job.target_column = "target_norm";
            job.seed = 42;
            job.efs = struct(num_input_mfs=2, population_size=10, ...
                max_generations=1, crossover_fraction=.8, ...
                pattern_max_iterations=1, use_parallel=false);
            jobPath = fullfile(root, "job.json");
            file = fopen(jobPath, "w");
            fprintf(file, "%s", jsonencode(job));
            fclose(file);

            adaptforecast.runJob(jobPath);
            testCase.verifyTrue(isfile(fullfile(job.output_dir, "model.mat")));
            testCase.verifyTrue(isfile(fullfile(job.output_dir, "predictions.csv")));
            testCase.verifyTrue(isfile(fullfile(job.output_dir, "rules.csv")));
            testCase.verifyTrue(isfile(fullfile(job.output_dir, "activations.csv")));
            predictions = readtable(fullfile(job.output_dir, "predictions.csv"));
            testCase.verifyEqual(height(predictions), height(testData));

            predictJob = struct;
            predictJob.schema_version = 1;
            predictJob.model_file = fullfile(job.output_dir, "model.mat");
            predictJob.input_csv = testPath;
            predictJob.output_csv = fullfile(root, "saved-model-predictions.csv");
            predictJob.feature_columns = "sales_lag_1";
            predictJobPath = fullfile(root, "predict-job.json");
            file = fopen(predictJobPath, "w");
            fprintf(file, "%s", jsonencode(predictJob));
            fclose(file);
            adaptforecast.predictJob(predictJobPath);
            savedModelPredictions = readtable(predictJob.output_csv);
            testCase.verifyEqual(savedModelPredictions.prediction_norm, ...
                predictions.prediction_norm, AbsTol=1e-12);

            trained = load(fullfile(job.output_dir, "model.mat"), "summary");
            testCase.verifyLessThanOrEqual(trained.summary.num_rules, 2);
            testCase.verifyTrue(trained.summary.upper_parameters_locked);
            testCase.verifyEqual(trained.summary.upper_parameter_max_delta, 0, AbsTol=1e-12);

            firstPredictions = predictions.prediction_norm;
            job.output_dir = fullfile(root, "repeat-output");
            file = fopen(jobPath, "w");
            fprintf(file, "%s", jsonencode(job));
            fclose(file);
            adaptforecast.runJob(jobPath);
            repeated = readtable(fullfile(job.output_dir, "predictions.csv"));
            testCase.verifyEqual(repeated.prediction_norm, firstPredictions, AbsTol=1e-12);
        end
    end
end
