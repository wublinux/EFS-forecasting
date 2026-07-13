function runJob(jobPath)
%RUNJOB Execute a complete file-contract training and prediction job.

arguments
    jobPath (1,1) string
end

job = jsondecode(fileread(jobPath));
outputDir = string(job.output_dir);
if ~isfolder(outputDir)
    mkdir(outputDir);
end
trainTable = readtable(job.train_csv, VariableNamingRule="preserve");
testTable = readtable(job.test_csv, VariableNamingRule="preserve");
job.feature_columns = string(job.feature_columns);

[model, summary] = adaptforecast.train(trainTable, job);
[predictions, activations] = adaptforecast.predict(model, testTable, job.feature_columns);
rules = adaptforecast.explain(model, activations);

save(fullfile(outputDir, "model.mat"), "model", "summary", "job", "-v7.3");
writetable(predictions, fullfile(outputDir, "predictions.csv"));
writetable(activations, fullfile(outputDir, "activations.csv"));
writetable(rules, fullfile(outputDir, "rules.csv"));
summaryFile = fopen(fullfile(outputDir, "training_summary.json"), "w");
cleaner = onCleanup(@() fclose(summaryFile));
fprintf(summaryFile, "%s\n", jsonencode(summary, PrettyPrint=true));
end

