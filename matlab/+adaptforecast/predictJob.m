function predictJob(jobPath)
%PREDICTJOB Predict from a saved model through the public file contract.

arguments
    jobPath (1,1) string
end

job = jsondecode(fileread(jobPath));
saved = load(job.model_file, "model");
inputTable = readtable(job.input_csv, VariableNamingRule="preserve");
[predictions, ~] = adaptforecast.predict(saved.model, inputTable, string(job.feature_columns));
writetable(predictions, job.output_csv);
end
