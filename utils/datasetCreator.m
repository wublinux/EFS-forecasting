function [trnX, trnY, vldX, vldY] = createDataset(salesDataNorm, weatherData, timeLag, trainRatio)
%CREATEDATASET Create input-output dataset for training and validation
%   Creates time-lagged input features from sales and weather data

    numSamples = length(salesDataNorm);
    
    % Initialize input/output matrices
    % Input: [timeLag sales values + temperature + humidity]
    inputData = zeros(numSamples - timeLag, timeLag + 2); 
    outputData = zeros(numSamples - timeLag, 1);    
    
    % Create training samples using time-lag approach
    for t = timeLag:numSamples-1
        inputData(t-timeLag+1, 1:timeLag) = salesDataNorm(t-timeLag+1:t); 
        inputData(t-timeLag+1, timeLag+1) = weatherData.temperature(t); 
        inputData(t-timeLag+1, timeLag+2) = weatherData.humidity(t);    
        outputData(t-timeLag+1) = salesDataNorm(t+1); 
    end
    
    % Split dataset into training and validation sets
    trainSize = round(trainRatio * size(inputData, 1));
    
    trnX = inputData(1:trainSize, :);    
    trnY = outputData(1:trainSize, :);   
    vldX = inputData(trainSize+1:end, :);
    vldY = outputData(trainSize+1:end, :);
end
