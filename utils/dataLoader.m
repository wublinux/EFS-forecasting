function [salesDataNorm, weatherData, originalData] = loadAndPreprocessData(dataFile, randomSeed)
%LOADANDPREPROCESSDATA Load CSV data and preprocess for training
%   Loads sales data from CSV, handles missing values, normalizes data,
%   and generates synthetic weather data.
%
%   Returns:
%   - salesDataNorm: Normalized sales time series
%   - weatherData: Structure with normalized weather parameters
%   - originalData: Original data table (for reference)

    % Read CSV dataset
    data = readtable(dataFile);
    originalData = data;  % Store original data for reference
    
    % Extract sales columns (2016 and 2017)
    sales2016 = data{:, 2}';  
    sales2017 = data{:, 3}';
    
    % Combine into single time series
    salesData = [sales2016, sales2017]; 
    
    % Handle missing values (fill with mean)
    salesData(isnan(salesData)) = mean(salesData, 'omitnan');
    
    % Normalize data to [0,1] range
    minVal = min(salesData);
    maxVal = max(salesData);
    salesDataNorm = (salesData - minVal) / (maxVal - minVal);
    
    % Generate synthetic weather data
    rng(randomSeed);
    numDays = length(salesDataNorm);
    
    % Generate weather parameters
    temperature = 15 + 10 * randn(1, numDays); 
    rainfall = max(0, 5 * randn(1, numDays));  
    humidity = 50 + 20 * randn(1, numDays);    
    
    % Normalize weather data
    weatherData.temperature = normalizeData(temperature);
    weatherData.rainfall = normalizeData(rainfall);
    weatherData.humidity = normalizeData(humidity);
end

function normalizedData = normalizeData(data)
%NORMALIZEDATA Normalize data to [0,1] range
    minVal = min(data);
    maxVal = max(data);
    normalizedData = (data - minVal) / (maxVal - minVal);
end
