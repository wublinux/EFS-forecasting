%% Sales Prediction using Evolving Fuzzy System (EFS)
% Main script for sales prediction using fuzzy inference system
% 
% This script implements a three-stage learning approach:
% 1. Rule learning using genetic algorithm
% 2. Parameter tuning
% 3. Advanced asymmetric lag tuning
%
% Usage: Run this script to train and evaluate the FIS model

clear; clc; close all;

%% Configuration Parameters
config = struct();
config.dataFile = 'Final data ver3.csv';
config.randomSeed = 42;
config.timeLag = 3;
config.trainRatio = 0.5;
config.numInputMFs = 2;
config.gaPopulationSize = 100;
config.gaMaxGenerations = 300;
config.gaCrossoverFraction = 0.8;
config.enableTuning = true;

%% Stage 1: Data Loading and Preprocessing
fprintf('=== Stage 1: Loading and Preprocessing Data ===\n');
[salesDataNorm, weatherData] = loadAndPreprocessData(config.dataFile, config.randomSeed);

%% Stage 2: Generate Training Dataset
fprintf('=== Stage 2: Generating Training Dataset ===\n');
[trnX, trnY, vldX, vldY] = createDataset(salesDataNorm, weatherData, config.timeLag, config.trainRatio);

%% Stage 3: Initialize FIS Structure
fprintf('=== Stage 3: Initializing FIS Structure ===\n');
fisin = initializeFIS(config.timeLag, config.numInputMFs);

%% Stage 4: Three-Stage Learning Process
fprintf('=== Stage 4: Starting Three-Stage Learning Process ===\n');

if config.enableTuning
    % Stage 4.1: Rule Learning
    fprintf('--- Stage 4.1: Rule Learning ---\n');
    fisout1 = learnRules(fisin, trnX, trnY, config);
    
    % Stage 4.2: Parameter Tuning
    fprintf('--- Stage 4.2: Parameter Tuning ---\n');
    fisout2 = tuneParameters(fisout1, trnX, trnY, config);
    
    % Stage 4.3: Advanced Tuning with Asymmetric Lag
    fprintf('--- Stage 4.3: Advanced Tuning ---\n');
    fisout3 = advancedTuning(fisout2, trnX, trnY, config);
    
    % Finalize FIS
    fisout3.Rules = [fisout3.Rules, fisin.Rules];
    fisout3 = autoNameOutputMFs(fisout3);
else
    fisout3 = fisin;
end

%% Stage 5: Model Evaluation and Analysis
fprintf('=== Stage 5: Model Evaluation ===\n');
evaluateModel(fisout3, vldX, vldY);

fprintf('\n=== Training Complete ===\n');
