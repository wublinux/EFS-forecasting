%% Sales Prediction - Quick Start Template
% 即用的 MaaS (Model-as-a-Service) 模板
% 
% 使用方法:
% 1. 準備您的數據 CSV 檔案 (格式：日期，銷售數據 1, 銷售數據 2, ...)
% 2. 修改第 15 行的數據檔案路徑
% 3. 運行此腳本即可開始預測
%
% 輸出:
% - 訓練好的 FIS 模型
% - 預測結果 (RMSE, MAE)
% - 可視化圖表
% - 可保存模型供後續使用

clear; clc; close all;

%% ==================== 用戶配置區 ====================
% 只需修改以下參數即可開始使用

% 1. 數據檔案路徑 (請修改為您的 CSV 檔案路徑)
dataFile = 'Final data ver3.csv';  % <-- 修改這裡

% 2. 預測目標列 (CSV 中的第幾列是銷售數據)
targetColumns = [2, 3];  % 預設使用第 2 和第 3 列作為連續時間序列

% 3. 訓練比例 (多少數據用於訓練)
trainRatio = 0.7;  % 70% 訓練，30% 驗證

% 4. 時間滯後 (用過去幾天的數據預測明天)
timeLag = 3;  % 用過去 3 天的數據

% 5. 是否啟用完整訓練 (包含三個階段)
enableFullTraining = true;  % true=完整訓練，false=快速模式

% ===================================================

fprintf('========================================\n');
fprintf('  銷售預測系統 - 即用模板\n');
fprintf('========================================\n\n');

%% 步驟 1: 載入數據
fprintf('📥 正在載入數據...\n');
[salesDataNorm, weatherData, originalData] = loadAndPreprocessData(dataFile, 42);
fprintf('   ✓ 數據載入完成 (%d 個樣本)\n', length(salesDataNorm));

%% 步驟 2: 創建數據集
fprintf('📊 正在創建訓練數據集...\n');
[trnX, trnY, vldX, vldY] = createDataset(salesDataNorm, weatherData, timeLag, trainRatio);
fprintf('   ✓ 訓練集：%d 樣本, 驗證集：%d 樣本\n', size(trnX, 1), size(vldX, 1));

%% 步驟 3: 初始化 FIS
fprintf('🔧 正在初始化模糊推理系統...\n');
numInputMFs = 2;  % 每個輸入的歸屬函數數量
fisin = initializeFIS(timeLag, numInputMFs);
fprintf('   ✓ FIS 初始化完成 (%d 個輸入，%d 個規則)\n', timeLag+2, numInputMFs^(timeLag+2));

%% 步驟 4: 訓練模型
if enableFullTraining
    fprintf('🚀 開始三階段訓練 (可能需要幾分鐘)...\n\n');
    
    % 配置參數
    config = struct();
    config.timeLag = timeLag;
    config.numInputMFs = numInputMFs;
    config.gaPopulationSize = 50;
    config.gaMaxGenerations = 100;
    config.gaCrossoverFraction = 0.8;
    config.enableTuning = true;
    config.trainRatio = trainRatio;
    
    % 階段 1: 規則學習
    fprintf('   [1/3] 規則學習中...\n');
    fisout1 = learnRules(fisin, trnX, trnY, config);
    
    % 階段 2: 參數調優
    fprintf('   [2/3] 參數調優中...\n');
    fisout2 = tuneParameters(fisout1, trnX, trnY, config);
    
    % 階段 3: 高級調優
    fprintf('   [3/3] 高級調優中...\n');
    fisout3 = advancedTuning(fisout2, trnX, trnY, config);
    
    % 完成 FIS
    fisout3.Rules = [fisout3.Rules, fisin.Rules];
    fisout3 = autoNameOutputMFs(fisout3);
    
    trainedFIS = fisout3;
    fprintf('\n   ✓ 訓練完成!\n\n');
else
    fprintf('⚡ 快速模式 - 跳過訓練階段\n');
    trainedFIS = fisin;
end

%% 步驟 5: 評估與預測
fprintf('📈 正在評估模型性能...\n');
fprintf('----------------------------------------\n');
evaluateModel(trainedFIS, vldX, vldY);
fprintf('----------------------------------------\n\n');

%% 步驟 6: 保存模型 (可選)
saveModel = input('是否保存訓練好的模型？(y/n): ', 's');
if lower(saveModel) == 'y'
    modelFilename = sprintf('trained_model_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
    save(modelFilename, 'trainedFIS', 'config', 'timeLag', 'numInputMFs');
    fprintf('✓ 模型已保存到：%s\n\n', modelFilename);
end

%% 步驟 7: 進行新預測 (示例)
fprintf('💡 使用範例:\n');
fprintf('   要使用訓練好的模型進行新預測，請使用:\n');
fprintf('   predictedValue = evalfis(trainedFIS, newInputData);\n\n');

% 示例：用最後一筆驗證數據進行預測
if size(vldX, 1) > 0
    sampleInput = vldX(1, :);
    prediction = evalfis(trainedFIS, sampleInput);
    fprintf('   示例預測結果：%.4f\n', prediction);
end

fprintf('\n========================================\n');
fprintf('  預測完成!\n');
fprintf('========================================\n');
