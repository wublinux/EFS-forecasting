%% 批量預測工具 - Batch Prediction Tool
% 用於對新數據進行批量預測
%
% 使用方法:
% 1. 先運行 quick_start_template.m 訓練並保存模型
% 2. 修改下方的模型檔案路徑和新數據檔案路徑
% 3. 運行此腳本進行批量預測

clear; clc; close all;

%% ==================== 配置區 ====================

% 已訓練模型的檔案路徑
modelFile = 'trained_model_20250622_120000.mat';  % <-- 修改為您的模型檔案

% 新數據檔案 (要預測的數據)
newDataFile = 'Final data ver3.csv';  % <-- 修改為新數據檔案

% 輸出結果保存位置
outputFile = 'prediction_results.csv';  % <-- 修改為輸出檔案名

% ================================================

fprintf('========================================\n');
fprintf('  批量預測工具\n');
fprintf('========================================\n\n');

%% 步驟 1: 載入已訓練的模型
fprintf('📥 正在載入模型...\n');
if exist(modelFile, 'file')
    load(modelFile);
    fprintf('   ✓ 模型載入成功\n');
    fprintf('   - 時間滯後：%d\n', timeLag);
    fprintf('   - 輸入歸屬函數數量：%d\n\n', numInputMFs);
else
    error('❌ 錯誤：找不到模型檔案 %s\n請先運行 quick_start_template.m 訓練並保存模型', modelFile);
end

%% 步驟 2: 載入新數據
fprintf('📥 正在載入新數據...\n');
data = readtable(newDataFile);
fprintf('   ✓ 數據載入完成 (%d 行)\n', height(data));

%% 步驟 3: 數據預處理 (與訓練時相同)
[salesDataNorm, weatherData] = loadAndPreprocessData(newDataFile, 42);

%% 步驟 4: 創建預測輸入
fprintf('🔧 正在準備預測數據...\n');
numSamples = length(salesDataNorm);
predictInputs = zeros(numSamples - timeLag, timeLag + 2);

for t = timeLag:numSamples-1
    predictInputs(t-timeLag+1, 1:timeLag) = salesDataNorm(t-timeLag+1:t);
    predictInputs(t-timeLag+1, timeLag+1) = weatherData.temperature(t);
    predictInputs(t-timeLag+1, timeLag+2) = weatherData.humidity(t);
end

fprintf('   ✓ 準備好 %d 個預測樣本\n\n', size(predictInputs, 1));

%% 步驟 5: 執行預測
fprintf('🚀 正在進行預測...\n');
predictions = evalfis(trainedFIS, predictInputs);
fprintf('   ✓ 預測完成\n\n');

%% 步驟 6: 保存結果
fprintf('💾 正在保存預測結果...\n');

% 創建結果表格
resultsTable = table();
resultsTable.SampleIndex = (1:length(predictions))';
resultsTable.PredictedValue = predictions;

% 如果有原始數據，加入反標準化的值
if exist('originalData', 'var')
    % 計算反標準化參數
    sales2016 = data{:, 2}';
    sales2017 = data{:, 3}';
    salesData = [sales2016, sales2017];
    minVal = min(salesData);
    maxVal = max(salesData);
    
    % 反標準化預測值
    resultsTable.PredictedOriginalScale = predictions * (maxVal - minVal) + minVal;
end

% 保存為 CSV
writetable(resultsTable, outputFile);
fprintf('   ✓ 結果已保存到：%s\n\n', outputFile);

%% 步驟 7: 顯示統計信息
fprintf('📊 預測統計:\n');
fprintf('   最小值：%.4f\n', min(predictions));
fprintf('   最大值：%.4f\n', max(predictions));
fprintf('   平均值：%.4f\n', mean(predictions));
fprintf('   標準差：%.4f\n\n', std(predictions));

%% 步驟 8: 可視化
figure('Name', '預測結果', 'Position', [100, 100, 1000, 600]);
subplot(2,1,1);
plot(predictions, 'b-', 'LineWidth', 1.5);
hold on;
yline(mean(predictions), 'r--', '平均線');
xlabel('樣本索引');
ylabel('預測值 (標準化)');
title('銷售預測趨勢');
grid on;

subplot(2,1,2);
histogram(predictions, 20, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
xlabel('預測值');
ylabel('頻次');
title('預測值分佈');
grid on;

fprintf('✅ 所有工作已完成!\n');
fprintf('========================================\n');
