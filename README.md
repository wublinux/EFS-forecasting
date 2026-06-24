# Sales Prediction using Evolving Fuzzy System (EFS)

This is a final year project implementing an Evolving Fuzzy System (EFS) for sales prediction.

## Overview

The project uses a three-stage learning approach with Sugeno-type Fuzzy Inference System (FIS):

1. **Rule Learning**: Uses genetic algorithm to learn optimal fuzzy rules
2. **Parameter Tuning**: Optimizes membership function parameters
3. **Advanced Tuning**: Fine-tunes asymmetric lag parameters

## 項目結構

```
/workspace
├── quick_start_template.m    # 【推薦】即用模板，修改數據路徑即可使用
├── batch_predict.m           # 批量預測工具（需先有訓練好的模型）
├── main.m                    # 原始主程式（進階用戶使用）
├── utils/
│   ├── dataLoader.m          # 數據載入和預處理
│   └── datasetCreator.m      # 訓練數據集創建
├── fis/
│   ├── initializeFIS.m       # FIS 結構初始化
│   ├── learnRules.m          # 階段 1：規則學習
│   ├── tuneParameters.m      # 階段 2：參數調優
│   ├── advancedTuning.m      # 階段 3：高級調優
│   ├── printFuzzyRules.m     # 規則可視化
│   └── autoNameOutputMFs.m   # 輸出歸屬函數命名
├── evaluation/
│   └── evaluateModel.m       # 模型評估和指標
├── Final data ver3.csv       # 示例數據集
└── README.md                 # 本文件
```

## Requirements

- MATLAB with Fuzzy Logic Toolbox
- Parallel Computing Toolbox (optional, for faster training)

## 快速開始 (Quick Start)

### 方法一：即用模板 (推薦)

最簡單的使用方式，只需修改一個參數即可開始：

```matlab
% 1. 複製 quick_start_template.m
% 2. 修改第 15 行的數據檔案路徑
dataFile = '您的數據檔案.csv';  % <-- 只需修改這裡
% 3. 運行腳本
quick_start_template
```

### 方法二：批量預測

如果您已有訓練好的模型，可以使用批量預測工具：

```matlab
% 1. 先運行 quick_start_template.m 訓練並保存模型
% 2. 修改 batch_predict.m 中的模型檔案路徑
modelFile = 'trained_model_yyyyMMdd_HHMMSS.mat';
% 3. 運行批量預測
batch_predict
```

## 使用範例

### 完整工作流程

```matlab
% 步驟 1: 訓練模型 (約 2-5 分鐘)
run quick_start_template

% 步驟 2: 按照提示保存模型 (.mat 檔案)

% 步驟 3: 對新數據進行預測
run batch_predict
```

### 自定義配置

在 `quick_start_template.m` 中調整以下參數：

```matlab
trainRatio = 0.7;        % 訓練數據比例 (70%)
timeLag = 3;             % 用過去 3 天預測明天
enableFullTraining = true;  % true=完整訓練，false=快速模式
```

## 數據格式要求

CSV 檔案應包含以下結構：
- 第 1 列：日期（可選）
- 第 2 列：第一段時間序列數據（如 2016 年銷售數據）
- 第 3 列：第二段時間序列數據（如 2017 年銷售數據）
- 其他列：其他特徵（當前版本未使用）

系統會自動將兩段時間序列連接成連續數據進行訓練。

## 輸出說明

模型會輸出：
- **RMSE** (均方根誤差) - 預測誤差的標準差
- **MAE** (平均絕對誤差) - 預測誤差的平均值
- **可視化圖表** - 預測值 vs 實際值對比圖
- **規則激活熱力圖** - 顯示各規則的激活強度
- **模糊規則列表** - 人類可讀的模糊規則
- **預測結果 CSV** (批量預測時) - 包含預測值和統計信息

## 常見問題

### Q1: 如何用自己的數據？
只需準備一個 CSV 檔案，確保第 2 和第 3 列包含您的時間序列數據，然後在 `quick_start_template.m` 中修改 `dataFile` 變數。

### Q2: 訓練需要多長時間？
- **快速模式** (`enableFullTraining = false`)：幾秒鐘
- **完整訓練** (`enableFullTraining = true`)：2-5 分鐘（取決於數據量和計算資源）

### Q3: 如何保存和載入模型？
- **保存**：運行 `quick_start_template.m` 後按提示輸入 'y' 保存模型
- **載入**：在 `batch_predict.m` 中指定模型檔案路徑即可

### Q4: 如何提高預測準確度？
- 增加訓練數據量
- 調整 `trainRatio` 使用更多數據訓練
- 啟用完整訓練模式
- 調整 `timeLag` 參數嘗試不同的時間窗口

## License

本項目供教育用途。
