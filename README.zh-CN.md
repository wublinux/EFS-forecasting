# EFS-forecasting / AdaptForecast

AdaptForecast 是一个面向**小样本、品类级、下一日需求预测**的可审计研究系统。MATLAB
参考核心负责 Interval Type-2 Sugeno 模糊推理与进化优化；Python 负责数据协议、无泄漏
预处理、基线模型、实验编排、命令行和 Streamlit，并提供一个明确独立标识的
`python-it2` 后端，供没有 MATLAB 的环境使用。

项目源于“Evolutionary Fuzzy System for Supply Chain and Operations Management”毕设报告，
但本仓库不会把报告中的构想自动当作已证实能力。

> **证据状态：**软件结构、合成数据和 smoke workflow 可以公开测试；私有论文数据的结果
> 必须在本地重新生成后才能引用。在线演化、台风事件预警、因果天气影响和六种时间粒度
> 决策支持尚未实现。

## 研究协议

- 输入为目标日前 3 天销量，以及目标日已知或可预报的平均温度、湿度和降雨。
- 2016 年用于训练，2017 年作为完全独立测试集。
- 缺失值兜底、中位数和 Min-Max 范围只从 2016 年拟合，避免测试信息泄漏。
- 比较周季节朴素法、ARIMA、32 单元单层 LSTM、无天气 EFS 和目标日天气 EFS。
- 输出原始尺度 RMSE、MAE、MASE、sMAPE，同时保留归一化 RMSE/MAE 用于论文对照。
- 完整实验中 EFS 和 LSTM 使用 `42/43/44` 三个随机种子。

EFS 的三个阶段为：GA 学习规则，Pattern Search 调整隶属函数和输出参数，最后在锁定
上界参数后调整 Type-2 `LowerScale/LowerLag`。

## 环境要求

- Python 3.10+
- MATLAB R2024b+
- Fuzzy Logic Toolbox、Global Optimization Toolbox
- Parallel Computing Toolbox 可选，完整训练建议安装

没有 MATLAB 时仍可校验数据、运行 Python 基线、浏览已有结果，并训练独立的 Python
IT2 后端。MATLAB 与 Python IT2 共享实验协议，但不假定数值等价；新生成的每行指标都会
记录后端身份。
仓库内置一个明确标注为 smoke 的合成结果快照，因此干净检出后无需 MATLAB 也能查看
预测曲线、规则表和真实激活热图。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,benchmark,app]"

adaptforecast validate-data data/sample/synthetic_demand.csv
adaptforecast benchmark --config configs/benchmark.smoke.yaml
adaptforecast app
```

`smoke` 配置只验证流程，不能作为论文结果。三随机种子的完整审计使用
`configs/benchmark.yaml`。

没有 MATLAB 时可运行明确标注的 Python IT2 smoke 流程：

```bash
adaptforecast benchmark --config configs/benchmark.python-it2.smoke.yaml
```

## 私有数据

仓库中的 `data/sample/synthetic_demand.csv` 是脚本生成的纯合成数据，不来自 Walmart
China 数据。原始数据公开权尚未确认，因此必须留在仓库外：

```powershell
$env:ADAPTFORECAST_DATA_DIR = "C:\path\to\private-data"
```

旧 27 列宽表可转换为统一长表：

```bash
adaptforecast prepare private-wide.csv canonical-private.csv --legacy-wide
```

转换后可直接运行仓库外的完整私有审计，数据和结果都不会写入公开仓库：

```powershell
$env:ADAPTFORECAST_DATA_DIR = "C:\path\to\private-data"
adaptforecast benchmark --config configs/benchmark.private.example.yaml
```

如需使用独立 Python 实现完成八品类三种子实验，使用
`configs/benchmark.private.python-it2.example.yaml`。不得把其 EFS 指标与 MATLAB EFS
指标混称为同一实现的结果。

统一字段为 `date, category, sales, avg_temp, max_temp, min_temp, humidity, rainfall`。
MIT 许可证只覆盖代码，不覆盖任何私有或第三方数据。

## 对外接口

```text
adaptforecast validate-data INPUT.csv
adaptforecast prepare INPUT.csv OUTPUT.csv
adaptforecast benchmark --config configs/benchmark.yaml
adaptforecast predict --model model.mat|model.npz --input prepared.csv --features sales_lag_1,...
adaptforecast app
```

MATLAB 公开 `adaptforecast.train/predict/evaluate/explain`。Python 使用版本化 JSON/CSV
契约和 `matlab -batch` 调用，不依赖 MATLAB Engine。独立 Python IT2 后端沿用相同文件
契约，并导出经过哈希校验的 `model.npz` 和可读的 `model.json` 元数据。

安装 MATLAB R2024b+ 后，干净检出可直接使用合成 sales-only 模型验证预测契约：

```bash
adaptforecast predict \
  --model data/precomputed/synthetic-smoke/efs/ice_cream/sales_only/42/model.mat \
  --input data/precomputed/synthetic-smoke/efs/ice_cream/sales_only/42/inputs/test.csv \
  --features sales_lag_1,sales_lag_2,sales_lag_3 \
  --output predictions-example.csv
```

每次 benchmark 都生成独立 artifact 目录，包括配置、数据 SHA-256、后端身份、指标、
预测、模型、规则表和激活矩阵。

## 负责任地解释结果

- 所有模型使用相同年份切分和公共评价窗口。
- 模糊规则及天气关系只代表统计关联，不代表因果关系。
- 没有对应 manifest、数据哈希和运行产物的结果不得写成已复现结论。
- 论文中的在线演化、3–5 天预警和多粒度决策支持保留为未来工作。

详细内容见 [方法](docs/methodology.md)、[复现指南](docs/reproducibility.md)、
[论文对齐](docs/thesis-alignment.md)和[局限](docs/limitations.md)。
