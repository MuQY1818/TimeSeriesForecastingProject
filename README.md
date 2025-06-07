# 特征工程的悖论：当简单模型在时间序列预测中超越复杂架构

本项目旨在探索一个在时间序列预测中非常有趣的现象：在进行了深度特征工程后，一个结构简单的神经网络（MLP）的性能，反而远超理论上更先进、更复杂的序列模型（LSTM+Attention）。

这个仓库包含了复现该研究的全部代码和数据。

## 项目核心发现

- **简单模型的巨大成功**：一个简单的MLP模型（`SimpleNN`）和一个经典的随机森林模型，在接收了包含滞后、滑动平均、周期性等信息的特征后，取得了极高的预测精度（R² > 0.94）。
- **复杂模型的意外失败**：为序列数据而生的`EnhancedNN`（LSTM+Attention）模型性能却最差（R² ≈ 0.18）。
- **"特征-模型不匹配"**：我们认为，根本原因在于深度特征工程已经将时序信息"显式化"，这对于MLP等模型是极佳的养料，但对于企图自行学习时序模式的LSTM网络而言，反而成为了干扰，导致其陷入优化困境。

### 与相关研究的讨论

本项目的发现与学界中"少即是多"（Less is More）的理念不谋而合。例如，Ukil等人在2022年的论文《When less is more powerful》中提出，通过使用Shapley值识别并**剔除贡献度低的训练样本**，可以显著提升时间序列分类模型的性能。

他们的工作聚焦于**训练样本的选择**，而本项目则从一个不同的、但同样关键的角度进行探索：**输入特征的选择及其与模型架构的适配性**。我们的核心贡献在于：

1.  **揭示了"特征-模型不匹配"现象**：我们证明了精心构建的特征对于某些模型（如MLP）是"蜜糖"，但对于另一些模型（如LSTM）可能成为"砒霜"。
2.  **验证了特征消融的价值**：通过特征消融实验，我们不仅解释了模型间的性能差异，还精确定位了导致`SimpleNN`成功的关键特征组。

因此，我们的研究可以视为对Ugil等人工作的补充和延伸：他们证明了"更少的样本"可以带来更强的模型，而我们证明了"更少的特征"（或者说，**更合适的特征**）同样是通往高性能的关键路径。

## 文件结构

```
.
├── model_comparison.py       # 主脚本：运行主要模型的性能对比分析
├── ablation_study.py         # 专用脚本：运行深入的、集中的特征消融实验
├── total_cleaned.csv         # 实验用的日销售数据
├── requirements.txt          # 项目运行所需的Python依赖
├── README.md                 # 就是您正在看的这个文件
└── .gitignore                # 指定Git忽略哪些文件
```

当您运行`model_comparison.py`后，会自动生成一个新的`models`文件夹，其中包含：
- `models/`
  - `*.joblib`: 保存的集成模型和数据缩放器。
  - `*.pth`: 保存的PyTorch神经网络模型。
  - `model_comparison.png`: 保存的模型性能对比图。
  - `model_predictions.csv`: 保存的各模型在测试集上的预测结果。
  - `main_training_log.csv`: `model_comparison.py`中神经网络的详细训练日志，记录了每个epoch的loss。

运行`ablation_study.py`后，会在`models/`文件夹中生成或更新：
- `focused_ablation_study.png`: 保存的精细化特征消融实验对比图。
- `ablation_training_log.csv`: `ablation_study.py`中所有神经网络的详细训练日志。

## 如何运行

1.  **克隆仓库**
    ```bash
    git clone https://github.com/MuQY1818/TimeSeriesForecastingProject.git
    cd TimeSeriesForecastingProject
    ```

2.  **创建虚拟环境并安装依赖**
    建议使用`conda`管理虚拟环境，以避免包版本冲突。
    ```bash
    # 创建名为 'ts-forecast' 的新环境 (会自动安装Python)
    conda create -n ts-forecast python=3.8 -y

    # 激活环境
    conda activate ts-forecast

    # 安装依赖
    pip install -r requirements.txt
    ```

3.  **运行主模型对比**
    ```bash
    python model_comparison.py
    ```
    这个脚本会训练和评估所有的基准模型（ARIMA, XGBoost, SimpleNN等），并生成它们的性能对比图和特征重要性图。

4.  **运行深入的消融实验 (可选)**
    在运行完主脚本后，您可以运行这个专用脚本来复现更精细的消融研究。
    ```bash
    python ablation_study.py
    ```

5.  **查看结果**
    脚本运行完毕后，所有的模型、图表、预测数据和详细的训练日志都会保存在自动创建的 `models/` 文件夹中。 