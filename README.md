# 特征工程的悖论：当简单模型在时间序列预测中超越复杂架构

本项目旨在探索一个在时间序列预测中非常有趣的现象：在进行了深度特征工程后，一个结构简单的神经网络（MLP）的性能，反而远超理论上更先进、更复杂的序列模型（LSTM+Attention）。

这个仓库包含了复现该研究的全部代码和数据。

## 项目核心发现

- **简单模型的巨大成功**：一个简单的MLP模型（`SimpleNN`）和一个经典的随机森林模型，在接收了包含滞后、滑动平均、周期性等信息的特征后，取得了极高的预测精度（R² > 0.94）。
- **复杂模型的意外失败**：为序列数据而生的`EnhancedNN`（LSTM+Attention）模型性能却最差（R² ≈ 0.18）。
- **"特征-模型不匹配"**：我们认为，根本原因在于深度特征工程已经将时序信息"显式化"，这对于MLP等模型是极佳的养料，但对于企图自行学习时序模式的LSTM网络而言，反而成为了干扰，导致其陷入优化困境。

## 文件结构

```
.
├── model_comparison.py       # 主脚本：包含数据处理、模型训练与评估的所有逻辑
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


## 如何运行

1.  **克隆仓库**
    ```bash
    git clone <your-repository-url>
    cd TimeSeriesForecastingProject
    ```

2.  **创建虚拟环境并安装依赖**
    建议使用虚拟环境以避免包版本冲突。
    ```bash
    python -m venv venv
    # 激活虚拟环境 (Windows)
    .\venv\Scripts\activate
    # 激活虚拟环境 (macOS/Linux)
    # source venv/bin/activate

    # 安装依赖
    pip install -r requirements.txt
    ```

3.  **运行脚本**
    ```bash
    python model_comparison.py
    ```

4.  **查看结果**
    脚本运行完毕后，所有的模型、图表和预测数据都会保存在自动创建的 `models/` 文件夹中。 