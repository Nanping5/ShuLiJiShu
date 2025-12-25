# 项目结构说明

## 📁 目录结构

```
Code/
├── config.py                    # 配置文件（所有超参数和路径）
├── train.py                     # 训练主脚本
├── evaluate.py                  # 评估主脚本
├── README.md                    # 项目说明文档
├── .gitignore                   # Git忽略文件
│
├── models/                      # 模型模块
│   ├── __init__.py             # 模块初始化
│   └── cnn_model.py            # CNN模型定义
│
├── data/                        # 数据模块
│   ├── __init__.py             # 模块初始化
│   ├── dataset.py              # 数据集类定义
│   ├── COMPILED_AIRFOIL_DATA.csv  # 原始数据（5.26 MB）
│   └── dataset.npz             # 预处理数据（2.88 MB）
│
├── utils/                       # 工具模块
│   ├── __init__.py             # 模块初始化
│   ├── logger.py               # 日志工具
│   ├── trainer.py              # 训练器类
│   └── visualizer.py           # 可视化工具
│
└── outputs/                     # 输出目录（自动创建）
    ├── models/                  # 模型文件
    │   ├── best_model.pth      # 最佳模型
    │   └── final_model.pth     # 最终模型
    ├── figures/                 # 图表
    │   ├── training_history.png
    │   ├── predictions.png
    │   └── error_analysis.png
    ├── logs/                    # 日志文件
    │   └── training.log
    └── summaries/               # 摘要报告
        ├── training_summary.txt
        └── evaluation_report.txt
```

## 📝 文件说明

### 核心脚本

| 文件 | 说明 | 用途 |
|------|------|------|
| `config.py` | 配置管理 | 所有超参数、路径配置 |
| `train.py` | 训练脚本 | 模型训练主流程 |
| `evaluate.py` | 评估脚本 | 模型评估和分析 |

### 模型模块 (models/)

| 文件 | 说明 |
|------|------|
| `cnn_model.py` | CNN模型定义，包含AirfoilCNN类 |

### 数据模块 (data/)

| 文件 | 说明 |
|------|------|
| `dataset.py` | 数据集类定义，包含AirfoilDataset类 |
| `COMPILED_AIRFOIL_DATA.csv` | 原始数据集（33,705样本） |
| `dataset.npz` | 预处理后的数据集（图像+标签） |

### 工具模块 (utils/)

| 文件 | 说明 |
|------|------|
| `logger.py` | 日志系统，统一的日志管理 |
| `trainer.py` | 训练器类，封装训练逻辑 |
| `visualizer.py` | 可视化工具，生成图表 |

## 🔧 配置说明

所有配置集中在 `config.py`：

### 数据配置 (DataConfig)
- `USE_SUBSET`: 是否使用子集
- `SUBSET_SIZE`: 子集大小
- `IMG_SIZE`: 图像尺寸
- `TRAIN_RATIO/VAL_RATIO/TEST_RATIO`: 数据分割比例

### 模型配置 (ModelConfig)
- `CONV_CHANNELS`: 卷积层通道数
- `FC_HIDDEN_DIMS`: 全连接层维度
- `DROPOUT_RATE`: Dropout比例

### 训练配置 (TrainConfig)
- `BATCH_SIZE`: 批大小
- `EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率
- `PATIENCE`: 早停耐心值

### 路径配置
- `DATA_DIR`: 数据目录
- `MODEL_DIR`: 模型保存目录
- `FIGURE_DIR`: 图表保存目录
- `LOG_DIR`: 日志保存目录

## 🚀 使用流程

### 1. 修改配置
```python
# 编辑 config.py
DataConfig.USE_SUBSET = True      # 快速测试用子集
TrainConfig.EPOCHS = 50           # 训练轮数
TrainConfig.LEARNING_RATE = 0.001 # 学习率
```

### 2. 训练模型
```bash
python train.py
```

### 3. 评估模型
```bash
python evaluate.py
```

### 4. 查看结果
- 日志: `outputs/logs/training.log`
- 模型: `outputs/models/best_model.pth`
- 图表: `outputs/figures/`
- 报告: `outputs/summaries/`

## 📊 输出说明

### 训练输出
- `best_model.pth`: 验证损失最低的模型
- `final_model.pth`: 最后一轮的模型
- `training_history.png`: 训练曲线
- `training_summary.txt`: 训练摘要

### 评估输出
- `predictions.png`: 预测vs实际散点图
- `error_analysis.png`: 误差分析图（4子图）
- `evaluation_report.txt`: 详细评估报告

## 🎯 代码特点

### 1. 模块化设计
- 清晰的职责分离
- 易于维护和扩展
- 代码复用性强

### 2. 配置管理
- 集中式配置
- 参数可调
- 路径统一

### 3. 日志系统
- 使用logging模块
- 文件+控制台输出
- 支持不同级别

### 4. 输出组织
- 按类型分文件夹
- 便于查找管理
- 自动创建目录

### 5. 面向对象
- Trainer类封装训练
- Dataset类封装数据
- 提高代码质量

## 🔍 扩展开发

### 添加新模型
1. 在 `models/` 创建新文件
2. 继承 `nn.Module`
3. 在 `__init__.py` 导出

### 修改数据处理
1. 编辑 `data/dataset.py`
2. 修改 `AirfoilDataset` 类

### 添加新可视化
1. 在 `utils/visualizer.py` 添加函数
2. 在 `evaluate.py` 调用

## 📌 注意事项

1. **数据文件**: `dataset.npz` 需要先运行数据预处理生成
2. **GPU使用**: 自动检测CUDA，无GPU则使用CPU
3. **日志文件**: 会追加写入，不会覆盖
4. **模型保存**: 自动保存最佳模型，避免过拟合

---

*项目重构完成于 2025-12-19*
