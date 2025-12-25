# 机翼气动系数预测 - CNN项目（重构版）

## 📁 项目结构

```
Code/
├── config.py                    # 配置文件（所有超参数）
├── train.py                     # 训练主脚本
├── evaluate.py                  # 评估主脚本
│
├── models/                      # 模型模块
│   ├── __init__.py
│   └── cnn_model.py            # CNN模型定义
│
├── data/                        # 数据模块
│   ├── __init__.py
│   ├── dataset.py              # 数据集类
│   ├── COMPILED_AIRFOIL_DATA.csv
│   └── dataset.npz
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── logger.py               # 日志工具
│   ├── trainer.py              # 训练器类
│   └── visualizer.py           # 可视化工具
│
└── outputs/                     # 输出目录
    ├── models/                  # 模型文件
    │   ├── best_model.pth
    │   └── final_model.pth
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

## 🚀 快速开始

### 1. 环境配置

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow scipy
```

### 2. 配置参数

编辑 `config.py` 修改超参数：

```python
# 数据配置
DataConfig.USE_SUBSET = True      # 是否使用子集
DataConfig.SUBSET_SIZE = 5000     # 子集大小
DataConfig.IMG_SIZE = 128         # 图像尺寸

# 训练配置
TrainConfig.BATCH_SIZE = 32       # 批大小
TrainConfig.EPOCHS = 50           # 训练轮数
TrainConfig.LEARNING_RATE = 0.001 # 学习率
TrainConfig.PATIENCE = 10         # 早停耐心值
```

### 3. 训练模型

```bash
python train.py
```

输出：
- `outputs/models/best_model.pth` - 最佳模型
- `outputs/figures/training_history.png` - 训练曲线
- `outputs/logs/training.log` - 训练日志
- `outputs/summaries/training_summary.txt` - 训练摘要

### 4. 评估模型

```bash
python evaluate.py
```

输出：
- `outputs/figures/predictions.png` - 预测结果
- `outputs/figures/error_analysis.png` - 误差分析
- `outputs/summaries/evaluation_report.txt` - 评估报告

## 📊 项目特点

### ✅ 代码重构亮点

1. **模块化设计**
   - 模型、数据、工具分离
   - 清晰的职责划分
   - 易于维护和扩展

2. **配置管理**
   - 集中式配置文件
   - 所有超参数可调
   - 路径统一管理

3. **日志系统**
   - 使用logging模块
   - 同时输出到控制台和文件
   - 支持不同日志级别

4. **输出组织**
   - 按类型分文件夹
   - 模型、图表、日志、摘要分离
   - 便于查找和管理

5. **面向对象**
   - Trainer类封装训练逻辑
   - Dataset类封装数据处理
   - 代码复用性强

## 🎯 核心模块说明

### config.py - 配置管理
```python
# 数据配置
DataConfig.USE_SUBSET = True
DataConfig.IMG_SIZE = 128

# 模型配置
ModelConfig.CONV_CHANNELS = [32, 64, 128]
ModelConfig.FC_HIDDEN_DIMS = [512, 256]

# 训练配置
TrainConfig.BATCH_SIZE = 32
TrainConfig.LEARNING_RATE = 0.001
```

### models/cnn_model.py - CNN模型
```python
from models import create_model

model = create_model(config=ModelConfig(), device='cuda')
```

### utils/logger.py - 日志工具
```python
from utils import get_logger

logger = get_logger("my_module")
logger.info("这是一条日志")
```

### utils/trainer.py - 训练器
```python
from utils.trainer import Trainer

trainer = Trainer(model, criterion, optimizer, device, logger)
history = trainer.train(train_loader, val_loader, epochs=50)
```

## 📈 性能指标

当前模型（5000样本子集）：
- R² = 0.11
- MAE = 0.0091
- MAPE = 31.07%
- 相对误差<10%: 20.4%

改进建议：
1. 使用全部33,705样本（设置 `USE_SUBSET=False`）
2. 增加训练轮数（`EPOCHS=100`）
3. 调整学习率（`LEARNING_RATE=0.0005`）

## 🔧 自定义开发

### 添加新模型

1. 在 `models/` 创建新文件
2. 继承 `nn.Module`
3. 在 `__init__.py` 导出

### 添加新的可视化

1. 在 `utils/visualizer.py` 添加函数
2. 在 `evaluate.py` 调用

### 修改数据处理

1. 编辑 `data/dataset.py`
2. 修改 `AirfoilDataset` 类

## 📝 日志查看

```bash
# 查看训练日志
cat outputs/logs/training.log

# 查看训练摘要
cat outputs/summaries/training_summary.txt

# 查看评估报告
cat outputs/summaries/evaluation_report.txt
```

## 🎓 学术使用

本项目实现了论文中的CNN方法：
- 输入：翼型图像 + 攻角(AoA)
- 输出：阻力系数(Cd)
- 架构：3层卷积 + 3层全连接
- 数据：2946个翼型，33705个样本

## 📧 问题反馈

如有问题，请检查：
1. `outputs/logs/training.log` - 训练日志
2. `config.py` - 配置是否正确
3. `data/dataset.npz` - 数据是否存在

---

*重构完成于 2025-12-19*
