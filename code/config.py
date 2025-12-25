"""
配置文件 - 集中管理所有超参数和路径
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "COMPILED_AIRFOIL_DATA.csv"
PROCESSED_DATA_PATH = DATA_DIR / "dataset.npz"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
SUMMARY_DIR = OUTPUT_DIR / "summaries"

# 创建输出目录
for dir_path in [OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, LOG_DIR, SUMMARY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 数据预处理配置 ====================
class DataConfig:
    """数据预处理配置"""
    # 是否使用数据子集（快速测试）
    USE_SUBSET = False  # 使用全部数据
    SUBSET_SIZE = 5000  # 仅在USE_SUBSET=True时生效
    
    # 图像参数
    IMG_SIZE = 128
    LINE_WIDTH = 2
    
    # CST参数
    CST_N_POINTS = 200
    TE_THICKNESS = 0.0
    
    # 数据分割比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 数据归一化 - 关闭标签归一化！这是之前效果差的主因
    NORMALIZE_LABELS = False  # 不归一化标签，直接预测原始Cd值
    NORMALIZE_PARAMS = True   # AoA参数归一化（有帮助）
    
    # 数据增强 - 轻度增强
    USE_AUGMENTATION = False  # 先关闭，确保基础模型能工作
    AUG_ROTATION = 3  # 旋转角度范围（度）
    AUG_SCALE = 0.05  # 缩放范围
    AUG_NOISE = 0.005  # 噪声强度


# ==================== 模型配置 ====================
class ModelConfig:
    """模型架构配置"""
    # 输入参数
    IMG_SIZE = 128
    INPUT_CHANNELS = 1
    
    # 卷积层配置 - 简化模型，减少过拟合风险
    CONV_CHANNELS = [32, 64, 128]  # 3层卷积足够
    KERNEL_SIZE = 3
    PADDING = 1
    POOL_SIZE = 2
    USE_BATCH_NORM = True  # 批归一化
    
    # 全连接层配置 - 简化
    FC_HIDDEN_DIMS = [256, 64]  # 简化全连接层
    DROPOUT_RATE = 0.2  # 降低Dropout
    USE_LAYER_NORM = False  # 层归一化（可选）
    
    # AoA融合方式: 'concat' 或 'film'
    AOA_FUSION = 'film'  # 使用FiLM方式融合AoA
    
    # 输出维度
    OUTPUT_DIM = 1  # Cd预测


# ==================== 训练配置 ====================
class TrainConfig:
    """训练超参数配置"""
    # 基本参数
    BATCH_SIZE = 64  # 适中的批大小
    EPOCHS = 100  # 训练轮数
    LEARNING_RATE = 0.0005  # 较小的学习率，更稳定
    
    # 优化器
    OPTIMIZER = "Adam"
    WEIGHT_DECAY = 1e-5  # 轻度正则化
    BETAS = (0.9, 0.999)  # Adam优化器参数
    
    # 损失函数
    LOSS_FUNCTION = "MSE"
    
    # 早停策略
    EARLY_STOPPING = True
    PATIENCE = 20  # 耐心值
    MIN_DELTA = 1e-7  # 最小改进阈值
    
    # 学习率调度
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "ReduceLROnPlateau"
    SCHEDULER_PATIENCE = 8  # 学习率调度耐心值
    SCHEDULER_FACTOR = 0.5  # 学习率衰减因子
    SCHEDULER_MIN_LR = 1e-6  # 最小学习率
    
    # 梯度裁剪
    USE_GRAD_CLIP = True
    GRAD_CLIP_VALUE = 1.0  # 梯度裁剪阈值
    
    # 设备
    DEVICE = "cuda"  # "cuda" or "cpu"
    
    # 日志
    LOG_INTERVAL = 5  # 每N个epoch打印一次


# ==================== 评估配置 ====================
class EvalConfig:
    """评估配置"""
    # 准确性阈值（相对误差）
    ACCURACY_THRESHOLDS = [5, 10, 15, 20]  # 百分比
    
    # 可视化
    FIGURE_DPI = 150
    FIGURE_FORMAT = "png"
    
    # 案例展示数量
    N_BEST_CASES = 10
    N_WORST_CASES = 10


# ==================== 日志配置 ====================
class LogConfig:
    """日志配置"""
    # 日志级别
    LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # 日志格式
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # 日志文件
    LOG_FILE = LOG_DIR / "training.log"
    
    # 控制台输出
    CONSOLE_OUTPUT = True


# ==================== 可视化配置 ====================
class VisConfig:
    """可视化配置"""
    # 中文字体
    FONT_FAMILY = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
    # 图表样式
    STYLE = "default"
    
    # 颜色方案
    COLORS = {
        'train': 'steelblue',
        'val': 'coral',
        'test': 'green',
        'error': 'red'
    }


# ==================== 导出配置 ====================
def get_config_summary():
    """获取配置摘要"""
    summary = f"""
{'=' * 70}
配置摘要
{'=' * 70}

数据配置:
  - 使用子集: {DataConfig.USE_SUBSET}
  - 子集大小: {DataConfig.SUBSET_SIZE if DataConfig.USE_SUBSET else '全部'}
  - 图像尺寸: {DataConfig.IMG_SIZE}x{DataConfig.IMG_SIZE}
  - 数据分割: {DataConfig.TRAIN_RATIO}/{DataConfig.VAL_RATIO}/{DataConfig.TEST_RATIO}

模型配置:
  - 卷积通道: {ModelConfig.CONV_CHANNELS}
  - 全连接层: {ModelConfig.FC_HIDDEN_DIMS}
  - Dropout率: {ModelConfig.DROPOUT_RATE}

训练配置:
  - 批大小: {TrainConfig.BATCH_SIZE}
  - 训练轮数: {TrainConfig.EPOCHS}
  - 学习率: {TrainConfig.LEARNING_RATE}
  - 优化器: {TrainConfig.OPTIMIZER}
  - 早停: {TrainConfig.EARLY_STOPPING} (耐心={TrainConfig.PATIENCE})

输出路径:
  - 模型: {MODEL_DIR}
  - 图表: {FIGURE_DIR}
  - 日志: {LOG_DIR}
  - 摘要: {SUMMARY_DIR}
"""
    return summary


if __name__ == "__main__":
    print(get_config_summary())
