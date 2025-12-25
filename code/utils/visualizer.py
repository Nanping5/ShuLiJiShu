"""
可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import VisConfig, EvalConfig


# 设置中文字体
plt.rcParams['font.sans-serif'] = VisConfig.FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(history: dict, save_path: Path = None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    ax.plot(epochs, history['train_losses'], 
            label='训练损失', 
            color=VisConfig.COLORS['train'], 
            linewidth=2)
    ax.plot(epochs, history['val_losses'], 
            label='验证损失', 
            color=VisConfig.COLORS['val'], 
            linewidth=2)
    
    # 标记最佳点
    best_epoch = history['best_epoch'] + 1
    best_loss = history['best_val_loss']
    ax.scatter([best_epoch], [best_loss], 
              color='red', s=100, zorder=5, 
              label=f'最佳 (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('训练与验证损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=EvalConfig.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()


def plot_predictions(actuals: np.ndarray, predictions: np.ndarray, 
                    r2: float, save_path: Path = None):
    """
    绘制预测vs实际
    
    Args:
        actuals: 实际值
        predictions: 预测值
        r2: R²分数
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(actuals, predictions, alpha=0.5, s=20, 
              color=VisConfig.COLORS['test'])
    
    # 理想线
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, label='理想预测')
    
    ax.set_xlabel('实际 Cd')
    ax.set_ylabel('预测 Cd')
    ax.set_title(f'测试集预测结果 (R²={r2:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=EvalConfig.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()


def plot_error_analysis(errors: np.ndarray, relative_errors: np.ndarray,
                       actuals: np.ndarray, save_path: Path = None):
    """
    绘制误差分析
    
    Args:
        errors: 绝对误差
        relative_errors: 相对误差
        actuals: 实际值
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 误差分布
    axes[0, 0].hist(errors, bins=50, color='steelblue', 
                   edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('预测误差 (Cd)')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('预测误差分布')
    axes[0, 0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 相对误差分布
    axes[0, 1].hist(relative_errors, bins=50, color='coral', 
                   edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('相对误差 (%)')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('相对误差分布')
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 误差vs实际值
    axes[1, 0].scatter(actuals, errors, alpha=0.5, s=20, color='green')
    axes[1, 0].set_xlabel('实际 Cd')
    axes[1, 0].set_ylabel('预测误差')
    axes[1, 0].set_title('误差 vs 实际值')
    axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 累积误差分布
    sorted_rel_errors = np.sort(relative_errors)
    cumulative = np.arange(1, len(sorted_rel_errors) + 1) / len(sorted_rel_errors) * 100
    axes[1, 1].plot(sorted_rel_errors, cumulative, linewidth=2, color='purple')
    axes[1, 1].axvline(10, color='r', linestyle='--', linewidth=2, label='10%误差线')
    axes[1, 1].set_xlabel('相对误差 (%)')
    axes[1, 1].set_ylabel('累积百分比 (%)')
    axes[1, 1].set_title('累积误差分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=EvalConfig.FIGURE_DPI, bbox_inches='tight')
    
    plt.close()
