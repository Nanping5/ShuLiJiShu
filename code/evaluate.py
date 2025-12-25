"""
模型评估主脚本
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import TrainConfig, ModelConfig, PROCESSED_DATA_PATH, MODEL_DIR, FIGURE_DIR, SUMMARY_DIR, EvalConfig
from models import create_model
from data import load_dataset
from utils import get_logger, LoggerContext
from utils.visualizer import plot_predictions, plot_error_analysis


def evaluate_model(model, test_loader, device, logger, label_stats=None):
    """
    评估模型性能
    
    Args:
        label_stats: 标签归一化统计信息（如果启用了归一化）
    
    Returns:
        predictions, actuals, metrics
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for img, param, label in test_loader:
            img, param = img.to(device), param.to(device)
            output = model(img, param)
            predictions.extend(output.cpu().numpy())
            actuals.extend(label.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # 反归一化（仅当启用了标签归一化时）
    if label_stats is not None:
        logger.info("  反归一化预测值和真实值...")
        predictions = predictions * label_stats['std'] + label_stats['mean']
        actuals = actuals * label_stats['std'] + label_stats['mean']
    
    logger.info(f"  Cd预测范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
    logger.info(f"  Cd真实范围: [{actuals.min():.6f}, {actuals.max():.6f}]")
    
    # 计算指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    
    # 相对误差（避免除零）
    relative_errors = np.abs(errors / (actuals + 1e-8)) * 100
    
    # 准确性分级
    accuracy_stats = {}
    for threshold in EvalConfig.ACCURACY_THRESHOLDS:
        count = np.sum(relative_errors < threshold)
        percentage = count / len(relative_errors) * 100
        accuracy_stats[threshold] = {
            'count': count,
            'percentage': percentage
        }
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_error': errors.mean(),
        'std_error': errors.std(),
        'median_abs_error': np.median(abs_errors),
        'mean_relative_error': relative_errors.mean(),
        'median_relative_error': np.median(relative_errors),
        'accuracy_stats': accuracy_stats
    }
    
    return predictions, actuals, errors, relative_errors, metrics


def main():
    """主评估流程"""
    # 设置日志
    logger = get_logger("evaluate")
    logger.info("=" * 70)
    logger.info("开始评估流程")
    logger.info("=" * 70)
    
    # 检查设备
    device = torch.device(TrainConfig.DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据（优化：获取归一化信息）
    with LoggerContext(logger, "加载测试数据"):
        from config import DataConfig
        datasets = load_dataset(
            str(PROCESSED_DATA_PATH),
            normalize_labels=DataConfig.NORMALIZE_LABELS,
            normalize_params=DataConfig.NORMALIZE_PARAMS,
            use_augmentation=False  # 测试集不使用数据增强
        )
        test_dataset = datasets['test']
        test_loader = DataLoader(
            test_dataset,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=False
        )
        logger.info(f"测试集: {len(test_dataset)} 样本")
        logger.info(f"标签归一化: {DataConfig.NORMALIZE_LABELS}")
        
        # 获取归一化统计信息
        label_stats = test_dataset.label_stats if hasattr(test_dataset, 'label_stats') else None
    
    # 加载模型
    with LoggerContext(logger, "加载训练好的模型"):
        model = create_model(ModelConfig(), device=str(device))
        checkpoint = torch.load(MODEL_DIR / 'best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载第 {checkpoint['epoch']+1} 轮的最佳模型")
    
    # 评估模型（优化：传入归一化统计信息）
    with LoggerContext(logger, "评估模型性能"):
        predictions, actuals, errors, relative_errors, metrics = evaluate_model(
            model, test_loader, device, logger, label_stats=label_stats
        )
    
    # 打印评估结果
    logger.info("-" * 70)
    logger.info("评估结果:")
    logger.info(f"  MSE:  {metrics['mse']:.8f}")
    logger.info(f"  RMSE: {metrics['rmse']:.8f}")
    logger.info(f"  MAE:  {metrics['mae']:.8f}")
    logger.info(f"  R²:   {metrics['r2']:.6f}")
    logger.info(f"  MAPE: {metrics['mean_relative_error']:.2f}%")
    
    logger.info("\n准确性分级（相对误差）:")
    for threshold, stats in metrics['accuracy_stats'].items():
        logger.info(f"  < {threshold}%: {stats['percentage']:.2f}% ({stats['count']}/{len(actuals)} 样本)")
    
    # 生成可视化
    with LoggerContext(logger, "生成可视化图表"):
        # 预测vs实际
        plot_predictions(
            actuals, predictions, metrics['r2'],
            save_path=FIGURE_DIR / "predictions.png"
        )
        logger.info(f"预测图表已保存: {FIGURE_DIR / 'predictions.png'}")
        
        # 误差分析
        plot_error_analysis(
            errors, relative_errors, actuals,
            save_path=FIGURE_DIR / "error_analysis.png"
        )
        logger.info(f"误差分析已保存: {FIGURE_DIR / 'error_analysis.png'}")
    
    # 保存评估报告
    with LoggerContext(logger, "保存评估报告"):
        report_path = SUMMARY_DIR / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""
{'=' * 70}
模型评估报告
{'=' * 70}

测试集信息:
  - 样本数量: {len(actuals)}
  - Cd范围: [{actuals.min():.6f}, {actuals.max():.6f}]
  - 平均Cd: {actuals.mean():.6f}

基本指标:
  - 均方误差 (MSE):        {metrics['mse']:.8f}
  - 均方根误差 (RMSE):     {metrics['rmse']:.8f}
  - 平均绝对误差 (MAE):    {metrics['mae']:.8f}
  - 决定系数 (R²):         {metrics['r2']:.6f}

误差统计:
  - 平均误差:              {metrics['mean_error']:.8f}
  - 误差标准差:            {metrics['std_error']:.8f}
  - 中位数绝对误差:        {metrics['median_abs_error']:.8f}

相对误差统计:
  - 平均相对误差 (MAPE):   {metrics['mean_relative_error']:.2f}%
  - 中位数相对误差:        {metrics['median_relative_error']:.2f}%

准确性分级（基于相对误差）:
""")
            for threshold, stats in metrics['accuracy_stats'].items():
                f.write(f"  相对误差 < {threshold}%: {stats['percentage']:.2f}% ({stats['count']}/{len(actuals)} 样本)\n")
            
            f.write(f"""
输出文件:
  - 预测图表: {FIGURE_DIR / 'predictions.png'}
  - 误差分析: {FIGURE_DIR / 'error_analysis.png'}
""")
        logger.info(f"评估报告已保存: {report_path}")
    
    logger.info("=" * 70)
    logger.info("评估流程完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
