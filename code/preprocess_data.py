"""
数据预处理脚本 - 从CSV生成完整数据集
"""
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import math
from pathlib import Path

from config import DataConfig, RAW_DATA_PATH, PROCESSED_DATA_PATH
from utils import get_logger

logger = get_logger("preprocess")

logger.info("=" * 70)
logger.info("数据预处理 - 从CSV生成翼型图像数据集")
logger.info("=" * 70)


def cst_airfoil(coeffs, N=200, te_thickness=0.0):
    """
    使用CST方法生成翼型坐标
    
    Args:
        coeffs: [X1 to X8] CST系数
        N: 生成点数
        te_thickness: 尾缘厚度
    
    Returns:
        x, y: 翼型坐标 (闭合曲线)
    """
    order = len(coeffs) - 1
    psi = np.linspace(0, 1, N)
    
    # 类函数 (N1=0.5, N2=1.0 for airfoil)
    C = np.sqrt(psi) * (1 - psi)
    
    # Bernstein多项式
    B = np.zeros((order + 1, N))
    for k in range(order + 1):
        B[k] = math.comb(order, k) * (psi ** k) * ((1 - psi) ** (order - k))
    
    # 形状函数
    S = np.dot(coeffs, B)
    
    # 上下表面 (简化：假设对称)
    y_upper = C * S + psi * te_thickness / 2
    y_lower = -C * S - psi * te_thickness / 2
    
    # 组合成闭合曲线
    x = psi
    x_full = np.concatenate((x[::-1], x))
    y_full = np.concatenate((y_lower[::-1], y_upper))
    
    return x_full, y_full


def coord_to_image(x, y, img_size=128, line_width=2):
    """
    将翼型坐标转换为图像
    
    Args:
        x, y: 翼型坐标
        img_size: 图像尺寸
        line_width: 线宽
    
    Returns:
        img_array: (H, W, 1) 归一化图像数组
    """
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    
    # 归一化坐标
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * (img_size - 1)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8) * (img_size - 1) * 0.8 + img_size * 0.1
    
    # 翻转y轴（图像坐标系）
    points = list(zip(x_norm, img_size - 1 - y_norm))
    
    # 绘制多边形
    draw.polygon(points, fill=255, outline=255)
    
    # 转换为numpy数组并归一化
    img_array = np.array(img) / 255.0
    return img_array[..., np.newaxis]


def main():
    # 1. 加载数据
    logger.info(f"加载原始数据: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"✓ 数据加载成功: {df.shape}")
    logger.info(f"  样本总数: {len(df)}")
    logger.info(f"  翼型数量: {df['Filename'].nunique()}")
    logger.info(f"  攻角范围: {sorted(df['AoA'].unique())}")
    
    # 2. 决定使用的样本数
    if DataConfig.USE_SUBSET:
        df_subset = df.sample(n=min(DataConfig.SUBSET_SIZE, len(df)), 
                             random_state=DataConfig.RANDOM_SEED).reset_index(drop=True)
        logger.info(f"⚠ 使用子集: {len(df_subset)} 样本")
    else:
        df_subset = df
        logger.info(f"✓ 使用全部数据: {len(df_subset)} 样本")
    
    # 3. 批量生成图像
    logger.info("-" * 70)
    logger.info("开始生成翼型图像...")
    
    images = []
    labels_cd = []
    labels_cl = []
    params_aoa = []
    
    total = len(df_subset)
    log_interval = max(1, total // 20)  # 每5%打印一次
    
    for idx, row in df_subset.iterrows():
        if idx % log_interval == 0:
            logger.info(f"  进度: {idx}/{total} ({idx/total*100:.1f}%)")
        
        try:
            # 提取CST系数
            coeffs = [row[f'CST Coeff {i}'] for i in range(1, 9)]
            
            # 生成翼型坐标
            x, y = cst_airfoil(coeffs, N=DataConfig.CST_N_POINTS, 
                              te_thickness=DataConfig.TE_THICKNESS)
            
            # 转换为图像
            img = coord_to_image(x, y, img_size=DataConfig.IMG_SIZE, 
                                line_width=DataConfig.LINE_WIDTH)
            
            # 保存
            images.append(img)
            labels_cd.append(row['Cd'])
            labels_cl.append(row['Cl'])
            params_aoa.append(row['AoA'])
            
        except Exception as e:
            logger.warning(f"样本 {idx} 处理失败: {e}")
            continue
    
    # 转换为numpy数组
    images = np.array(images, dtype=np.float32)
    labels_cd = np.array(labels_cd, dtype=np.float32)
    labels_cl = np.array(labels_cl, dtype=np.float32)
    params_aoa = np.array(params_aoa, dtype=np.float32)
    
    logger.info(f"✓ 图像生成完成!")
    logger.info(f"  图像形状: {images.shape}")
    logger.info(f"  Cd标签形状: {labels_cd.shape}")
    logger.info(f"  Cl标签形状: {labels_cl.shape}")
    logger.info(f"  AoA参数形状: {params_aoa.shape}")
    
    # 4. 数据集分割
    logger.info("-" * 70)
    logger.info("分割数据集...")
    
    # 第一次分割: 训练集 vs 临时集
    X_train, X_temp, y_cd_train, y_cd_temp, y_cl_train, y_cl_temp, p_train, p_temp = train_test_split(
        images, labels_cd, labels_cl, params_aoa,
        test_size=(DataConfig.VAL_RATIO + DataConfig.TEST_RATIO),
        random_state=DataConfig.RANDOM_SEED
    )
    
    # 第二次分割: 验证集 vs 测试集
    val_ratio_adjusted = DataConfig.VAL_RATIO / (DataConfig.VAL_RATIO + DataConfig.TEST_RATIO)
    X_val, X_test, y_cd_val, y_cd_test, y_cl_val, y_cl_test, p_val, p_test = train_test_split(
        X_temp, y_cd_temp, y_cl_temp, p_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=DataConfig.RANDOM_SEED
    )
    
    logger.info(f"✓ 数据集分割完成:")
    logger.info(f"  训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(images)*100:.1f}%)")
    logger.info(f"  验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(images)*100:.1f}%)")
    logger.info(f"  测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(images)*100:.1f}%)")
    
    # 5. 保存数据集
    logger.info("-" * 70)
    logger.info(f"保存数据集: {PROCESSED_DATA_PATH}")
    
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        # 训练集
        X_train=X_train, y_cd_train=y_cd_train, y_cl_train=y_cl_train, p_train=p_train,
        # 验证集
        X_val=X_val, y_cd_val=y_cd_val, y_cl_val=y_cl_val, p_val=p_val,
        # 测试集
        X_test=X_test, y_cd_test=y_cd_test, y_cl_test=y_cl_test, p_test=p_test
    )
    
    file_size = PROCESSED_DATA_PATH.stat().st_size / 1024 / 1024
    logger.info(f"✓ 数据集已保存: {file_size:.2f} MB")
    
    # 6. 数据统计
    logger.info("-" * 70)
    logger.info("数据集统计:")
    logger.info(f"\n训练集 Cd 统计:")
    logger.info(f"  均值: {y_cd_train.mean():.6f}")
    logger.info(f"  标准差: {y_cd_train.std():.6f}")
    logger.info(f"  范围: [{y_cd_train.min():.6f}, {y_cd_train.max():.6f}]")
    
    logger.info(f"\n训练集 AoA 分布:")
    unique_aoa, counts = np.unique(p_train, return_counts=True)
    for aoa, count in zip(unique_aoa, counts):
        logger.info(f"  AoA {int(aoa):2d}°: {count:4d} 样本")
    
    logger.info("=" * 70)
    logger.info("数据预处理完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
