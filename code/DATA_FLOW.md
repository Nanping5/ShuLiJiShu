# 数据流程详解 - CNN输入说明

## 🎯 核心问题：CNN输入的是什么？

**答案：是的，CNN输入的是翼型图像！**

## 📊 完整数据流程

### 1️⃣ 原始数据（CSV）

**文件**: `data/COMPILED_AIRFOIL_DATA.csv`

```
Filename, AoA, CST Coeff 1-8, Cl, Cd
E180, 0, -0.084, -0.041, ..., 0.144, 0.015
```

**包含**:
- 翼型名称
- 攻角(AoA)
- 8个CST系数（几何参数）
- 升力系数(Cl)
- 阻力系数(Cd)

### 2️⃣ 数据预处理（生成图像）

**脚本**: `preprocess_data.py`

#### 步骤A: CST系数 → 翼型坐标
```python
def cst_airfoil(coeffs, N=200):
    """
    输入: 8个CST系数 [X1, X2, ..., X8]
    输出: 翼型坐标 (x, y)
    
    过程:
    1. 使用CST参数化方法
    2. 生成200个点的翼型轮廓
    3. 返回闭合曲线坐标
    """
    # CST数学公式
    # 生成翼型形状
    return x, y  # 翼型坐标
```

#### 步骤B: 翼型坐标 → 图像
```python
def coord_to_image(x, y, img_size=128):
    """
    输入: 翼型坐标 (x, y)
    输出: 128×128灰度图像
    
    过程:
    1. 创建空白图像（黑色背景）
    2. 将坐标归一化到图像尺寸
    3. 绘制翼型轮廓（白色）
    4. 返回图像数组
    """
    img = Image.new('L', (128, 128), 0)  # 灰度图
    draw.polygon(points, fill=255)  # 绘制翼型
    return img_array  # (128, 128, 1)
```

**示例图像**:
```
黑色背景 + 白色翼型轮廓
┌─────────────────┐
│                 │
│   ╱‾‾‾‾‾‾╲     │  ← 翼型轮廓（白色）
│  ╱        ╲    │
│ ╱__________╲   │
│                 │
└─────────────────┘
  128×128像素
```

### 3️⃣ 数据集格式

**文件**: `data/dataset.npz`

```python
{
    'X_train': (26964, 128, 128, 1),  # 训练图像
    'p_train': (26964,),              # 攻角参数
    'y_cd_train': (26964,),           # Cd标签
    
    'X_val': (3370, 128, 128, 1),    # 验证图像
    'p_val': (3370,),                 # 攻角参数
    'y_cd_val': (3370,),              # Cd标签
    
    'X_test': (3371, 128, 128, 1),   # 测试图像
    'p_test': (3371,),                # 攻角参数
    'y_cd_test': (3371,)              # Cd标签
}
```

### 4️⃣ PyTorch数据加载

**类**: `AirfoilDataset`

```python
class AirfoilDataset(Dataset):
    def __init__(self, images, params, labels):
        # 转换维度: (N, H, W, 1) → (N, 1, H, W)
        self.images = torch.tensor(images).permute(0, 3, 1, 2)
        self.params = torch.tensor(params).unsqueeze(1)
        self.labels = torch.tensor(labels).unsqueeze(1)
    
    def __getitem__(self, idx):
        return self.images[idx], self.params[idx], self.labels[idx]
```

**输出格式**:
```python
img:   (1, 128, 128)  # 单通道图像
param: (1,)           # AoA参数
label: (1,)           # Cd标签
```

### 5️⃣ CNN模型输入

**模型**: `AirfoilCNN`

```python
def forward(self, img, param):
    """
    输入:
        img:   (batch, 1, 128, 128)  ← 翼型图像！
        param: (batch, 1)             ← 攻角参数
    
    处理流程:
        1. 图像 → 卷积层 → 特征提取
        2. 特征 + 参数 → 全连接层
        3. 输出 → Cd预测
    
    输出:
        (batch, 1)  # Cd预测值
    """
    x = self.conv(img)           # 卷积处理图像
    x = torch.cat((x, param), dim=1)  # 拼接参数
    return self.fc(x)            # 全连接输出
```

## 🔍 详细数据流

```
原始CSV数据
    ↓
[CST系数] → cst_airfoil() → [翼型坐标]
    ↓
[翼型坐标] → coord_to_image() → [128×128图像]
    ↓
[图像数组] → AirfoilDataset → [PyTorch张量]
    ↓
[图像张量] → CNN卷积层 → [特征向量]
    ↓
[特征 + AoA] → 全连接层 → [Cd预测]
```

## 📸 实际例子

### 输入数据
```python
# 原始CSV一行
Filename: "NACA0012"
AoA: 5°
CST Coeff: [-0.084, -0.041, ..., 0.170, 0.040]
Cd: 0.0153  # 这是我们要预测的目标
```

### 处理过程
```python
# 1. 生成翼型坐标
x, y = cst_airfoil([-0.084, -0.041, ..., 0.170, 0.040])
# x, y: 200个点的翼型轮廓

# 2. 转换为图像
img = coord_to_image(x, y)
# img: (128, 128, 1) 灰度图像

# 3. 输入CNN
prediction = model(img, aoa=5)
# prediction: 0.0148 (预测的Cd值)
```

## 🎯 为什么这样设计？

### 优势1: 端到端学习
```
传统方法: CST系数 → 手工特征 → 预测
我们的方法: 图像 → CNN自动提取特征 → 预测
```

### 优势2: 几何信息完整
- 图像包含完整的翼型形状
- CNN可以学习几何特征（厚度、弯度、前缘形状等）
- 不依赖手工设计的特征

### 优势3: 符合课题要求
- ✅ 使用卷积神经网络（CNN）
- ✅ 处理几何目标（翼型图像）
- ✅ 预测风阻系数（Cd）

## 📊 数据维度总结

| 阶段 | 数据类型 | 维度 | 说明 |
|------|---------|------|------|
| CSV | CST系数 | (8,) | 几何参数 |
| 预处理 | 翼型坐标 | (400,) | x,y各200点 |
| 图像 | 灰度图 | (128, 128, 1) | 单通道图像 |
| PyTorch | 张量 | (1, 128, 128) | CHW格式 |
| 批处理 | 批张量 | (batch, 1, 128, 128) | 批量输入 |
| CNN输出 | 预测值 | (batch, 1) | Cd预测 |

## 🔬 验证CNN确实接收图像

### 测试代码
```python
# 查看实际输入
import torch
from data import load_dataset

datasets = load_dataset('data/dataset.npz')
img, param, label = datasets['train'][0]

print(f"图像形状: {img.shape}")        # (1, 128, 128)
print(f"图像类型: {img.dtype}")        # torch.float32
print(f"图像范围: [{img.min():.2f}, {img.max():.2f}]")  # [0.00, 1.00]
print(f"是否为图像: {len(img.shape) == 3 and img.shape[0] == 1}")  # True

# 可视化图像
import matplotlib.pyplot as plt
plt.imshow(img.squeeze(), cmap='gray')
plt.title('翼型图像')
plt.show()
```

## ✅ 结论

**我们的实现完全正确！**

1. ✅ CNN输入的是**翼型图像**（128×128灰度图）
2. ✅ 图像是从CST系数**生成**的翼型轮廓
3. ✅ 同时输入**攻角参数**作为辅助信息
4. ✅ 输出**阻力系数Cd**的预测值

这是一个**图像回归任务**，使用CNN从翼型图像中提取几何特征，结合攻角参数，预测气动性能。

---

*文档创建于 2025-12-19*
