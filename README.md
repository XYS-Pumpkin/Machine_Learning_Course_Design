# 基于卷积神经网络的手写数字识别（CNN vs 传统模型）

## 📘 项目简介  
本项目基于 PyTorch 实现手写数字识别系统，使用卷积神经网络（CNN）对 MNIST 数据集进行训练与测试，并与多种传统分类器（逻辑回归、决策树、线性SVM、K近邻、随机森林、朴素贝叶斯）进行性能对比，分析训练效率与分类精度。

---

## 🛠 环境配置  

### 1. 创建 Python 虚拟环境

```bash
# 创建名为 .MLCD 的虚拟环境
# 如果使用 venv：
python3.12 -m venv .MLCD
.MLCD\Scripts\activate      # Windows

# 安装依赖（基于 pip）
pip install -r requirements.txt
```

---

## 📁 项目结构  
```
Machine_Learning_Course_Design/
├── data/                        # MNIST 数据集（首次运行自动下载）
│
├── models/                      # 模型结构文件
│   └── cnn.py                   # CNN 网络定义
│
├── output/                      # CNN 实验结果输出目录（按实验自动创建）
│   └── cnn_{优化器}_lr{lr}_ep{ep}_{GPU/CPU}/
│       ├── cnn.pth              # 保存的模型参数
│       ├── CNN训练验证损失曲线.png
│       └── test/                # 测试阶段结果
│           ├── 混淆矩阵热力图.png
│           ├── ROC曲线.png
│           └── 分类报告.txt
│
├── traditional_models/          # 传统模型结果输出
│   ├── 模型性能条形图.png
│   ├── 模型雷达图.png
│   ├── 模型训练时长条形图.png
│   ├── 各模型的ROC、PR曲线、混淆矩阵等图表
│   └── 分类报告.txt
│
├── train.py                     # CNN训练脚本（支持自定义参数）
├── test.py                      # CNN模型测试脚本（自动识别输出目录）
├── traditional_models.py        # 各传统分类器训练与评估脚本
├── requirements.txt             # 项目依赖包
└── README.md                    # 项目说明文件

```

---

## 🚀 使用方法  

### 1. 训练 CNN 模型  
```bash
python train.py
```

**功能说明**：  
- 支持选择是否使用 GPU（USE_GPU）、自定义学习率、轮次、优化器
- 自动保存模型、损失图、训练时间到指定目录  

**输出目录示例**：  
```
output/cnn_lr0.001_ep20_GPU/
├── cnn.pth
├── CNN训练验证损失曲线.png
└── test/
    ├── 混淆矩阵热力图.png
    ├── ROC曲线.png
    └── 分类报告.txt
```

### 2. 测试 CNN 模型  
```bash
python test.py
```
- 自动加载指定目录的模型文件进行评估  
- 输出指标：准确率、精确率、召回率、F1值、Kappa 系数、混淆矩阵、ROC 曲线

### 3. 训练并评估传统分类模型  
```bash
python traditional_models.py
```
**包含模型**：  
- 逻辑回归（LogisticRegression）
- 决策树（DecisionTree）
- 线性SVM（LinearSVC，采样10000条数据）
- K近邻（KNN）
- 随机森林（RandomForest）
- 朴素贝叶斯（Naive Bayes）

**输出内容**：  
- 混淆矩阵图、ROC、PR、F1伪曲线
- 模型训练时长对比图
- 模型性能条形图和雷达图
- 分类报告（文本）

---

## 🧠 模型与算法说明

### CNN 模型结构（models/cnn.py）
- 两层卷积 + ReLU + 最大池化：
  - Conv1: 输入通道1 → 输出通道32，卷积核3x3，padding=1
  - Conv2: 32通道 → 64通道，卷积核3x3，padding=1
- 最大池化：kernel_size=2, stride=2
- 全连接层：
  - Flatten → Linear(64×7×7 → 128) → ReLU
  - Linear(128 → 10)，输出为10类
- 最后使用 Softmax（通过 `CrossEntropyLoss` 内部实现）

**训练参数配置**：
- Optimizer：支持 `Adam`, `SGD(momentum=0.9)`, `RMSprop`
- Loss Function：`CrossEntropyLoss`
- Learning rate：可调，默认 `0.001`
- Epochs：默认 `20`，支持动态设置
- Batch size：64
- 数据归一化：`mean=0.1307`, `std=0.3081`
- 输出模型与图像文件路径自动命名，如：
  - `output/cnn_Adam_lr0.001_ep20_GPU/cnn.pth`

---

### 传统模型参数设置（见 traditional_models.py）

| 模型       | 参数说明                                |
|------------|-----------------------------------------|
| 逻辑回归   | `LogisticRegression(max_iter=1000)`     |
| 决策树     | `DecisionTreeClassifier(max_depth=15)`  |
| 线性SVM    | `LinearSVC(max_iter=3000)`（10k样本）   |
| K近邻      | `KNeighborsClassifier(n_neighbors=3)`   |
| 随机森林   | `RandomForestClassifier(n_estimators=100, max_depth=15)` |
| 朴素贝叶斯 | `GaussianNB()`（默认参数）              |

说明：
- 所有模型使用 MNIST 展平数据（784维）
- 所有输入先进行标准化 `StandardScaler()`
- 线性SVM 为节省时间，仅用训练集前10000个样本


## 📊 输出图示与指标  

- 准确率（Accuracy）  
- 精确率（Precision）  
- 召回率（Recall）  
- F1 值（F1-score）  
- Kappa 系数  
- AUC 指标（多类 ROC）  

**输出图**：
- CNN 损失曲线图
- 各模型混淆矩阵热力图
- 各模型 ROC 曲线
- 模型性能雷达图 / 条形图
- 分类报告文本

---

## 📝 实验总结与结论  

- CNN 模型在 GPU 下训练速度远快于 CPU（约提升 3 倍）
- 学习率设置对模型性能有显著影响，0.001~0.0005 较优
- Adam 收敛速度快，SGD 更稳定，RMSprop 折中
- 在 MNIST 上 CNN 可达 99.05% 精度，显著优于传统模型（最高约 96.5%）
- 传统分类器在资源受限设备上仍具部署价值

---

## ❗注意事项  
1. 初次运行将自动下载 MNIST 数据集  
2. SVM 训练慢，代码已内置样本数限制优化  
3. 所有输出会按结构保存，便于批量实验对比分析  
4. 图像保存路径包含参数名（如学习率、轮次等）

---

## 📌 推荐实验流程  
```bash
# 训练 CNN 模型
python train.py

# 测试 CNN 模型
python test.py

# 训练传统模型并生成图表
python traditional_models.py
```

---
## 参考文献

1. LeCun et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
2. Krizhevsky et al. (2012). ImageNet classification with deep convolutional neural networks. NeurIPS.
3. 周志华. 《机器学习》. 清华大学出版社.
4. Goodfellow et al. (2016). Deep Learning. MIT Press.
