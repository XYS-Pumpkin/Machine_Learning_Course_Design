# 手写数字识别系统（CNN 与传统分类器对比）

本项目基于经典的 MNIST 数据集，设计并实现了多个传统机器学习分类器（逻辑回归、决策树、KNN、随机森林、SVM、朴素贝叶斯）与一个卷积神经网络（CNN）模型，系统对比其在手写数字识别任务中的性能表现。

## 项目结构

.
├── models/                  # 自定义模型文件，如 cnn.py
├── traditional_models.py    # 训练并评估传统分类器
├── train.py                 # CNN 模型训练脚本（可选 CPU / GPU）
├── test.py                  # CNN 测试与评估脚本
├── data/                    # MNIST 数据集下载路径
├── output/                  # CNN 模型输出结果路径（支持命名子目录）
│   ├── cnn_lr0.001_ep20_GPU/
│   └── ...
└── traditional_models/      # 存放传统分类器图像与报告结果

## 环境依赖

- Python 3.8+
- PyTorch >= 1.12
- torchvision
- numpy
- scikit-learn
- matplotlib

安装推荐：

pip install torch torchvision numpy scikit-learn matplotlib

## 运行方法

1. 训练 CNN 模型

python train.py

可配置项：
- 是否使用 GPU：修改 USE_GPU = True / False
- 学习率、轮次、输出目录：在 train.py 内部配置

2. 测试 CNN 模型

python test.py

测试文件将自动从 output/<子目录>/cnn.pth 加载模型并输出测试报告与图像，结果保存在该目录下的 test/ 子目录中。

3. 传统分类器训练与评估

python traditional_models.py

支持训练以下模型并保存：
- 逻辑回归（Logistic Regression）
- 决策树（Decision Tree）
- 支持向量机（SVM）
- K 近邻（KNN）
- 随机森林（Random Forest）
- 朴素贝叶斯（Naive Bayes）

自动输出图像：
- 各模型的混淆矩阵
- ROC 曲线、PR 曲线、伪 F1 曲线
- 综合雷达图、性能条形图、训练时长图
- 分类报告汇总表

## 示例结果展示

CNN（GPU，lr=0.001，20轮）准确率高达 99.05%，而表现最好的传统分类器为随机森林，准确率达 96.5%。

| 模型         | 准确率   | 训练时长（秒） |
|--------------|----------|----------------|
| CNN (GPU)    | 99.05%   | 143.17         |
| CNN (CPU)    | 99.05%   | 413.61         |
| Logistic     | 92.16%   | 7.83           |
| DecisionTree | 88.31%   | 10.04          |
| LinearSVM    | 86.71%   | 477.81         |
| KNN          | 94.52%   | 0.11           |
| RandomForest | 96.50%   | 29.74          |
| NaiveBayes   | 52.40%   | 0.48           |

## 实验结论

- CNN 表现明显优于传统分类器，尤其在 recall 和精度上更稳定；
- 随机森林和 KNN 在传统方法中具有较好性能；
- GPU 显著加速了 CNN 的训练，尤其在多轮训练中时间优势明显；
- 不同学习率与训练轮次对模型表现影响明显，需综合考虑训练成本与性能；
- 本系统具有良好的可拓展性和可视化分析能力。

## 参考文献

1. LeCun et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
2. Krizhevsky et al. (2012). ImageNet classification with deep convolutional neural networks. NeurIPS.
3. 周志华. 《机器学习》. 清华大学出版社.
4. Goodfellow et al. (2016). Deep Learning. MIT Press.
