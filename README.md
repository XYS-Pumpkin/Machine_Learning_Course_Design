# 基于卷积神经网络的手写数字识别

## 📌 项目简介
本项目实现了一个基于 PyTorch 框架的卷积神经网络（CNN），用于对 MNIST 手写数字图像进行分类识别，并与三种传统机器学习方法（逻辑回归、决策树、支持向量机）进行性能对比。对比评估指标包括准确率（Accuracy）、查准率（Precision）、查全率（Recall）和 F1 值。

---

## 🧠 项目结构
```
digit_cnn/
├── data/                    # MNIST 数据自动下载存放目录
├── models/
│   └── cnn.py              # CNN 模型定义
├── train.py                # 训练 CNN 模型脚本
├── test.py                 # 测试 CNN 模型并输出评价指标
├── traditional_models.py   # 三种传统分类器的对比测试脚本
├── requirements.txt        # 项目依赖库
├── README.md               # 当前说明文档
```

---

## 🚀 快速开始

### 1. 安装依赖
确保你使用 Python 3.12+，建议先创建虚拟环境。

```bash
pip install -r requirements.txt
```

### 2. 训练 CNN 模型
```bash
python train.py
```
训练完成后，模型参数会被保存到 `models/cnn.pth`。

### 3. 测试 CNN 模型
```bash
python test.py
```
将输出模型在测试集上的准确率、查准率、查全率、F1值。

### 4. 运行传统分类器比较
```bash
python traditional_models.py
```
输出逻辑回归、决策树和 SVM 的四项评价指标对比结果。

---

## 📊 模型评价指标
- **Accuracy 准确率**：整体分类正确的比例
- **Precision 查准率**：预测为某类中，真正为该类的比例
- **Recall 查全率**：该类中被成功预测出来的比例
- **F1-score F1 值**：查准率与查全率的调和平均值

---

## 📎 使用到的库
- torch, torchvision
- numpy
- matplotlib
- scikit-learn

---

## 📬 联系方式
作者：Your Name（可填学号）  
课程：2025年《机器学习课程设计》  
如有问题，请联系作者或在提交平台上反馈。

---

✅ 项目准备完毕，感谢使用！
