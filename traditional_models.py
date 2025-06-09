import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torchvision import datasets, transforms
from matplotlib import rcParams

# 设置黑体中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

all_metrics = {}
train_times = {}

# 1. 加载 MNIST 数据集

def load_data():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    X_train = train_data.data.view(-1, 28 * 28).numpy()
    y_train = train_data.targets.numpy()
    X_test = test_data.data.view(-1, 28 * 28).numpy()
    y_test = test_data.targets.numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train_bin = label_binarize(y_train, classes=np.arange(10))
    y_test_bin = label_binarize(y_test, classes=np.arange(10))
    return X_train, y_train, y_train_bin, X_test, y_test, y_test_bin

# 2. 训练和评估模型

def main():
    os.makedirs("traditional_models", exist_ok=True)
    X_train, y_train, y_train_bin, X_test, y_test, y_test_bin = load_data()

    # 线性SVM优化：采用更大规模（10000），提高max_iter
    X_svm, _, y_svm, _ = train_test_split(X_train, y_train, train_size=10000, stratify=y_train, random_state=42)

    models = {
        "逻辑回归": LogisticRegression(max_iter=1000),
        "决策树": DecisionTreeClassifier(max_depth=15),
        "线性SVM": (LinearSVC(max_iter=3000), X_svm, y_svm),
        "K近邻": KNeighborsClassifier(n_neighbors=3),
        "随机森林": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        "朴素贝叶斯": GaussianNB()
    }

    for name in models:
        print(f"\n正在训练模型：{name}")
        if name == "线性SVM":
            model, X_fit, y_fit = models[name]
        else:
            model = models[name]
            X_fit, y_fit = X_train, y_train

        start_time = time.time()
        model.fit(X_fit, y_fit)
        duration = time.time() - start_time
        train_times[name] = duration
        print(f"训练时长：{duration:.2f} 秒")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)

        all_metrics[name] = (acc, prec, rec, f1, kappa, cm, report)

        # 混淆矩阵图
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        plt.title(f"{name} 混淆矩阵热力图")
        plt.tight_layout()
        plt.savefig(f"traditional_models/{name}_混淆矩阵热力图.png")
        plt.close()

        # ROC & PR 曲线
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            if y_score.ndim == 1:
                y_score = np.stack([1 - y_score, y_score], axis=1)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("假正率")
        plt.ylabel("真正率")
        plt.title(f"{name} ROC曲线")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"traditional_models/{name}_ROC曲线.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
        ap = average_precision_score(y_test_bin, y_score, average='macro')
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("召回率")
        plt.ylabel("查准率")
        plt.title(f"{name} PR曲线")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"traditional_models/{name}_PR曲线.png")
        plt.close()

        thresholds = np.linspace(0.1, 0.9, 9)
        f1_scores = []
        for t in thresholds:
            pred_bin = (y_score > t).astype(int)
            correct = (pred_bin == y_test_bin).sum()
            f1_scores.append(correct / pred_bin.size)
        plt.figure()
        plt.plot(thresholds, f1_scores, marker='o')
        plt.xlabel("阈值")
        plt.ylabel("伪F1分数")
        plt.title(f"{name} 伪F1分数曲线")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"traditional_models/{name}_F1分数伪曲线.png")
        plt.close()

    # 条形图：训练时长
    names = list(train_times.keys())
    times = list(train_times.values())
    plt.figure(figsize=(10, 6))
    plt.bar(names, times)
    plt.ylabel("训练时长（秒）")
    plt.title("各模型训练时长比较")
    plt.tight_layout()
    plt.savefig("traditional_models/模型训练时长条形图.png")
    plt.close()

    # 条形图：综合性能
    acc, prec, rec, f1, kappa = zip(*[v[:5] for v in all_metrics.values()])
    x = np.arange(len(names))
    width = 0.15
    plt.figure(figsize=(12,6))
    plt.bar(x - 2*width, acc, width, label='准确率')
    plt.bar(x - width, prec, width, label='查准率')
    plt.bar(x, rec, width, label='查全率')
    plt.bar(x + width, f1, width, label='F1值')
    plt.bar(x + 2*width, kappa, width, label='Kappa系数')
    plt.xticks(x, names)
    plt.ylabel("得分")
    plt.title("多模型性能对比条形图")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("traditional_models/模型性能条形图.png")
    plt.close()

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(8,8))
    for name in names:
        values = list(all_metrics[name][:5])
        values += values[:1]
        plt.polar(angles, values, label=name, alpha=0.3)
    plt.xticks(angles[:-1], ['准确率', '查准率', '查全率', 'F1值', 'Kappa'])
    plt.title("模型综合能力雷达图")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("traditional_models/模型雷达图.png")
    plt.close()

    # 分类报告输出为txt文件
    with open("traditional_models/分类报告.txt", "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"模型：{name}\n")
            f.write(all_metrics[name][6])
            f.write("\n---------------------\n")

if __name__ == "__main__":
    main()
