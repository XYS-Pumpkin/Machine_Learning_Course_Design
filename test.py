import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import numpy as np
from models.cnn import CNN
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay

# 设置中文字体为黑体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ====== 路径设置 ======
model_dir = "output/cnn_RMSprop_lr0.001_ep20_GPU"  # 可修改这行为任何模型目录
model_path = os.path.join(model_dir, "cnn.pth")
test_output_dir = os.path.join(model_dir, "test")
os.makedirs(test_output_dir, exist_ok=True)

# ====== 设置设备 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用的设备是:", device)
print("当前设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")

# ====== 加载测试数据 ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ====== 加载模型 ======
model = CNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ====== 预测 ======
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = nn.functional.softmax(outputs, dim=1).cpu()
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.numpy())

# ====== 指标计算 ======
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
kappa = cohen_kappa_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

# ====== 控制台输出（简洁） ======
print(f"\n模型在测试集上的表现：")
print(f"准确率 Accuracy     : {acc:.4f}")
print(f"查准率 Precision    : {prec:.4f}")
print(f"查全率 Recall       : {rec:.4f}")
print(f"F1值 F1-score       : {f1:.4f}")
print(f"Kappa 系数          : {kappa:.4f}")

# ====== 混淆矩阵图 ======
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.title("CNN 混淆矩阵热力图")
plt.tight_layout()
plt.savefig(os.path.join(test_output_dir, "CNN_混淆矩阵热力图.png"))
plt.close()
print("混淆矩阵图已保存")

# ====== ROC 曲线图 ======
all_labels_bin = label_binarize(all_labels, classes=np.arange(10))
all_probs_np = np.array(all_probs)
fpr, tpr, _ = roc_curve(all_labels_bin.ravel(), all_probs_np.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("假正率")
plt.ylabel("真正率")
plt.title("CNN ROC曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(test_output_dir, "CNN_ROC曲线.png"))
plt.close()
print("ROC曲线图已保存")

# ====== 保存分类报告文本 ======
with open(os.path.join(test_output_dir, "CNN_分类报告.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("分类报告已保存")
