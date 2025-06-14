import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

from models.cnn import CNN  # 确保模型在 models/cnn.py 中

# 设置中文字体为黑体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ===== 实验参数配置 =====
USE_GPU = True          # 是否使用GPU
learning_rate = 0.001   # 学习率
epochs = 20             # 训练轮次
optimizer_name = "Adam" # 可选："Adam", "SGD", "RMSprop"

# ===== 设备设置 =====
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用的设备是:", device)
try:
    if device.type == "cuda":
        print("当前设备名称:", torch.cuda.get_device_name(0))
    else:
        print("当前设备名称: 无 GPU")
except:
    print("当前设备名称: GPU 无法访问或未初始化")

# ===== 数据预处理 =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ===== 输出目录设置 =====
device_name = "GPU" if USE_GPU else "CPU"
output_dir = f"output/cnn_{optimizer_name}_lr{learning_rate}_ep{epochs}_{device_name}"
os.makedirs(output_dir, exist_ok=True)

# ===== 模型初始化与训练准备 =====
model = CNN().to(device)

# 选择优化器
if optimizer_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
elif optimizer_name == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
else:
    raise ValueError("不支持的优化器类型，请选择 'Adam', 'SGD' 或 'RMSprop'")

criterion = nn.CrossEntropyLoss()

train_loss_list = []
val_loss_list = []
start_time = time.time()

# ===== 训练过程 =====
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_loss_list.append(epoch_loss)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_epoch_loss = val_loss / len(val_loader)
    val_loss_list.append(val_epoch_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

# ===== 保存模型与绘图 =====
duration = time.time() - start_time
print(f"训练总时长：{duration:.2f} 秒")

torch.save(model.state_dict(), os.path.join(output_dir, "cnn.pth"))
print("模型训练完成，已保存至：", os.path.join(output_dir, "cnn.pth"))

plt.figure()
plt.plot(range(1, epochs + 1), train_loss_list, marker='o', label='训练损失')
plt.plot(range(1, epochs + 1), val_loss_list, marker='s', label='验证损失')
plt.xlabel("轮次（Epoch）")
plt.ylabel("损失（Loss）")
plt.title(f"CNN训练与验证损失曲线（{optimizer_name}, lr={learning_rate}, ep={epochs}, {device_name}）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "CNN训练验证损失曲线.png"))
print("损失曲线已保存至：", os.path.join(output_dir, "CNN训练验证损失曲线.png"))
