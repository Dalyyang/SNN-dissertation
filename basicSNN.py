import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, surrogate
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# ---------- 优化参数 ----------
T_sim = 25  # 减少时间步长
batch_size = 128  # 增大批次大小
learning_rate = 0.001
num_epochs = 50  # 增加训练轮数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解决Windows上的多进程问题
num_workers = 0 if os.name == 'nt' else 4

# ---------- TensorBoard ----------
writer = SummaryWriter(log_dir='./runs/snn_mnist_hidden_optimized')
global_step = 0

# ---------- 数据增强 ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
])

# 简化测试转换
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

# 使用安全的num_workers设置
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)


# ---------- Poisson 编码器（添加缩放因子） ----------
class PoissonEncoder:
    def __init__(self, T, scale=0.8):
        self.T = T
        self.scale = scale  # 脉冲发放概率缩放因子

    def encode(self, images):  # [B, 1, 28, 28]
        B, C, H, W = images.shape
        images = images.view(B, -1).unsqueeze(0).repeat(self.T, 1, 1)  # [T, B, 784]
        rand_vals = torch.rand_like(images)
        spikes = (rand_vals < images * self.scale).float()  # 应用缩放因子
        return spikes


# ---------- 兼容版本的网络结构 ----------
class OptimizedSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.bn1 = nn.BatchNorm1d(400)  # 使用标准批归一化
        self.lif1 = neuron.LIFNode(
            tau=2.0,  # 时间常数
            v_threshold=1.0,  # 阈值
            surrogate_function=surrogate.ATan(),  # 替代梯度
            detach_reset=True
        )
        self.dropout1 = nn.Dropout(0.3)  # 使用标准Dropout

        # 第二层
        self.fc2 = nn.Linear(400, 10, bias=False)
        self.bn2 = nn.BatchNorm1d(10)
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

    def forward(self, x_spike):  # [T, B, 784]
        T, B, _ = x_spike.shape
        hidden_spikes = []
        out_spikes = []

        # 初始化膜电位
        functional.reset_net(self)

        for t in range(T):
            # 第一层
            x = self.fc1(x_spike[t])
            x = self.bn1(x)
            x = self.lif1(x)
            x = self.dropout1(x)
            hidden_spikes.append(x)  # 记录隐藏层脉冲

            # 第二层
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.lif2(x)
            out_spikes.append(x)

        # 返回输出和隐藏层脉冲（用于正则化）
        return torch.stack(out_spikes).sum(dim=0), torch.stack(hidden_spikes)


# ---------- 初始化 ----------
net = OptimizedSNN().to(device)
encoder = PoissonEncoder(T_sim, scale=0.7)  # 使用缩放因子
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = torch.optim.Adam(  # 使用标准Adam优化器
    net.parameters(),
    lr=learning_rate,
    weight_decay=1e-4  # 权重衰减
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)  # 更简单的调度器


# ---------- 发放率正则化函数 ----------
def firing_rate_regularization(spikes, target=0.25, factor=0.01):
    """
    spikes: [T, B, N] 脉冲序列
    target: 目标发放率
    factor: 正则化强度
    """
    mean_fr = spikes.mean()  # 平均发放率
    return factor * (mean_fr - target) ** 2


# ---------- 训练函数 ----------
def train(epoch):
    global global_step
    net.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        functional.reset_net(net)

        # 生成脉冲输入
        spike_input = encoder.encode(images)

        # 前向传播
        outputs, hidden_spikes = net(spike_input)

        # 计算分类损失
        loss_ce = criterion(outputs, labels)

        # 计算发放率正则化损失
        loss_reg = firing_rate_regularization(hidden_spikes, target=0.25, factor=0.01)

        # 总损失
        loss = loss_ce + loss_reg

        # 反向传播与优化
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        optimizer.step()

        # 计算准确率
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        # TensorBoard记录
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Loss/CE', loss_ce.item(), global_step)
        writer.add_scalar('Loss/Reg', loss_reg.item(), global_step)

        # 计算隐藏层平均发放率
        mean_fr = hidden_spikes.mean().item()
        writer.add_scalar('FiringRate/hidden', mean_fr, global_step)

        global_step += 1

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, Reg: {loss_reg.item():.4f}), "
                  f"Acc: {100. * correct / labels.size(0):.2f}%, "
                  f"FR: {mean_fr:.4f}")

    # 计算整个epoch的训练准确率
    train_acc = 100. * total_correct / total_samples
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    print(f"Epoch {epoch} Training Accuracy: {train_acc:.2f}%")


# ---------- 测试函数 ----------
def test(epoch):
    net.eval()
    correct = 0
    total = 0
    total_fr = 0
    test_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            functional.reset_net(net)

            # 生成脉冲输入
            spike_input = encoder.encode(images)

            # 前向传播
            outputs, hidden_spikes = net(spike_input)

            # 计算准确率
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # 计算平均发放率
            total_fr += hidden_spikes.mean().item()
            test_batches += 1

    accuracy = 100. * correct / total
    avg_fr = total_fr / test_batches

    writer.add_scalar('Accuracy/test', accuracy, epoch)
    writer.add_scalar('FiringRate/test', avg_fr, epoch)

    print(f"Test Accuracy: {accuracy:.2f}%, Avg Firing Rate: {avg_fr:.4f}")
    return accuracy


# ---------- 主循环 ----------
if __name__ == '__main__':
    best_acc = 0
    print("Starting training...")
    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")

    try:
        for epoch in range(1, num_epochs + 1):
            train(epoch)
            acc = test(epoch)

            # 更新学习率
            scheduler.step()

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), 'best_snn_model.pth')
                print(f"Saved new best model with accuracy: {best_acc:.2f}%")

        writer.close()
        print(f"Final Best Accuracy: {best_acc:.2f}%")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        writer.close()
