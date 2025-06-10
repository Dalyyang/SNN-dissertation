
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, surrogate
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

T_sim = 25
batch_size = 128
learning_rate = 0.001
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0 if os.name == 'nt' else 4

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
writer = SummaryWriter(log_dir='./runs/snn_prune_debug')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)

class PoissonEncoder:
    def __init__(self, T, scale=0.7):
        self.T = T
        self.scale = scale

    def encode(self, images):
        B, C, H, W = images.shape
        images = images.view(B, -1).unsqueeze(0).repeat(self.T, 1, 1)
        return (torch.rand_like(images) < images * self.scale).float()

class OptimizedSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.bn1 = nn.BatchNorm1d(400)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(400, 10, bias=False)
        self.bn2 = nn.BatchNorm1d(10)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)

    def forward(self, x_spike):
        T, B, _ = x_spike.shape
        functional.reset_net(self)
        hidden_spikes = []
        out_spikes = []
        for t in range(T):
            x = self.fc1(x_spike[t])
            x = self.bn1(x)
            x = self.lif1(x)
            x = self.dropout1(x)
            hidden_spikes.append(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.lif2(x)
            out_spikes.append(x)
        return torch.stack(out_spikes).sum(dim=0), torch.stack(hidden_spikes)

net = OptimizedSNN().to(device)
encoder = PoissonEncoder(T_sim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def firing_rate_regularization(spikes, target=0.25, factor=0.01):
    return factor * (spikes.mean() - target) ** 2

def train(epoch):
    net.train()
    total_correct = 0
    global_step = epoch * len(train_loader)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        spike_input = encoder.encode(images)
        outputs, hidden_spikes = net(spike_input)
        loss = criterion(outputs, labels) + firing_rate_regularization(hidden_spikes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        writer.add_scalar('Loss/train', loss.item(), global_step + batch_idx)
    acc = 100. * total_correct / len(train_loader.dataset)
    writer.add_scalar('Accuracy/train', acc, epoch)
    print(f"[Train] Epoch {epoch}: Acc = {acc:.2f}%")

def test(epoch):
    net.eval()
    correct, total_fr = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            spike_input = encoder.encode(images)
            outputs, hidden_spikes = net(spike_input)
            assert outputs.shape[1] == 10, "fc2输出节点数量错误，应为10"
            assert labels.min() >= 0 and labels.max() < 10, "标签越界"
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_fr += hidden_spikes.mean().item()
    acc = 100. * correct / len(test_loader.dataset)
    fr = total_fr / len(test_loader)
    writer.add_scalar('Accuracy/test', acc, epoch)
    writer.add_scalar('FiringRate/test', fr, epoch)
    print(f"[Test] Epoch {epoch}: Acc = {acc:.2f}%, FR = {fr:.4f}")
    return acc

def activity_based_pruning(net, dataloader, encoder, threshold=0.01, device='cuda'):
    net.eval()
    with torch.no_grad():
        total_fr = torch.zeros(400, device=device)
        for images, _ in dataloader:
            images = images.to(device)
            spikes = encoder.encode(images)
            _, hidden = net(spikes)
            total_fr += hidden.mean(dim=(0, 1))
        avg_fr = total_fr / len(dataloader)
        prune_idx = (avg_fr < threshold).nonzero(as_tuple=True)[0]

        if len(prune_idx) > 350:
            print("⚠️ 剪枝太狠，跳过")
            return torch.tensor([], dtype=torch.long)

        prune_idx = prune_idx[prune_idx < 400]  # 保险起见过滤

        with torch.no_grad():
            net.fc1.weight[:, prune_idx] = 0
            net.fc2.weight[:, prune_idx] = 0  # 修复：用列索引

        return prune_idx


def multi_round_pruning_finetune(net, train_loader, test_loader, encoder, device,
                                  max_rounds=5, min_neurons=50, threshold=0.01, finetune_epochs=5):
    pruning_step = 0
    remaining_neurons = 400
    for r in range(max_rounds):
        print(f"=== Prune Round {r+1}/{max_rounds} ===")
        if os.path.exists('best_snn_model.pth'):
            net.load_state_dict(torch.load('best_snn_model.pth'))
        acc_before = test(1000 + pruning_step)
        writer.add_scalar('Accuracy/prune_before', acc_before, pruning_step)
        pruned = activity_based_pruning(net, train_loader, encoder, threshold, device)
        remaining_neurons -= len(pruned)
        writer.add_scalar('NeuronCount/remaining', remaining_neurons, pruning_step)
        if remaining_neurons <= min_neurons:
            print("⚠️ 神经元太少，停止剪枝")
            break
        try:
            acc_after = test(1000 + pruning_step)
            writer.add_scalar('Accuracy/prune_after', acc_after, pruning_step)
        except Exception as e:
            print(f"⚠️ 测试失败：{e}")
            break
        for epoch in range(1, finetune_epochs + 1):
            train(epoch)
            acc = test(epoch)
            writer.add_scalar('Accuracy/prune_finetune', acc, pruning_step * 10 + epoch)
        torch.save(net.state_dict(), 'best_snn_model.pth')
        pruning_step += 1

if __name__ == '__main__':
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        acc = test(epoch)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), 'best_snn_model.pth')
            print(f"✅ New best model saved: {best_acc:.2f}%")
    print("\n=== 启动多轮剪枝 + 微调 ===")
    multi_round_pruning_finetune(
        net, train_loader, test_loader, encoder,
        device=device, max_rounds=5, min_neurons=50,
        threshold=0.01, finetune_epochs=5
    )
    writer.close()
