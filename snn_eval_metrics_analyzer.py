import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, surrogate
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
# 动态剪枝相关函数
from snn_train_with_pruning import activity_based_pruning, multi_round_pruning_finetune

# ---------- 配置参数 ----------
T_sim = 25
batch_size = 128
learning_rate = 0.001
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0 if os.name == 'nt' else 4
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------- TensorBoard ----------
writer = SummaryWriter(log_dir='./runs/snn_prune_metrics')

# ---------- 数据加载 ----------
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

# ---------- 编码器 ----------
class PoissonEncoder:
    def __init__(self, T, scale=0.7):
        self.T = T
        self.scale = scale
    def encode(self, images):
        B, C, H, W = images.shape
        imgs = images.view(B, -1).unsqueeze(0).repeat(self.T, 1, 1)
        return (torch.rand_like(imgs) < imgs * self.scale).float()

# ---------- 模型定义 ----------
class OptimizedSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.bn1 = nn.BatchNorm1d(400)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0,
                                  surrogate_function=surrogate.ATan(), detach_reset=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(400, 10, bias=False)
        self.bn2 = nn.BatchNorm1d(10)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0,
                                  surrogate_function=surrogate.ATan(), detach_reset=True)
    def forward(self, x_spike):
        T, B, _ = x_spike.shape
        functional.reset_net(self)
        out_spikes, hidden_spikes = [], []
        for t in range(T):
            x = self.fc1(x_spike[t]); x = self.bn1(x); x = self.lif1(x); x = self.dropout1(x)
            hidden_spikes.append(x)
            x = self.fc2(x); x = self.bn2(x); x = self.lif2(x)
            out_spikes.append(x)
        return torch.stack(out_spikes).sum(dim=0), torch.stack(hidden_spikes)

# ---------- 指标函数 ----------
def compute_params_and_sparsity(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    sparsity = 1 - nonzero/total
    return total, nonzero, sparsity

def compute_macs(model, input_shape=(1,784)):
    macs, _ = get_model_complexity_info(model, input_shape, as_strings=False,
                                        print_per_layer_stat=False)
    return macs

def measure_gpu_latency(model, input_shape=(1,784), runs=100):
    model.eval()
    dummy = torch.randn((1,)+input_shape).to(device)
    # 预热
    for _ in range(10): _ = model(dummy)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs): _ = model(dummy)
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end)/runs

def estimate_quantized_size(model, bit_width=8):
    # 简易方式：直接保存 state_dict 并测文件大小
    path = f'model_int{bit_width}.pth'
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path)
    # 删除临时文件
    try: os.remove(path)
    except: pass
    return size

# ---------- 初始化 ----------
encoder = PoissonEncoder(T_sim)
net = OptimizedSNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---------- 预计算并记录初始指标 ----------
total, nonzero, sp = compute_params_and_sparsity(net)
macs = compute_macs(net)
gpu_lat = measure_gpu_latency(net)
qt8_size = estimate_quantized_size(net, 8)
writer.add_scalars('InitMetrics', {
    'Params': total,
    'Sparsity': sp,
    'MACs': macs,
    'GPU_Latency_ms': gpu_lat,
    'Size_INT8_bytes': qt8_size
}, 0)
print(f"Init: Params={total}, Sparsity={sp:.2f}, MACs={macs}, "
      f"GPU_Latency={gpu_lat:.2f}ms, INT8 Size={qt8_size/1e6:.2f}MB")

# ---------- 正常训练与测试 ----------
def train(epoch):
    net.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        spikes = encoder.encode(images)
        outputs, _ = net(spikes)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    writer.add_scalar('Train/Loss', avg_loss, epoch)


def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            spikes = encoder.encode(images)
            outputs, _ = net(spikes)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    writer.add_scalar('Test/Accuracy', acc, epoch)
    return acc

# ---------- 主流程 ----------
if __name__ == '__main__':
    best_acc = 0
    for epoch in range(1, num_epochs+1):
        train(epoch)
        acc = test(epoch)
        scheduler.step()
        # 记录每 epoch 后指标
        total, nonzero, sp = compute_params_and_sparsity(net)
        macs = compute_macs(net)
        gpu_lat = measure_gpu_latency(net)
        writer.add_scalars('EpochMetrics', {
            'Sparsity': sp,
            'MACs': macs,
            'GPU_Latency_ms': gpu_lat
        }, epoch)
        # 打印每轮测试准确率和稀疏度
        print(f"Epoch {epoch}: Test Acc: {acc*100:.2f}%, Sparsity: {sp*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), 'best_snn_model.pth')
    # 保持原有剪枝与微调流程
    multi_round_pruning_finetune(net, train_loader, test_loader, encoder,
                                  device=device, max_rounds=5,
                                  min_neurons=50, threshold=0.01,
                                  finetune_epochs=5)
    writer.close()
