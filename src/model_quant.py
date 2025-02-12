import copy
import tenseal as ts
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.quantization import quantize_dynamic
from tqdm import tqdm
from quantization import *  # 确保quantization.py中有正确的量化函数实现
import argparse

class LeNet5(nn.Module):
    def __init__(self, num_classes = 10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
# 定义原始模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=None)  # 不使用预训练权重
        input_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(input_features),
            nn.Dropout(0.5),
            nn.Linear(input_features, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

# 数据加载
def load_data(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'fashionmnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# 训练函数
def train(model, train_loader, criterion, optimizer, device='cuda', epochs=10):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(train_loader))
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
    print("Training complete.")

# 测试函数
def test(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    return accuracy


def evaluate_quantized_model(original_model, test_loader, n_bits, method):
    """评估量化模型的准确率"""
    quantizer = Quantizer()
    quantized_model = copy.deepcopy(original_model)
    
    # 提取所有需要量化的层
    layers = {}
    for name, module in original_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_data = module.weight.detach().cpu().numpy()
            layers[name] = {
                'weight': weight_data,
                'shape': module.weight.shape
            }
    
    # 量化处理
    quantized_layers = {}
    for name, layer in layers.items():
        weight = layer['weight'].flatten()
        layer_max = np.max(weight)
        layer_min = np.min(weight)
        
        # 根据方法设置额外参数
        extra_params = {}
        if method == 'sigma':
            extra_params['sigma_bits'] = [n_bits] * 4  # 假设分为4个sigma区间
        elif method == 'block':
            extra_params['block_size'] = 64  # 假设块大小为64
        
        # 执行量化
        quantized_weight, params = quantizer.quantize_weights_unified(
            weight, 
            n_bits=n_bits,
            method=method,
            global_max=layer_max,
            global_min=layer_min,
            **extra_params
        )
        quantized_layers[name] = {
            'quantized_weight': quantized_weight,
            'shape': layer['shape'],
            'params': params
        }
    
    # 反量化处理
    dequantized_layers = {}
    for name, layer in quantized_layers.items():
        dequantized = quantizer.dequantize_weights_unified(
            layer['quantized_weight'], 
            layer['params']
        )
        dequantized_layers[name] = {
            'dequantized_weight': dequantized,
            'shape': layer['shape']
        }
    
    # 重建量化模型
    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name in dequantized_layers:
                dequantized = dequantized_layers[name]['dequantized_weight']
                weight = np.reshape(dequantized, dequantized_layers[name]['shape'])
                module.weight.data = torch.from_numpy(weight).float()
    
    # 测试准确率
    return test(quantized_model, test_loader, device='cpu')


def main():
    parser = argparse.ArgumentParser(description='Model Quantization')
    parser.add_argument('--model', type=str, choices=['ResNet18', 'LeNet5'], default='ResNet18')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'fashionmnist'], default='cifar100')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_classes = 10 if args.dataset in ['cifar10', 'fashionmnist'] else 100
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes).to(device)
        model_name = 'resnet18'
    elif args.model == 'LeNet5':
        model = LeNet5(num_classes=num_classes).to(device)
        model_name = 'lenet5'
        

    train_loader, test_loader = load_data(dataset=args.dataset, batch_size=256)
    learning_rate = 0.001
    epochs = 100

    save_path = f'{model_name}_{args.dataset}.pth'
    if not os.path.exists(save_path):
        print("Model not trained. Training now...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("Training the model...")
        train(model, train_loader, criterion, optimizer, device=device, epochs=epochs)
        torch.save(model.state_dict(), save_path)
    else:
        print("Model already trained. Loading from disk.")
        model.load_state_dict(torch.load(save_path))
    
    original_model = model.to('cpu')

    # 测试原始模型
    print("\nTesting original model...")
    original_acc = test(original_model, test_loader, device='cpu')
    print(f"Original Model Accuracy: {original_acc:.2f}%\n")

    # 量化参数测试
    results = []
    bits_range = range(2, 9)  # 2-8比特
    methods = ['sigma', 'block', 'naive']

    results_dict = {method: [] for method in methods}
    
    for bits in bits_range:
        for method in methods:
            print(f"Testing bits={bits}, method={method}")
            acc = evaluate_quantized_model(original_model, test_loader, bits, method)
            results_dict[method].append(acc)
            print(f"Results - Bits: {bits}, Method: {method}, Accuracy: {acc:.2f}%\n")
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # 添加大标题
    plt.suptitle(f'{model_name}_{args.dataset}', fontsize=16, y=1.05)
    
    # 子图1：所有方法对比
    plt.subplot(131)
    for method in methods:
        plt.plot(list(bits_range), results_dict[method], 
                marker='o', label=method)
    plt.title('All Methods Comparison')
    plt.xlabel('Bits')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # 子图2：与原始精度对比
    plt.subplot(132)
    for method in methods:
        acc_drop = [original_acc - acc for acc in results_dict[method]]
        plt.plot(list(bits_range), acc_drop, 
                marker='o', label=method)
    plt.title('Accuracy Drop')
    plt.xlabel('Bits')
    plt.ylabel('Accuracy Drop (%)')
    plt.grid(True)
    plt.legend()
    
    # 子图3：相对精度损失
    plt.subplot(133)
    for method in methods:
        relative_loss = [(original_acc - acc)/original_acc * 100 
                        for acc in results_dict[method]]
        plt.plot(list(bits_range), relative_loss, 
                marker='o', label=method)
    plt.title('Relative Accuracy Loss')
    plt.xlabel('Bits')
    plt.ylabel('Relative Loss (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{args.dataset}_quantization_results.png")
    
    # 保存数值结果
    with open('quantization_results.csv', 'w') as f:
        f.write('Bits,Original,' + ','.join(methods) + '\n')
        for i, bits in enumerate(bits_range):
            row = [str(bits), str(original_acc)]
            for method in methods:
                row.append(str(results_dict[method][i]))
            f.write(','.join(row) + '\n')



if __name__ == '__main__':
    main()