import matplotlib.pyplot as plt
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
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'fashionmnist'], default='cifar10')
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
    bits = 8
    method = 'sigma'

    original_acc_stat = []
    quanti_acc_stat = []

    save_path = f'{model_name}_{args.dataset}.pth'
    if not os.path.exists(save_path):
        print("Model not trained. Training now...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("Training the model...")
        for epoch in range(epochs):
            train(model, train_loader, criterion, optimizer, device=device, epochs=1)
            original_acc = test(model, test_loader, device=device)
            original_acc_stat.append(original_acc)
            
            original_model = copy.deepcopy(model)
            quanti_acc = evaluate_quantized_model(original_model, test_loader, bits, method)
            quanti_acc_stat.append(quanti_acc)
            
            print(f"Epoch {epoch + 1}, Original Model Accuracy: {original_acc:.2f}%, Quantized Model Accuracy: {quanti_acc:.2f}%")
            
        
        torch.save(model.state_dict(), save_path)
    else:
        print("Model already trained. Loading from disk.")
        model.load_state_dict(torch.load(save_path))
    
    original_model = model.to('cpu')


    print("\nTesting original model...")
    original_acc = test(original_model, test_loader, device='cpu')
    print(f"Original Model Accuracy: {original_acc:.2f}%\n")
    
    print("\nTesting quantized model...")
    quanti_acc = evaluate_quantized_model(original_model, test_loader, bits, method)
    print(f"Quantized Model Accuracy: {quanti_acc:.2f}%\n")
    
    # Plot the original and quantized model accuracy

    plt.figure(figsize=(10, 5))
    plt.plot(original_acc_stat, label='Original Model')
    plt.plot(quanti_acc_stat, label='Quantized Model')
    
    # Add maximum accuracy labels
    max_orig_acc = max(original_acc_stat)
    max_quant_acc = max(quanti_acc_stat)
    
    plt.annotate(f'Max: {max_orig_acc:.2f}%', 
                xy=(original_acc_stat.index(max_orig_acc), max_orig_acc),
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Max: {max_quant_acc:.2f}%',
                xy=(quanti_acc_stat.index(max_quant_acc), max_quant_acc),
                xytext=(10, -10), textcoords='offset points')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} on {args.dataset}')
    plt.legend()
    plt.savefig(f'{model_name}_{args.dataset}.png')
    
    



  

 

if __name__ == '__main__':
    main()