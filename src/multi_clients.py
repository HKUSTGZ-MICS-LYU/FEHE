# Standard libraries
from tqdm import tqdm
import argparse
import copy
import os
import random

# Data processing and visualization
import numpy as np
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

# PyTorch vision
import torchvision
import torchvision.transforms as transforms

# Local imports
from models import *
from utils.quantization import *

# Encrypted computation
import tenseal as ts

### 数据加载和分配 ###
def load_dataset(dataset_name, batch_size=128):
    """加载数据集并返回训练集，测试集和类别数"""
    num_classes = 10
    
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    elif dataset_name == 'FASHIONMNIST':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainset, testloader, num_classes

def distribute_data(trainset, num_clients, iid=True):
    """将数据集分配给客户端，支持IID和Non-IID分布"""
    if iid:
        # IID分配：随机均匀分配
        indices = np.random.permutation(len(trainset))
        client_data_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID分配：按标签排序后分配，每个客户端拥有部分类别的数据
        labels = np.array(trainset.targets)
        sorted_indices = np.argsort(labels)
        client_data_indices = []
        for i in range(num_clients):
            start = i * len(trainset) // num_clients
            end = (i + 1) * len(trainset) // num_clients
            client_data_indices.append(sorted_indices[start:end])
    return client_data_indices

### 客户端类 ###
class Client:
    def __init__(self, client_id, model, data_indices, trainset, device, context):
        """客户端初始化"""
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.data_indices = data_indices
        self.trainset = trainset
        self.trainloader = DataLoader(Subset(trainset, data_indices), batch_size=128, shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.device = device
        self.context = context

    def train(self, criterion, epochs=1):
        """客户端本地训练"""
        self.model.train()
        for _ in range(epochs):
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
    def update_global_model(self, aggregated_weights):
        """更新全局模型"""
        for name, param in self.model.named_parameters():
            param.data = aggregated_weights[name]
            
def encrypt_weights():
    """使用BFV加密权重"""
    encrypt_weights = {}
    pass

def decrypt_weights():
    """使用BFV解密权重"""
    decrypt_weights = {}
    pass

def quantize_weights():
    """量化权重"""
    quantize_weights = {}
    pass

def dequantize_weights():
    """反量化权重"""
    dequantize_weights = {}
    pass
        

### 联邦学习工具函数 ###
def aggregate_weights(encrypted_weights_list):
    """聚合客户端权重"""
    aggregated_weights = {name: torch.zeros_like(param.data) for name, param in encrypted_weights_list[0].model.named_parameters()}
    num_selected = len(encrypted_weights_list)
    for client in encrypted_weights_list:
        for name, param in client.model.named_parameters():
            aggregated_weights[name] += param.data / num_selected
    return aggregated_weights



def test_model(model, testloader, device):
    """测试模型准确率"""
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total



### 联邦学习主函数 ###
def federated_learning(
    global_model, 
    trainset, 
    testloader, 
    num_clients=50, 
    select_num_clients=20,
    num_rounds=100, 
    local_epochs=1, 
    iid=True, 
    n_bits=8, 
    method='sigma', 
    device='cpu', 
    model_name='Model', 
    dataset_name='Dataset'
    ):
    
    context = ts.context(
        ts.SCHEME_TYPE.BFV, 
        poly_modulus_degree=4096, 
        plain_modulus=1032193
        )
    context.generate_galois_keys()
    context.global_scale = 2**40  # 设置精度参数
    
    # 数据分配和客户端初始化
    client_data_indices = distribute_data(trainset, num_clients, iid)
    clients = [Client(i, global_model, indices, trainset, device, context) for i, indices in enumerate(client_data_indices)]
    
    criterion = nn.CrossEntropyLoss()
    test_accs = []

    # 联邦学习循环
    for round in tqdm(range(num_rounds), desc="Federated Learning Rounds"):
        # 随机选择20个客户端
        selected_clients = random.sample(clients, select_num_clients)
        
        # 本地训练
        for client in selected_clients:
            client.train(criterion, epochs=local_epochs)
        
        # 收集和加密权重
        key = torch.randn(1).to(device)  # 模拟加密密钥
        encrypted_weights_list = [encrypt_weights({name: param.data for name, param in client.model.named_parameters()}, key) 
                                for client in selected_clients]
        
        # 聚合加密的权重
        aggregated_encrypted_weights = aggregate_weights(encrypted_weights_list)
        
        # 解密聚合的权重
        aggregated_weights = decrypt_weights(aggregated_encrypted_weights, key)
        
        # 更新全局模型
        for client in selected_clients:
            client.update_global_model(aggregated_weights)
            
        # 评估全局模型
        clients_acc = []
        for client in clients:
            clients_acc.append(test_model(client.model, testloader, device))
        
        # 评估全局模型
        global_acc = np.mean(clients_acc)
        test_accs.append(global_acc)
        print(f"\nRound {round+1}/{num_rounds}: Test Acc: {global_acc:.2f}%")
        

    


### 主函数 ###
def main():
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--model', type=str, choices=['LeNet5', 'ResNet18', 'VGG19', 'PreActResNet18', 'GoogLeNet', 
                                                     'DenseNet121', 'ResNeXt29_2x64d', 'MobileNet', 'MobileNetV2', 
                                                     'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 
                                                     'RegNetX_200MF', 'SimpleDLA'], default='LeNet5')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'FASHIONMNIST'], default='FASHIONMNIST')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--num_clients', type=int, default=1)
    parser.add_argument('--select_num_clients', type=int, default=1)
    parser.add_argument('--iid', action='store_true', help='Use IID data distribution')
    parser.add_argument('--n_bits', type=int, default=8, help='Bits for quantization')
    parser.add_argument('--method', type=str, default='sigma', help='Quantization method')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('='*50)
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model}')
    print(f'Learning rate: {args.lr}')
    print(f'Rounds: {args.rounds}')
    print(f'Local epochs: {args.local_epochs}')
    print(f'Number of clients: {args.num_clients}')
    print(f'Data distribution: {"IID" if args.iid else "Non-IID"}')
    print(f'Device: {device}')
    print(f'Quantization: {args.n_bits} bits, method: {args.method}')
    print('='*50)

    # 加载数据集
    trainset, testloader, num_classes = load_dataset(args.dataset)

    # 初始化全局模型
    model_dict = {
        'LeNet5': lambda: LeNet5(num_classes=num_classes),
        'ResNet18': ResNet18,
        'VGG19': lambda: VGG('VGG19'),
        'PreActResNet18': PreActResNet18,
        'GoogLeNet': GoogLeNet,
        'DenseNet121': DenseNet121,
        'ResNeXt29_2x64d': ResNeXt29_2x64d,
        'MobileNet': MobileNet,
        'MobileNetV2': MobileNetV2,
        'DPN92': DPN92,
        'ShuffleNetG2': ShuffleNetG2,
        'SENet18': SENet18,
        'ShuffleNetV2': lambda: ShuffleNetV2(1),
        'EfficientNetB0': EfficientNetB0,
        'RegNetX_200MF': RegNetX_200MF,
        'SimpleDLA': SimpleDLA
    }
    global_model = model_dict[args.model]().to(device)
    
    if device == 'cuda':
        global_model = torch.nn.DataParallel(global_model)
        cudnn.benchmark = True

    # 启动联邦学习
    federated_learning(
        global_model=global_model,
        trainset=trainset,
        testloader=testloader,
        num_clients=args.num_clients,
        select_num_clients=args.select_num_clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        iid=args.iid,
        n_bits=args.n_bits,
        method=args.method,
        device=device,
        model_name=args.model,
        dataset_name=args.dataset
    )
    print('Federated learning complete.')

if __name__ == '__main__':
    main()