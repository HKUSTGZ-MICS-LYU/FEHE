# Standard libraries
import time
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
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

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
    
    # 通用基础增强组合（可根据数据集特性调整）
    base_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    ])

    # 针对不同数据集的增强策略
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            base_transform,
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.RandomGrayscale(p=0.1),  # 随机灰度化
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # 高斯模糊
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # 随机擦除
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x)*0.05)], p=0.5)  # 添加噪声
        ])
        
    elif dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            base_transform,
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # 自动增强
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
    elif dataset_name == 'FASHIONMNIST':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 仿射变换
            transforms.ElasticTransform(alpha=20.0),  # 弹性变形
            transforms.RandomVerticalFlip(p=0.3),  # 垂直翻转
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x)*0.1)], p=0.5)
        ])
        base_transform = None  # 不应用基础组合中的裁剪

    # 测试集统一处理（保持原始数据）
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)) if dataset_name == 'FASHIONMNIST' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465) if dataset_name == 'CIFAR10' else 
            (0.5071, 0.4867, 0.4408) if dataset_name == 'CIFAR100' else 
            (0.2860,), 
            (0.2023, 0.1994, 0.2010) if dataset_name == 'CIFAR10' else 
            (0.2675, 0.2565, 0.2761) if dataset_name == 'CIFAR100' else 
            (0.3530,))
    ])

    # 加载数据集
    dataset_class = {
        'CIFAR10': torchvision.datasets.CIFAR10,
        'CIFAR100': torchvision.datasets.CIFAR100,
        'FASHIONMNIST': torchvision.datasets.FashionMNIST
    }
    
    trainset = dataset_class[dataset_name](
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    testset = dataset_class[dataset_name](
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # 调整类别数
    num_classes = 100 if dataset_name == 'CIFAR100' else 10
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return trainset, testloader, num_classes

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def distribute_data(trainset, num_clients, iid=True):
    """将数据集分配给客户端，支持IID和Non-IID分布"""
    if iid:
        # IID分配：随机均匀分配
        # indices = np.random.permutation(len(trainset))
        # client_data_indices = np.array_split(indices, num_clients)
        
        indices = np.arange(len(trainset))  
        client_data_indices = [indices for _ in range(num_clients)]
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
    def __init__(self, client_id, model, data_indices, trainset, device):
        """客户端初始化"""
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.data_indices = data_indices
        self.trainset = trainset
        self.trainloader = DataLoader(Subset(trainset, data_indices), batch_size=128, shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.device = device
        self.accuracy = {}

    def train(self, criterion, epochs=1):
        """客户端本地训练"""
        self.model.train()
        for _ in range(epochs):
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                self.optimizer.step()
                
    def update_global_model(self, aggregated_weights):
        """更新全局模型"""
        current_dict = self.model.state_dict()
        for name in current_dict:
            if name in aggregated_weights:
                current_dict[name] = aggregated_weights[name].to(self.device)
        self.model.load_state_dict(current_dict)
            
def encrypt_weights(weights_dict, context, chunk_size=4096):
    """使用BFV加密权重"""
    encrypted_weights = {}
    
    for name, weights in weights_dict.items():
        encrypted_weights[name] = []
        
        for weight in weights:
            # 将权重转换为一维数组
            flattened = weight.flatten()
            encrypted_chunks = []
            
            # 如果权重大于chunk_size，分块处理
            if len(flattened) > chunk_size:
                # 计算需要多少块
                num_chunks = (len(flattened) + chunk_size - 1) // chunk_size
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(flattened))
                    chunk = flattened[start_idx:end_idx]
                    
                    # 使用BFV加密每个块
                    encrypted_chunk = ts.bfv_vector(context, chunk)
                    encrypted_chunks.append(encrypted_chunk)
            else:
                # 如果权重小于chunk_size，直接加密
                encrypted_chunk = ts.bfv_vector(context, flattened)
                encrypted_chunks.append(encrypted_chunk)
            
            # 保存加密后的数据和原始形状信息
            encrypted_weights[name].append(encrypted_chunks)
            
    return encrypted_weights

def decrypt_weights(aggregated_encrypted_weights, context):
    """使用BFV解密权重"""
    decrypt_weights = {}
    for name, weights in aggregated_encrypted_weights.items():
        decrypted_weights = []
        for weight in weights:
            decrypted = weight.decrypt(context.secret_key())
            decrypted_weights.extend(decrypted)
        decrypt_weights[name] = np.array(decrypted_weights)
    return decrypt_weights

def quantize_weights(
        selected_weights, # 选中客户端的权重
        quantizer, # 量化器
        n_bits=8, # 量化位数
        method='sigma' # 量化方法
):
    """量化权重"""
    need_quantized_weights = {}
    nonquantized_weights = {}
    quantized_weights = {}
    quantized_weights_param = {}
    
    # 首先对需要量化和不需要量化的权重进行分类
    for name, weights in selected_weights.items():
        if _is_quantizable(name):
            need_quantized_weights[name] = weights
        else:
            nonquantized_weights[name] = weights
            
    # 之后逐层来处理需要量化的
    for name, weights in need_quantized_weights.items():   
        flattened_weights = np.array([param.cpu().flatten() for param in weights]).flatten()
        
        #  要先找到最大值，最小值，均值，方差
        global_max = np.max(flattened_weights)
        global_min = np.min(flattened_weights)
        global_mu = np.mean(flattened_weights)
        global_sigma = np.std(flattened_weights)
    
 
        # 量化
        for weight in weights:
            q_weight, q_param = quantizer.quantize_weights_unified(
                np.array(weight.cpu().flatten()),
                n_bits = n_bits,
                sigma_bits = [n_bits] * 3,
                method = method,
                global_max = global_max,
                global_min = global_min,
                global_mu = global_mu,
                global_sigma = global_sigma
            )
            if name not in quantized_weights:
                quantized_weights[name] = []
                quantized_weights_param[name] = []
                
            quantized_weights[name].append(q_weight)
            quantized_weights_param[name].append(q_param)
    
    return quantized_weights, quantized_weights_param, nonquantized_weights
            
            
  
    
def _is_quantizable(name: str) -> bool:
    """
    decide whether to quantize the layer
    only quantize the weights and biases of the convolutional layer and the fully connected layer
    """
    
    # identify target layers
    is_target_layer = any(key in name.lower() for key in ('conv', 'fc', 'linear'))
    # identify weights and biases
    is_weight_or_bias = any(key in name for key in ('weight', 'bias'))
    # exclude batch normalization layers
    is_excluded = any(key in name for key in ('bn', 'batch', 'running_', 'num_batches'))
    
    return is_target_layer and is_weight_or_bias and not is_excluded
            


def dequantize_weights(quantizer, decrypted_weights, quantized_weights_param):
    """反量化权重"""
    dequantize_weights = {}
    for name, weights in decrypted_weights.items():
        dequantized_weights = quantizer.dequantize_weights_unified(weights, quantized_weights_param[name][0])
        dequantize_weights[name] = np.array(dequantized_weights)
    return dequantize_weights
        

def collect_weights_by_layer(client_models):
    weights_dict = {}
    # 遍历每个客户端
    for client_id, model in client_models.items():
        # 遍历客户端的每个参数
        for param_name, param_value in model.items():
            # 仅处理权重参数（以 .weight 结尾）
            if param_name not in weights_dict:
                weights_dict[param_name] = []
            weights_dict[param_name].append(param_value)
    return weights_dict    

### 联邦学习工具函数 ###
def aggregate_weights(encrypted_weights, non_quantized_params):
    """聚合客户端权重"""
    aggregated_encrypted_weights = {}
    aggregated_non_encrypted_weights = {}
    
    # 处理加密的权重
    for name, params in encrypted_weights.items():
        if name not in aggregated_encrypted_weights:
            aggregated_encrypted_weights[name] = np.array(params)
        else:
            aggregated_encrypted_weights[name] += np.array(params)
                    
    # 处理未加密的权重
    for name, param in non_quantized_params.items():
        if name not in aggregated_non_encrypted_weights:
            aggregated_non_encrypted_weights[name] = param
        else:
            aggregated_non_encrypted_weights[name] += param
            
    return aggregated_encrypted_weights, aggregated_non_encrypted_weights



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

def train_client(client, criterion, epochs):
    client.train(criterion, epochs=epochs)
    
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
    polynomial_degree = 4096
    
    # print the parameters of FL overall scheme
    print('=' * 50)
    print(f'{"FL Training Configuration":^50}')
    print('=' * 50)
    print(f'{"Dataset:":<20} {dataset_name}')
    print(f'{"Model:":<20} {model_name}')
    print(f'{"Total Clients:":<20} {num_clients}')
    print(f'{"Selected Clients:":<20} {select_num_clients}')
    print(f'{"Rounds:":<20} {num_rounds}')
    print(f'{"Local Epochs:":<20} {local_epochs}')
    print(f'{"IID Distribution:":<20} {iid}')
    print(f'{"Quant. Bits:":<20} {n_bits}')
    print(f'{"Quant. Method:":<20} {method}')
    print(f'{"Device:":<20} {device}')
    print(f'{"Polynomial Degree:":<20} {polynomial_degree}')
    print('=' * 50)
    
    

    # 设置quantization参数
    quantizer = Quantizer()
    
    # 数据分配和客户端初始化
    client_data_indices = distribute_data(trainset, num_clients, iid)
    clients = [Client(i, global_model, indices, trainset, device) for i, indices in enumerate(client_data_indices)]
    
    criterion = nn.CrossEntropyLoss()
    mean_clients_acc = []
    global_acc = []

    # 联邦学习循环
    for round in tqdm(range(num_rounds), desc="Federated Learning Rounds"):
        # 所有的客户端参与训练
     
            
        for client in clients:
            train_client(client, criterion, epochs=local_epochs)
            client_acc = test_model(client.model, testloader, device)
            client.accuracy[round] = client_acc
            
        
        if select_num_clients > num_clients:
            raise ValueError("Number of selected clients should be less than total number of clients.")
        
        # 每次随机种子
        random.seed(time.time())
        
        # 随机选择 select_num_clients 个客户端
        selected_clients = random.sample(clients, select_num_clients)
        
        # 收集选中客户端的权重
        selected_weights = {}
        for client in selected_clients:
            selected_weights[client.client_id] = {name: param.data for name, param in client.model.named_parameters()}
        
        selected_weights = collect_weights_by_layer(selected_weights)
        
        # 量化选中客户端的权重
        quantized_weights, quantized_weights_param, non_quantized_params = quantize_weights(selected_weights, quantizer, n_bits=n_bits, method=method)
        
        
        # 聚合加密的权重
        aggregated_quantized_weights, aggregated_non_encrypted_weights = aggregate_weights(quantized_weights, non_quantized_params)
        
        
        # 用选中的数量进行平均模型
        dequantized_weights = {name: param / select_num_clients for name, param in aggregated_quantized_weights.items()}
        aggregated_non_encrypted_weights = {name: param / select_num_clients for name, param in aggregated_non_encrypted_weights.items()}
        
        # 反量化权重
        dequantized_weights = dequantize_weights(quantizer, dequantized_weights, quantized_weights_param)
        
        # 根据dequantized_weights和aggregated_non_encrypted_weights构造新的全局模型
        new_global_model = copy.deepcopy(global_model)
        device = next(global_model.parameters()).device  # 获取模型当前的device
        
        for name, param in new_global_model.named_parameters():
            shape = param.data.shape
            if name in dequantized_weights:
                param.data = torch.from_numpy(dequantized_weights[name].reshape(shape)).to(device)
            else:
                param.data = torch.from_numpy(aggregated_non_encrypted_weights[name].reshape(shape)).to(device)
                
        # 更新全局模型
        global_model.load_state_dict(new_global_model.state_dict())
        test_acc = test_model(global_model, testloader, device)
        global_acc.append(test_acc)
        
        
        # 更新客户端模型
        for client in selected_clients:
            client.update_global_model(global_model.state_dict())
            
        # 评估mean_clients_acc
        clients_acc = []
        for client in clients:
            clients_acc.append(test_model(client.model, testloader, device))
        mean_acc = np.mean(clients_acc)
        mean_clients_acc.append(mean_acc)
        

    # 绘制测试准确率
    plot_test_accuracy(clients, global_acc, mean_clients_acc, model_name, dataset_name, num_clients, select_num_clients, iid)
    
def plot_test_accuracy(clients, global_acc, mean_clients_acc, model_name, dataset_name, num_clients, select_num_clients, iid):
    """Plot test accuracy metrics in three subplots and save metrics to separate file"""
    # Save metrics to file
    os.makedirs('results', exist_ok=True)
    metrics_file = f'results/metrics_{model_name}_{dataset_name}_C{num_clients}_S{select_num_clients}_{iid}.txt'
    with open(metrics_file, 'w') as f:
        f.write("=== Individual Client Accuracies ===\n")
        for client in clients:
            f.write(f"Client {client.client_id}: {list(client.accuracy.values())}\n")
            
        f.write("\n=== Global Model Accuracy ===\n")
        f.write(f"Rounds: {list(range(len(global_acc)))}\n")
        f.write(f"Accuracies: {global_acc}\n")
        
        f.write("\n=== Mean Client Accuracy ===\n")
        f.write(f"Rounds: {list(range(len(mean_clients_acc)))}\n")
        f.write(f"Accuracies: {mean_clients_acc}\n")

    # Create visualization plots
    plt.figure(figsize=(20, 6))
    
    # Plot individual client accuracies
    plt.subplot(1, 3, 1)
    for client in clients:
        rounds = list(client.accuracy.keys())
        accuracies = list(client.accuracy.values())
        plt.plot(rounds, accuracies, alpha=0.3)
    plt.title(f'Client Accuracies\n{model_name} on {dataset_name}')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot global model accuracy
    plt.subplot(1, 3, 2)
    plt.plot(global_acc, 'r-', linewidth=2, label='Global Model')
    plt.title(f'Global Model Accuracy\n{model_name} on {dataset_name}')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot mean client accuracy
    plt.subplot(1, 3, 3)
    plt.plot(mean_clients_acc, 'g-', linewidth=2, label='Mean Client Accuracy')
    plt.title(f'Average Client Accuracy\n{model_name} on {dataset_name}')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.savefig(f'results/plot_{model_name}_{dataset_name}_C{num_clients}_S{select_num_clients}_R{rounds}_{iid}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

    


### 主函数 ###
def main():
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--model', type=str, choices=['LeNet5', 'ResNet18', 'VGG19', 'PreActResNet18', 'GoogLeNet', 
                                                     'DenseNet121', 'ResNeXt29_2x64d', 'MobileNet', 'MobileNetV2', 
                                                     'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 
                                                     'RegNetX_200MF', 'SimpleDLA'], default='LeNet5')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'FASHIONMNIST'], default='FASHIONMNIST')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--select_num_clients', type=int, default=4)
    parser.add_argument('--iid', default = True, action='store_true', help='Use IID data distribution')
    parser.add_argument('--n_bits', type=int, default=8, help='Bits for quantization')
    parser.add_argument('--method', type=str, default='naive', help='Quantization method')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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