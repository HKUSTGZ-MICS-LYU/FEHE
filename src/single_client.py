# Standard libraries
from tqdm import tqdm
import argparse
import copy
import os

# Data processing and visualization
import numpy as np
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# PyTorch vision
import torchvision
import torchvision.transforms as transforms

# Local imports
from models import *
from models.alexnet import AlexNet
from utils.quantization import *


def is_quantizable(name: str) -> bool:
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

def load_dataset(dataset_name, batch_size=128):
    transform = transforms.Compose([])
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
        transform.transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
        trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'FASHIONMNIST':
        transform.transforms = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'IMAGENET':
        transform.transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        trainset = torchvision.datasets.ImageNet(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.ImageNet(root='./data', train=False, download=True, transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes

def train_model(model, trainloader, criterion, optimizer, scheduler, device, epochs=100, quantize=False, testloader=None, n_bits=8, method='sigma', model_name = 'model', dataset_name='Dataset'):
    model.to(device)
    model.train()
    best_acc = 0
    train_accs = []
    test_accs = []
    quant_accs = []

    # for epoch in range(epochs):
    #     correct, total = 0, 0
    #     progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
    #     for inputs, targets in progress_bar:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
            
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
    #         progress_bar.set_postfix(loss=loss.item())
    #     scheduler.step()
    #     train_acc = 100. * correct / total
    #     train_accs.append(train_acc)

    #     # 测试准确率
    #     test_acc = test_model(model, testloader)
    #     test_accs.append(test_acc)
        
        # 量化评估
    if quantize:
        quant_acc = evaluate_quantized_model(copy.deepcopy(model), testloader, n_bits, method)
        quant_accs.append(quant_acc)
        # print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Quant Acc: {quant_acc:.2f}%")
    # else:
    #     print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # if test_acc > best_acc:
    #     print(f"Saving model with test accuracy of {test_acc:.2f}% at epoch {epoch+1}")
    #     best_acc = test_acc
    #     torch.save(model.state_dict(), f'./src/Experiment/{model_name}_{dataset_name}/{model_name}_{dataset_name}_best.pth')

    return train_accs, test_accs, quant_accs

def test_model(model, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def evaluate_quantized_model(original_model, testloader, n_bits=8, method='sigma'):
    """
    Evaluate model performance after quantization.
    
    Args:
        original_model: The model to be quantized
        testloader: DataLoader for testing
        n_bits: Number of bits for quantization
        method: Quantization method to use
    
    Returns:
        float: Accuracy of the quantized model
    """
    # Initialize quantizer and create model copy
    quantization_layer_num = 0
    flatten_weight_len = 0
    quantizer = Quantizer()
    quantized_model = copy.deepcopy(original_model)
    
    # Extract weights from convolutional and linear layers
    layers = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) or is_quantizable(name):
            weight = module.weight.data.cpu()
            shape = weight.shape
            layers[name] = {
                'weight': weight.numpy(),
                'shape': shape
            }
 
    
    # Quantize each layer
    quantized_layers = {}
    for name, layer in layers.items():
        weight = layer['weight'].flatten()
        flatten_weight_len += len(weight)
        layer_max = np.max(weight)
        layer_min = np.min(weight)
        layer_mu = np.mean(weight)
        layer_sigma = np.std(weight)
        
        # Perform quantization
        quantized_weight, params = quantizer.quantize_weights_unified(
            weight=weight,
            n_bits=n_bits,
            method=method,
            global_max=layer_max,
            global_min=layer_min,
            global_mu=layer_mu,
            global_sigma=layer_sigma
        )
        
        quantized_layers[name] = {
            'quantized_weight': quantized_weight,
            'shape': layer['shape'],
            'params': params
        }

    # Update model weights with quantized values
    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in quantized_layers:
            layer = quantized_layers[name]
            dequantized = quantizer.dequantize_weights_unified(
                layer['quantized_weight'], 
                layer['params']
            )
            weight = np.reshape(dequantized, layer['shape'])
            module.weight.data = torch.from_numpy(weight).float()
    quantization_layer_num = flatten_weight_len // 4096 + 1
    print(f'Quantization layer num: {quantization_layer_num}')
    return test_model(quantized_model, testloader)

def plot_results(train_accs, test_accs, quant_accs=None, model_name='Model', dataset_name='Dataset'):

    
    # Create figure with higher DPI for better quality
    plt.figure(figsize=(12, 7), dpi=300)
    
    # Plot with better styling
    plt.plot(train_accs, label='Train', color='#2ecc71', linestyle='-', linewidth=2)
    plt.plot(test_accs, label='Test', color='#3498db', linestyle='-', linewidth=2)
    if quant_accs:
        plt.plot(quant_accs, label='Quantized', color='#e74c3c', linestyle='-', linewidth=2)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate maximum values
    max_test = max(test_accs)
    plt.annotate(f'Max Test: {max_test:.2f}%', 
                xy=(test_accs.index(max_test), max_test),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    if quant_accs:
        max_quant = max(quant_accs)
        plt.annotate(f'Max Quant: {max_quant:.2f}%',
                    xy=(quant_accs.index(max_quant), max_quant),
                    xytext=(10, -20),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Improve title and labels
    plt.title(f'{model_name} Performance on {dataset_name}', 
              fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Add legend with better positioning
    plt.legend(loc='lower right', frameon=True, framealpha=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with higher quality
    plt.savefig(f'./src/Experiment/{model_name}_{dataset_name}/{model_name}_{dataset_name}_results.png', 
                bbox_inches='tight',
                dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Nerual Network Training')
    
    parser.add_argument('--model', type=str, choices=[  'LeNet5', 
                                                        'ResNet18m', 
                                                        'VGG19'
                                                        'PreActResNet18',
                                                        'GoogLeNet',
                                                        'DenseNet121',
                                                        'ResNeXt29_2x64d',
                                                        'MobileNet',
                                                        'MobileNetV2',
                                                        'DPN92',
                                                        'ShuffleNetG2',
                                                        'SENet18',
                                                        'ShuffleNetV2',
                                                        'EfficientNetB0',
                                                        'RegNetX_200MF',
                                                        'SimpleDLA'
                                                        ], default='ResNet18')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'FASHIONMNIST', 'IMAGENET'], default='CIFAR10')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--quantize', default= True, action='store_true', help='Enable quantization evaluation')
    parser.add_argument('--n_bits', type=int, default=8, help='Bits for quantization')
    parser.add_argument('--method', type=str, default='sigma', help='Quantization method')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('='*50)
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model}')
    print(f'Learning rate: {args.lr}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Device: {device}')
    if args.quantize:
        print(f'Quantization: {args.n_bits} bits, method: {args.method}')
    print('='*50)
    
  
    
    # 自动匹配模型和数据集
    if args.model == 'LeNet5':
        args.dataset = 'FASHIONMNIST' 
    elif args.dataset == 'FAshionMNIST':
        args.model = 'LeNet5'
        
    trainloader, testloader, num_classes = load_dataset(args.dataset, args.batch_size)
    
    # 初始化模型
    if args.model == 'LeNet5':
        model = LeNet5()
    elif args.model == 'ResNet18':
        model = ResNet18()
    elif args.model == 'VGG19':
        model = VGG('VGG19')
    elif args.model == 'PreActResNet18':
        model = PreActResNet18()
    elif args.model == 'GoogLeNet':
        model = GoogLeNet()
    elif args.model == 'DenseNet121':
        model = DenseNet121()
    elif args.model == 'ResNeXt29_2x64d':
        model = ResNeXt29_2x64d()
    elif args.model == 'MobileNet':
        model = MobileNet()
    elif args.model == 'MobileNetV2':
        model = MobileNetV2()
    elif args.model == 'DPN92':
        model = DPN92()
    elif args.model == 'ShuffleNetG2':
        model = ShuffleNetG2()
    elif args.model == 'SENet18':
        model = SENet18()
    elif args.model == 'ShuffleNetV2':
        model = ShuffleNetV2(1)
    elif args.model == 'EfficientNetB0':
        model = EfficientNetB0()
    elif args.model == 'RegNetX_200MF':
        model = RegNetX_200MF()
    elif args.model == 'SimpleDLA':
        model = SimpleDLA()
    elif args.model == 'AlexNet':
        model = AlexNet()
    else:
        raise ValueError(f"Model {args.model} not supported.")
       

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train the model with specified parameters
    train_accs, test_accs, quant_accs = train_model(
        model=model,
        trainloader=trainloader, 
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        quantize=args.quantize,
        testloader=testloader,
        n_bits=args.n_bits,
        model_name=args.model,
        dataset_name=args.dataset
    )

    # 保存结果和绘图
    plot_results(
        train_accs, 
        test_accs, 
        quant_accs if args.quantize else None, 
        args.model, 
        args.dataset)
    print('Training complete.')

if __name__ == '__main__':
    main()