import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LeNet5(nn.Module):
    def __init__(self, num_classes):
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

def get_dataset(dataset_name):
    if dataset_name == 'FASHIONMNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                   download=True, transform=transform)
        num_classes = 10
    else:  # CIFAR10 or CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        if dataset_name == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform_test)
            num_classes = 10
        else:  # CIFAR100
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                   download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                  download=True, transform=transform_test)
            num_classes = 100
    return trainset, testset, num_classes

def train_model(model_name, dataset_name, epochs=100, batch_size=512, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取数据集
    trainset, testset, num_classes = get_dataset(dataset_name)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # 创建模型
    if model_name == 'LeNet5':
        model = LeNet5(num_classes).to(device)
    else:  # ResNet18
        model = resnet18(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_accs = []
    test_accs = []
    best_acc = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        train_accs.append(train_acc)
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{model_name}_{dataset_name}_best.pth')
        
        scheduler.step()
    
    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    
    # 获取图形的当前轴对象
    ax = plt.gca()
    
    # 计算右下角的位置
    x_max = len(train_accs)
    y_min = min(min(train_accs), min(test_accs))
    y_max = max(max(train_accs), max(test_accs))
    
    # 添加带背景的文本框
    plt.text(x_max * 0.7, y_min + (y_max - y_min) * 0.1, 
             f'Max Train: {max(train_accs):.2f}%\nMax Test: {max(test_accs):.2f}%',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             ha='left', va='bottom')
    
    plt.title(f'{model_name} on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'{model_name}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 训练所有组合
    combinations = [
        # ('LeNet5', 'FASHIONMNIST'),
        ('ResNet18', 'CIFAR10')
        # ('ResNet18', 'CIFAR100')
    ]
    
    for model_name, dataset_name in combinations:
        print(f"\nTraining {model_name} on {dataset_name}")
        train_model(model_name, dataset_name)
        
        
