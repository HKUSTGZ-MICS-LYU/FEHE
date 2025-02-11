import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.quantization import quantize_dynamic
from tqdm import tqdm

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
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Dataset not supported")

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

def main():
    # 参数设置
    dataset = 'cifar10'  # 或 'cifar100'
    num_classes = 10 if dataset == 'cifar10' else 100
    batch_size = 128
    epochs = 10
    learning_rate = 0.001

    # 加载数据
    train_loader, test_loader = load_data(dataset=dataset, batch_size=batch_size)

    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(num_classes=num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print("Training the model...")
    train(model, train_loader, criterion, optimizer, device=device, epochs=epochs)

    # 测试原始模型
    print("Testing the original model...")
    original_acc = test(model, test_loader, device)
    print(f"Original Model Accuracy: {original_acc:.2f}%")

    # 分层动态量化（仅量化卷积层和全连接层）
    print("Quantizing the model...")
    quantized_model = quantize_dynamic(
        model.to('cpu'),  # 量化需要在CPU上进行
        {nn.Conv2d, nn.Linear},  # 指定要量化的层类型
        dtype=torch.qint8
    )

    # 测试量化后模型
    print("Testing the quantized model...")
    quantized_acc = test(quantized_model, test_loader, device='cpu')
    print(f"Quantized Model Accuracy: {quantized_acc:.2f}%")

    # 精度比较
    print(f"Accuracy Drop: {original_acc - quantized_acc:.2f}%")

if __name__ == '__main__':
    main()