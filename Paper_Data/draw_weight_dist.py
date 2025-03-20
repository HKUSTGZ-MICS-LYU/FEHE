import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import ResNet18_Weights  
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)
    
# Load the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=False, num_workers=1)

    # 创建模型并移至设备
    net = ResNet().to(device)
    
    # 将损失函数移至设备
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        running_loss = 0.0
        print(f'Epoch {epoch + 1}')
        for i, data in enumerate(trainloader, 0):
            # 将输入数据移至设备
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


                
    print('Finished Training')

    # 测试模型时也需要将数据移至设备
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    # Function to get weights from a module
    def get_weights(module):
        weights = []
        for name, param in module.named_parameters():
            if 'weight' in name:
                weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    # Function to calculate percentage in sigma intervals
    def get_sigma_percentages(weights, mean, std):
        total = len(weights)
        percentages = {}
        for i in [1, 2, 3, 4]:
            mask = (weights >= mean - i*std) & (weights <= mean + i*std)
            percentages[i] = 100 * np.sum(mask) / total
        return percentages

    # Collect weights
    conv_weights = []
    fc_weights = []
    all_weights = []

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_weights.extend(module.weight.data.cpu().numpy().flatten())
        elif isinstance(module, nn.Linear):
            fc_weights.extend(module.weight.data.cpu().numpy().flatten())
        if hasattr(module, 'weight') and module.weight is not None:
            all_weights.extend(module.weight.data.cpu().numpy().flatten())

    conv_weights = np.array(conv_weights)
    fc_weights = np.array(fc_weights)
    all_weights = np.array(all_weights)

    # Set style parameters
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'DejaVu Sans',
        'axes.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 16,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlesize': 20
    })

    # Calculate limits for x-axis
    def get_xlim(weights):
        mean = np.mean(weights)
        std = np.std(weights)
        return mean - 4*std, mean + 4*std

    # Figure 1: All weights distribution
    fig1, ax1 = plt.subplots(figsize=(15, 10), dpi=600)
    mean = np.mean(all_weights)
    std = np.std(all_weights)
    percentages = get_sigma_percentages(all_weights, mean, std)

    xlim = get_xlim(all_weights)
    n, bins, patches = plt.hist(all_weights, 
                                bins=100, 
                                density=True, 
                                alpha=0.7,
                                color='#2E86C1',
                                linewidth=2,
                                range=xlim,
                                edgecolor='black')

    # Add standard deviation lines with percentages
    line_styles = [
        (mean + 4*std, '#FF8C00', f'+4σ, ({percentages[4]:.1f}%)', 3),
        (mean - 4*std, '#FF8C00', '-4σ', 3),
        (mean + 3*std, '#FF5733', f'+3σ ({percentages[3]:.1f}%)', 3),
        (mean - 3*std, '#FF5733', f'-3σ', 3),
        (mean + 2*std, '#FFC300', f'+2σ ({percentages[2]:.1f}%)', 2.5),
        (mean - 2*std, '#FFC300', f'-2σ', 2.5),
        (mean + std, '#28B463', f'+1σ ({percentages[1]:.1f}%)', 2),
        (mean - std, '#28B463', f'-1σ', 2),
        (mean, '#E74C3C', 'mean', 3.5)
    ]

    for x, color, label, width in line_styles:
        plt.axvline(x=x, color=color, linestyle='--' if 'σ' in label else '-',
                    linewidth=width, label=label, alpha=0.8)

    plt.title('Weight Distribution of All Layers in ResNet-18', 
                fontsize=24, pad=20, fontweight='bold')
    plt.xlabel('Weight Value', fontsize=20, labelpad=10)
    plt.ylabel('Frequency', fontsize=20, labelpad=10)
    plt.legend(fontsize=16, frameon=True, fancybox=True, framealpha=0.9,
                shadow=True, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_weights_distribution.png', bbox_inches='tight', dpi=600)
    plt.close()

    # Figure 2: Separate distributions for conv and fc layers
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=1200)

    # Convolutional layers
    mean_conv = np.mean(conv_weights)
    std_conv = np.std(conv_weights)
    conv_percentages = get_sigma_percentages(conv_weights, mean_conv, std_conv)
    xlim_conv = get_xlim(conv_weights)
    ax1.hist(conv_weights, bins=100, density=True, alpha=0.7,
                color='#2E86C1', linewidth=2, range=xlim_conv, edgecolor='black')
        
    for x, color, label, width in [
        (mean_conv + 4*std_conv, '#3498DB', f'+4σ ({conv_percentages[4]:.1f}%)', 3),
        (mean_conv - 4*std_conv, '#3498DB', '-4σ', 3),
        (mean_conv + 3*std_conv, '#3498DB', f'+3σ ({conv_percentages[3]:.1f}%)', 3),
        (mean_conv - 3*std_conv, '#3498DB', '-3σ', 3),
        (mean_conv + 2*std_conv, '#1ABC9C', f'+2σ ({conv_percentages[2]:.1f}%)', 3),
        (mean_conv - 2*std_conv, '#1ABC9C', '-2σ', 3),
        (mean_conv + std_conv, '#F1C40F', f'+1σ ({conv_percentages[1]:.1f}%)', 3),
        (mean_conv - std_conv, '#F1C40F', '-1σ', 3),
        (mean_conv, '#E74C3C', 'mean', 3.5)
    ]:
        ax1.axvline(x=x, color=color, linestyle='--' if 'σ' in label else '-',
                    linewidth=width, label=label, alpha=0.8)
    
    ax1.set_title('Convolutional Layers', fontsize=26, pad=20)
    ax1.set_xlabel('Weight Value', fontsize=24)
    ax1.set_ylabel('Frequency', fontsize=24)
    ax1.legend(fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Fully connected layers
    mean_fc = np.mean(fc_weights)
    std_fc = np.std(fc_weights)
    fc_percentages = get_sigma_percentages(fc_weights, mean_fc, std_fc)
    xlim_fc = get_xlim(fc_weights)
    ax2.hist(fc_weights, bins=100, density=True, alpha=0.7,
                color='#2E86C1', linewidth=2, range=xlim_fc, edgecolor='black')
    
    for x, color, label, width in [
        (mean_fc + 3*std_fc, '#3498DB', f'+3σ ({fc_percentages[3]:.1f}%)', 3),
        (mean_fc - 3*std_fc, '#3498DB', '-3σ', 3),
        (mean_fc + 2*std_fc, '#1ABC9C', f'+2σ ({fc_percentages[2]:.1f}%)', 3),
        (mean_fc - 2*std_fc, '#1ABC9C', '-2σ', 3),
        (mean_fc + std_fc, '#F1C40F', f'+1σ ({fc_percentages[1]:.1f}%)', 3),
        (mean_fc - std_fc, '#F1C40F', '-1σ', 3),
        (mean_fc, '#E74C3C', 'mean', 3.5)
    ]:
        ax2.axvline(x=x, color=color, linestyle='--' if 'σ' in label else '-',
                    linewidth=width, label=label, alpha=0.8)
                    
    ax2.set_title('Fully Connected Layers', fontsize=26, pad=20)
    ax2.set_xlabel('Weight Value', fontsize=24)
    ax2.set_ylabel('Frequency', fontsize=24)
    ax2.legend(fontsize=20)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Weight Distributions in ResNet-18',
                    fontsize=30, y=1.05, fontweight='bold')
    plt.tight_layout()
    plt.savefig('layer_type_weight_distributions.png',
                bbox_inches='tight', dpi=1200)
    plt.savefig('layer_type_weight_distributions.pdf',
                bbox_inches='tight', dpi=1200)
    plt.close()