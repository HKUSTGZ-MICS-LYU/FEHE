import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from flwr_datasets import FederatedDataset
from typing import Tuple
import matplotlib.pyplot as plt

def analyze_label_distribution(dataset, client_indices):
    """分析每个客户端数据集的标签分布并可视化
    
    Args:
        dataset: 数据集
        client_indices: 每个客户端的样本索引列表
    """
    # 获取数据集的类别数
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    else:
        num_classes = len(set(dataset.targets))
    
    # 统计每个客户端的标签分布
    distributions = []
    for indices in client_indices:
        if hasattr(dataset, 'targets'):
            client_labels = [dataset.targets[idx] for idx in indices]
        else:
            client_labels = [dataset[idx][1] for idx in indices]
        label_dist = np.bincount(client_labels, minlength=num_classes)
        distributions.append(label_dist)
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    plt.imshow(distributions, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Number of samples')
    
    # 在每个格子中添加具体数值
    for i in range(len(distributions)):
        for j in range(num_classes):
            plt.text(j, i, distributions[i][j], 
                    ha='center', va='center')
    
    plt.xlabel('Label')
    plt.ylabel('Client ID')
    plt.title('Label Distribution Across Clients')
    plt.xticks(range(num_classes))
    plt.yticks(range(len(client_indices)), 
              [f'Client {i}' for i in range(len(client_indices))])
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')

def _get_transforms(DATASET_NAME: str):
    """获取指定数据集的预处理变换。

    Args:
        DATASET_NAME (str): 数据集名称。

    Returns:
        transforms.Compose: 数据预处理变换。
    """
    if DATASET_NAME == "CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                 std=[0.2023, 0.1994, 0.2010]),
        ])
    elif DATASET_NAME == "CIFAR100":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
    elif DATASET_NAME == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif DATASET_NAME == "FASHIONMNIST":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        raise ValueError(f"无效的数据集名称: {DATASET_NAME}")

def load_full_dataset(DATASET_NAME: str):
    """加载完整的数据集，包括训练集和测试集，并应用预处理变换。

    Args:
        DATASET_NAME (str): 数据集名称。

    Returns:
        Tuple[Dataset, Dataset]: 训练集和测试集。
    """
    transform = _get_transforms(DATASET_NAME)
    if DATASET_NAME == "CIFAR10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif DATASET_NAME == "CIFAR100":
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif DATASET_NAME == "MNIST":
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET_NAME == "FASHIONMNIST":
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"无效的数据集名称: {DATASET_NAME}")
    return trainset, testset


def _split_IID(dataset, num_clients: int):
    """将数据集按IID方式有放回地划分为多个客户端的子集。

    Args:
        dataset: 待划分的数据集。
        num_clients (int): 客户端数量。
        samples_per_client (int): 每个客户端的样本数量。

    Returns:
        List[List[int]]: 每个客户端的样本索引列表。
    """
    total_samples = len(dataset)
    indices =list(range(total_samples))
    
    # 每个客户端的样本数量
    samples_per_client = total_samples // num_clients
    client_indices = []
    # 无放回地抽取样本
    for _ in range(num_clients):
        client_indices.append(np.random.choice(indices, samples_per_client, replace=False).tolist())
    
    analyze_label_distribution(dataset, client_indices)
    return client_indices

def _split_non_IID(dataset, num_clients: int, alpha: float, samples_per_client: int):
    """使用Dirichlet分布将数据集按non-IID方式有放回地划分为多个客户端的子集。

    Args:
        dataset: 待划分的数据集。
        num_clients (int): 客户端数量。
        alpha (float): Dirichlet分布的浓度参数，控制non-IID程度。
        samples_per_client (int): 每个客户端的样本数量。

    Returns:
        List[List[int]]: 每个客户端的样本索引列表。
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([label for _, label in dataset])
    num_classes = len(np.unique(labels))
    
    # 为每个类别生成Dirichlet分布的比例
    class_proportions = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_classes)
    
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        # 为每个客户端计算该类别的样本数量
        class_samples = (class_proportions[k] * samples_per_client).astype(int)
        for i in range(num_clients):
            # 有放回地抽取样本
            if len(idx_k) > 0:
                indices = np.random.choice(idx_k, class_samples[i], replace=True).tolist()
                client_indices[i].extend(indices)
    
    # 调整样本数量以严格匹配samples_per_client
    for i in range(num_clients):
        current_samples = len(client_indices[i])
        if current_samples < samples_per_client:
            # 样本不足时，从整个数据集中补充
            additional_indices = np.random.choice(len(dataset), samples_per_client - current_samples, replace=True).tolist()
            client_indices[i].extend(additional_indices)
        elif current_samples > samples_per_client:
            # 样本过多时，随机丢弃
            client_indices[i] = np.random.choice(client_indices[i], samples_per_client, replace=False).tolist()
    

    analyze_label_distribution(dataset, client_indices)
    return client_indices

def load_datasets(
    DATASET_NAME: str,
    CLIENT_NUMER: int,
    BATCH_SIZE: int,
    PARTITION_ID: int = None,
    FEDERATED: bool = False,
    IID: bool = False,
    alpha: float = 1.0,
    samples_per_client: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """动态加载数据集，支持集中式和联邦式学习，支持IID和non-IID数据划分。

    Args:
        DATASET_NAME (str): 数据集名称。
        CLIENT_NUMER (int): 联邦学习中的客户端数量（仅在FEDERATED=True时使用）。
        BATCH_SIZE (int): 训练和验证的批大小。
        PARTITION_ID (int, optional): 联邦学习中的数据分区ID。
        FEDERATED (bool): 是否使用联邦学习数据划分。
        IID (bool): 是否使用IID数据划分。
        alpha (float): non-IID数据划分的Dirichlet分布参数。
        samples_per_client (int): 每个客户端的样本数量。

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试DataLoader。
    """
    if FEDERATED:  # 联邦学习
        trainset, testset = load_full_dataset(DATASET_NAME)
        if samples_per_client is None:
            samples_per_client = len(trainset) // CLIENT_NUMER  # 默认值
        if IID:
            client_indices = _split_IID(trainset, CLIENT_NUMER)
        else:
            client_indices = _split_non_IID(trainset, CLIENT_NUMER, alpha, samples_per_client)
        client_subset = Subset(trainset, client_indices[PARTITION_ID])
        
        # 将客户端数据分为训练集和验证集
        num_client_samples = len(client_subset)
        val_size = int(0.2 * num_client_samples)
        train_size = num_client_samples - val_size
        train_subset, val_subset = random_split(
            client_subset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=BATCH_SIZE)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)
        return trainloader, valloader, testloader
    else:  # 集中式学习
        trainset, testset = load_full_dataset(DATASET_NAME)
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)
        return trainloader, None, testloader
