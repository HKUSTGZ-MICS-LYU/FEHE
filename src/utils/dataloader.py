import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from flwr_datasets import FederatedDataset
from typing import Tuple

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
    """将数据集按IID方式划分为多个客户端的子集。

    Args:
        dataset: 待划分的数据集。
        num_clients (int): 客户端数量。

    Returns:
        List[List[int]]: 每个客户端的样本索引列表。
    """
    total_samples = len(dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    client_size = total_samples // num_clients
    client_indices = [indices[i * client_size: (i + 1) * client_size] for i in range(num_clients)]
    # 处理余数
    for i in range(total_samples % num_clients):
        client_indices[i].append(indices[num_clients * client_size + i])
    return client_indices

def _split_non_IID(dataset, num_clients: int, alpha: float):
    """使用Dirichlet分布将数据集按non-IID方式划分为多个客户端的子集。

    Args:
        dataset: 待划分的数据集。
        num_clients (int): 客户端数量。
        alpha (float): Dirichlet分布的浓度参数，控制non-IID程度。

    Returns:
        List[List[int]]: 每个客户端的样本索引列表。
    """
    # 如果数据集有targets属性，使用它以提高效率；否则遍历数据集
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([label for _, label in dataset])
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        # 使用Dirichlet分布生成每个客户端的比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # 使用multinomial直接分配样本数量
        counts = np.random.multinomial(len(idx_k), proportions)
        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(idx_k[start:end].tolist())
            start = end
    return client_indices

def load_datasets(
    DATASET_NAME: str,
    CLIENT_NUMER: int,
    BATCH_SIZE: int,
    PARTITION_ID: int = None,
    FEDERATED: bool = False,
    IID: bool = False,
    alpha: float = 1.0,
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

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试DataLoader。
    """
    if FEDERATED:  # 联邦学习
        trainset, testset = load_full_dataset(DATASET_NAME)
        if IID:
            client_indices = _split_IID(trainset, CLIENT_NUMER)
        else:
            client_indices = _split_non_IID(trainset, CLIENT_NUMER, alpha)
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
