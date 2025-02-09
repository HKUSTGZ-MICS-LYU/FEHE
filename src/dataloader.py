from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from flwr_datasets import FederatedDataset
from typing import Tuple


def load_datasets(
    DATASET_NAME: str,
    NUM_CLIENTS: int,
    BATCH_SIZE: int,
    partition_id: int = None,
    federated: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Dynamically load datasets based on the specified dataset name and model requirements.

    Args:
        DATASET_NAME (str): Name of the dataset.
        NUM_CLIENTS (int): Number of clients for federated learning (used only if federated=True).
        BATCH_SIZE (int): Batch size for training and validation.
        partition_id (int, optional): ID for the data partition (used for federated learning).
        federated (bool): Whether to use federated dataset partitioning.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """

    # Define dataset-specific transforms
    
    # federated learning
    if federated:
        # Federated dataset partitioning
        if DATASET_NAME == "cifar10":
            fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
            partition = fds.load_partition(partition_id)
            # Divide the partition into train and test sets (80% train, 20% test)
            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            pytorch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                     std=[0.2023, 0.1994, 0.2010]),
            ])
            def apply_transforms(batch):
                batch["img"] = [pytorch_transform(img) for img in batch["img"]]
                return batch

            # Apply transforms and create DataLoaders
            partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
            )
            valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=BATCH_SIZE)
            return trainloader, valloader, testloader
            
        elif DATASET_NAME == "cifar100":
            fds = FederatedDataset(dataset="cifar100", partitioners={"train": NUM_CLIENTS})
            partition = fds.load_partition(partition_id)
            # Divide the partition into train and test sets (80% train, 20% test)
            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            pytorch_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                  std=[0.2023, 0.1994, 0.2010])
            ])
            def apply_transforms(batch):
                batch["img"] = [pytorch_transform(img) for img in batch["img"]]
                return batch

            # Apply transforms and create DataLoaders
            partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
            )
            valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=BATCH_SIZE)
            return trainloader, valloader, testloader
        
        
        elif DATASET_NAME == "mnist":
            fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
            partition = fds.load_partition(partition_id)
            # Divide the partition into train and test sets (80% train, 20% test)
            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            pytorch_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            def apply_transforms(batch):
                batch["image"] = [pytorch_transform(img) for img in batch["image"]]
                return batch

            # Apply transforms and create DataLoaders
            partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
            )
            valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=BATCH_SIZE)
            return trainloader, valloader, testloader
    
        elif DATASET_NAME == "fashionmnist":
            fds = FederatedDataset(dataset="zalando-datasets/fashion_mnist", partitioners={"train": NUM_CLIENTS})
            partition = fds.load_partition(partition_id)
            # Divide the partition into train and test sets (80% train, 20% test)
            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            pytorch_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            def apply_transforms(batch):
                batch["image"] = [pytorch_transform(img) for img in batch["image"]]
                return batch

            # Apply transforms and create DataLoaders
            partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
            )
            valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=BATCH_SIZE)
            return trainloader, valloader, testloader
        else:
            raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
        
        
    # Non-federated learning
    else:
        if DATASET_NAME == "cifar10":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            trainset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
            testset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
            testloader = DataLoader(testset, batch_size=BATCH_SIZE)
            return trainloader, None, testloader
        else:
            raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
            