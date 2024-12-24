from models import ResNet18, LeNet5, Net, AlexNet

MODEL_MAP = {
    "Net": Net,
    "ResNet18": ResNet18,
    "LeNet5": LeNet5,
    "AlexNet": AlexNet,
}

DATASET_MAP = {
    "mnist": "mnist",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "fashionmnist": "fashionmnist",
}