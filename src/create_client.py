
# ====================== create_client.py ======================
"""Author: Meng Xiangchen"""
"""Flower client implementation for federated learning with enhanced security features."""


# Standard library imports
import argparse
import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Third-party imports
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import flwr as fl
from flwr.common import Context, NDArrays, Scalar
from flwr.client import Client, NumPyClient, ClientApp

# Local imports
import utils
from utils.dataloader import load_datasets
from utils.encryption import *
from utils.quantization import *
from utils import *

from models import (
    LeNet5, ResNet18, VGG, PreActResNet18, GoogLeNet, DenseNet121,
    ResNeXt29_2x64d, MobileNet, MobileNetV2, DPN92, ShuffleNetG2,
    SENet18, ShuffleNetV2, EfficientNetB0, RegNetX_200MF, SimpleDLA
)
from utils.test import test
from utils.train import train
from utils.utils import get_parameters, set_parameters
import hashlib


@dataclass
class ClientConfig:
    """Client configuration dataclass."""
    partition_id: int 
    client_number: int 
    lr: float = 0.01
    scheduler: str = "cosine"
    optimizer: str = "adam"
    batch_size: int = 128
    model_name: str = "LeNet5"
    dataset_name: str = "FASHIONMNIST"
    encrypted_dir: str = "encrypted"
 
class SecureClient(NumPyClient):
    """Federadeted Learning Client with Encrypted and Quantized Support"""
    def __init__(
        self,
        config: ClientConfig,
        model: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader
    ):
        self.config = config
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.accuracy_log: Dict[int, float] = {}
        self.time_metrics: Dict[str, list] = {"train": [], "evaluate": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure encrypted directory exists
        Path(config.encrypted_dir).mkdir(parents=True, exist_ok=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self,
        parameters,
        config
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model on local data with enhanced security measures."""

        
        set_parameters(self.model, parameters)
        
        server_round = config.get("server_round")
    
    
        current_lr = self.config.lr * (0.1 ** (server_round // 30))

        # Merge configuration
        full_config = {
            "lr": current_lr,
            "min_lr": 0.001,
            "scheduler": self.config.scheduler,
            "total_rounds": 200,
            "server_round": config.get("server_round"),
            **config
        }
  
        # Training phase
        start_time = time.time()
        train(
            net = self.model, 
            trainloader = self.trainloader,
            epochs = 1,
            config=full_config,
            verbose=True
        )
        self.time_metrics["train"].append(time.time() - start_time)
    
        params_path = f"{self.config.encrypted_dir}/client_{self.config.partition_id}_params.pth"
        torch.save(self.model.state_dict(), params_path)
        
        return get_parameters(self.model), len(self.trainloader), {"pid": self.config.partition_id}
        
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local data."""  
        
    
        # Load aggregated parameters
        params_path = f"{self.config.encrypted_dir}/aggregated_params.pth"
        # # Read hash of aggregated params file
        # hash_path = f"{self.config.encrypted_dir}/aggregated_params.hash"
        # with open(hash_path, 'r') as f:
        #     expected_hash = f.read().strip()

        # # Calculate hash of current params file 
        # with open(params_path, 'rb') as f:
        #     actual_hash = hashlib.sha256(f.read()).hexdigest()

        # Verify hash matches before loading
        # if actual_hash != expected_hash:
            # raise ValueError("Parameter file hash mismatch - possible tampering detected")
        self.model.load_state_dict(torch.load(params_path))
        self.model.to(self.device)
        # Evaluation
        loss, accuracy = test(self.model, self.testloader, verbose=True)
          
        self.accuracy_log[config.get("server_round")] = accuracy
        
        
        if config.get("server_round") == 100:
            self._finalize_training()
        
        return loss, len(self.valloader), {"accuracy": float(accuracy)}
    
    def _finalize_training(self) -> None:
        """Handle post-training operations."""
        print(f"\nClient {self.config.partition_id} Accuracy Report:")
        for rnd, acc in sorted(self.accuracy_log.items()):
            print(f" Round {rnd}: {acc:.4f}")
            
        # Save metrics
        self._save_accuracy_csv()
        
        self._save_time_stats()

    def _save_accuracy_csv(self):
        """Save the accuracy log to a CSV file."""
        csv_path = f"client_{self.config.partition_id}_accuracy.csv"
        try:
            with open(csv_path, "w") as f:
                f.write("round,accuracy\n")
                for rnd, acc in sorted(self.accuracy_log.items()):
                    f.write(f"{rnd},{acc}\n")
            print(f"Accuracy log saved to {csv_path}")
        except Exception as e:
            print(f"Error saving accuracy log: {str(e)}")
    
    def _save_time_stats(self):
        """Save time metrics to a CSV file."""
        stats_path = f"client_{self.config.partition_id}_time_stats.csv"
        try:
            with open(stats_path, "w") as f:
                f.write("operation,round,time\n")
                for operation, times in self.time_metrics.items():
                    for round_num, t in enumerate(times, 1):
                        f.write(f"{operation},{round_num},{t}\n")
            print(f"Time statistics saved to {stats_path}")
        except Exception as e:
            print(f"Error saving time statistics: {str(e)}")

def load_model(model_name: str, dataset: str) -> torch.nn.Module:
    """Factory function to load a model by name."""
    num_classes = 10 if dataset == "CIFAR10" else 100
    
    model_map = {
        'LeNet5': LeNet5,
        'ResNet18': lambda: ResNet18(num_classes=num_classes),
        'VGG19': lambda: VGG("VGG19"),
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
    # Handle dataset-specific model selection
    if dataset == "FASHIONMNIST":
        return LeNet5()
        
    model = model_map.get(model_name)()
    return model


def client_fn(context: Context) -> fl.client.Client:
    """Client creation function for Flower framework."""
    config = ClientConfig(**context.node_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    trainloader, valloader, _ = load_datasets(
        dataset_name=config.dataset_name,
        client_number=config.client_number,
        batch_size=config.batch_size,
        partition_id=config.partition_id,
        federated=True
    )
    
    # Initialize model
    model = load_model(config.model_name, config.dataset_name).to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    return SecureClient(config, model, trainloader, valloader).to_client()

def main():
    """Main client execution flow."""
    parser = argparse.ArgumentParser(description="Secure Federated Learning Client")
    parser.add_argument("--partition-id",   type=int,   default=0)
    parser.add_argument("--client-number",  type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=0.05)
    parser.add_argument("--scheduler",      type=str,   default="step", choices=["cosine", "step"])
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--model_name",     type=str,   default="LeNet5")
    parser.add_argument("--dataset_name",   type=str,   default="FASHIONMNIST")
    args = parser.parse_args()
    config = ClientConfig(**vars(args))
    
    
    # Initialize components
    model = load_model(config.model_name, config.dataset_name)
    trainloader, valloader, testloader = load_datasets(
        DATASET_NAME = config.dataset_name,
        CLIENT_NUMER = config.client_number,
        BATCH_SIZE   = config.batch_size,
        PARTITION_ID = config.partition_id,
        FEDERATED    = True
    )
    
    # Start client
    client = SecureClient(
        config=config,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        valloader=valloader
    )
    fl.client.start_numpy_client(
        server_address=get_server_address(),
        client=client,
        grpc_max_message_length=1024**3
    )

def get_server_address() -> str:
    """Retrieve server address from file."""
    with open("server_address.txt") as f:
        return f.read().strip()

if __name__ == "__main__":
    main()