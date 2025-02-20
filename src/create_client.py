# Standard library imports
import argparse
import time
import json

# Third-party imports
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import flwr as fl
from flwr.common import Context
from flwr.client import Client, NumPyClient, ClientApp

# Local imports
from models import *
from utils.utils import get_parameters, set_parameters
from utils.train import train
from utils.test import test
from utils.dataloader import load_datasets


 

class FlowerClient(NumPyClient):
    # The characteristics of the client are defined in the constructor
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epoch_accuracy = {}
        self.quant_params = None

        self.time_stats = {
            'train': [],
            'evaluate': []
        }

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(config)
        set_parameters(self.net, parameters)
        
        full_config = {
            "initial_lr": 0.1,
            "min_lr": 0.001,
            "scheduler": args.scheduler,
            "total_rounds": config["total_rounds"],
            "server_round": config["server_round"],
            **config
        }
        
        train_start = time.time()
        train(self.net, self.trainloader, epochs=config["local_epochs"], config=full_config, verbose=True)
        train_time = time.time() - train_start
        self.time_stats['train'].append(train_time)

        # Save the new parameters to a file named with pid 
        updated_params = self.net.state_dict()
        torch.save(updated_params, f"encrypted/client_{self.pid}_params.pth")
                  
        return get_parameters(self.net), len(self.trainloader), {"pid": self.pid}

    def evaluate(self, parameters, config):        
        server_round = config["server_round"]
    
        # Read the new parameters from the file named with "aggregated_params.pth"
        parameters = torch.load(f"encrypted/aggregated_params.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.net.load_state_dict(parameters)
  
        loss, accuracy = test(self.net, self.valloader, verbose=True)
        self.epoch_accuracy[server_round] = accuracy            
        if server_round == config.get("total_rounds"):
            self.on_train_end()
            self._save_time_stats()  
        return loss, len(self.valloader), {"accuracy": float(accuracy)}
    
    def _reshape_parameters(self, parameters, decrypted_data):
        """Reshape the decrypted data to match the shape of the original parameters."""
        reshaped_params = []
        current_index = 0
        for shape in [np.shape(arr) for arr in parameters]:
            size = np.prod(shape)
            reshaped_arr = np.reshape(
                decrypted_data[current_index:current_index + size], 
                shape
            )
            reshaped_params.append(reshaped_arr)
            current_index += size
        return reshaped_params
    
    def on_train_end(self):
        """Print the accuracy of the client after training."""
        if not self.epoch_accuracy:
            return
            
        print(f"\nClient {self.pid} Accuracy Report:")
        for rnd, acc in sorted(self.epoch_accuracy.items()):
            print(f" Round {rnd}: {acc:.4f}")
        
        self._save_accuracy_csv()
    
    def _save_accuracy_csv(self):
        """Save the accuracy log to a CSV file."""
        csv_path = f"client_{self.pid}_accuracy.csv"
        try:
            with open(csv_path, "w") as f:
                f.write("round,accuracy\n")
                for rnd, acc in sorted(self.epoch_accuracy.items()):
                    f.write(f"{rnd},{acc}\n")
            print(f"Accuracy log saved to {csv_path}")
        except Exception as e:
            print(f"Error saving accuracy log: {str(e)}")
    
    def _save_time_stats(self):
        """保存时间统计到CSV文件"""
        stats_path = f"client_{self.pid}_time_stats.csv"
        try:
            with open(stats_path, "w") as f:
                f.write("operation,round,time\n")
                for operation, times in self.time_stats.items():
                    for round_num, t in enumerate(times, 1):
                        f.write(f"{operation},{round_num},{t}\n")
            print(f"Time statistics saved to {stats_path}")
        except Exception as e:
            print(f"Error saving time statistics: {str(e)}")

def client_fn(context) -> Client:
    """Create a Flower client representing a single organization."""
    BATCH_SIZE = context.node_config.get("batch_size")
    CLIENT_NUMER = context.node_config.get("num_supernodes")
    MODEL_NAME = context.node_config.get("model")
    DATASET_NAME = context.node_config.get("dataset")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("THE DEVICE IS", DEVICE)
    

    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(DATASET_NAME, 
                                            CLIENT_NUMER, 
                                            BATCH_SIZE,
                                            partition_id=partition_id,
                                            federated=True
                                            )
    if MODEL_NAME == 'LeNet5':
        model = LeNet5()
    elif MODEL_NAME == 'ResNet18':
        model = ResNet18()
    elif MODEL_NAME == 'VGG19':
        model = VGG('VGG19')
    elif MODEL_NAME == 'PreActResNet18':
        model = PreActResNet18()
    elif MODEL_NAME == 'GoogLeNet':
        model = GoogLeNet()
    elif MODEL_NAME == 'DenseNet121':
        model = DenseNet121()
    elif MODEL_NAME == 'ResNeXt29_2x64d':
        model = ResNeXt29_2x64d()
    elif MODEL_NAME == 'MobileNet':
        model = MobileNet()
    elif MODEL_NAME == 'MobileNetV2':
        model = MobileNetV2()
    elif MODEL_NAME == 'DPN92':
        model = DPN92()
    elif MODEL_NAME == 'ShuffleNetG2':
        model = ShuffleNetG2()
    elif MODEL_NAME == 'SENet18':
        model = SENet18()
    elif MODEL_NAME == 'ShuffleNetV2':
        model = ShuffleNetV2(1)
    elif MODEL_NAME == 'EfficientNetB0':
        model = EfficientNetB0()
    elif MODEL_NAME == 'RegNetX_200MF':
        model = RegNetX_200MF()
    elif MODEL_NAME == 'SimpleDLA':
        model = SimpleDLA()
        
    model = model.to(DEVICE)
    
    return FlowerClient(partition_id, model, trainloader, valloader).to_client()



def create_client_fn(batch_size, num_supernodes, model_name, dataset_name):
    """Return a *ClientApp* that Flower can call to construct clients."""
   
    def client_creation_func(context: Context) -> Client:
        # 原本 client_fn(context) 的逻辑
        context.node_config["model"] = model_name
        context.node_config["batch_size"] = batch_size
        context.node_config["num_supernodes"] = num_supernodes
        context.node_config["dataset"] = dataset_name
        return client_fn(context)

    # Return the client creation function
    client_app = ClientApp(client_fn=client_creation_func)
    return client_app

# The following code snippet is from src/create_client.py
def get_server_address():
    with open("server_address.txt", "r") as f:
        return f.read().strip()


def save_args_to_config(args):
    config = {
        "lr": args.lr,
        "min_lr": 0.001,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "model": args.model,
        "dataset": args.dataset,
    }
    with open("client_config.json", "w") as f:
        json.dump(config, f)
        
if __name__ == "__main__":
    """Create a Flower client representing a single organization."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Flower Client")
    # Client configuration
    parser.add_argument("--partition_id", type=int, default=0, 
                       help="Partition ID")
    parser.add_argument("--client_number", type=int, default=10, 
                       help="Number of clients")
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.01, 
                       help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step"])
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="Batch size")
    # Model selection
    model_choices = [   
                        'LeNet5',       'ResNet18m',        'VGG19',        'PreActResNet18',   'GoogLeNet',
                        'DenseNet121',  'ResNeXt29_2x64d',  'MobileNet',    'MobileNetV2',      'DPN92',
                        'ShuffleNetG2', 'SENet18',          'ShuffleNetV2', 'EfficientNetB0',   'RegNetX_200MF',
                        'SimpleDLA'
                    ]
    
    parser.add_argument("--model", type=str, default="ResNet18",
                       help="Model name", choices=model_choices)
    
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                       help="Dataset name", choices=["CIFAR10", "CIFAR100","FASHIONMNIST"])
        
    args = parser.parse_args()
    save_args_to_config(args)
   
    # Match the model name to the model class
    if args.model == "LeNet5":
        args.dataset = "FASHIONMNIST"
    elif args.dataset == 'FAshionMNIST':
        args.model = 'LeNet5'
        
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
        
    model = model.to(DEVICE)
    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    trainloader, valloader, _ = load_datasets(
        DATASET_NAME    =   args.dataset, 
        CLIENT_NUMER    =   args.client_number, 
        BATCH_SIZE      =   args.batch_size,
        PARTITION_ID    =   args.partition_id,
        FEDERATED       =   True
    )
    
    # Obtain the address of the server
    server_address = get_server_address()
    print(f"Connecting to server at {server_address}")
    
    # Create a Flower client
    client = FlowerClient(
        pid             =   args.partition_id,
        net             =   model,  
        trainloader     =   trainloader,
        valloader       =   valloader
    )
    
    # Start the Flower
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=1024*1024*1024
    )
    