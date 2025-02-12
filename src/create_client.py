import argparse
import numpy as np
import flwr as fl
from flwr.common import  Context
from flwr.client import Client, NumPyClient, ClientApp
import torch
from globals import MODEL_MAP
from utils import get_parameters, set_parameters
from train import train
from test import test
from dataloader import load_datasets
from quantization import *
import time
from flwr.common import *


 


class FlowerClient(Client):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.serialized_dataspace_log = []
        self.epoch_accuracy = {}
        self.quant_params = None
        self.time_stats = {
            'train': [],
            'evaluate': []
        }

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        selected_params = get_parameters(self.net)
        parameters = ndarrays_to_parameters(selected_params)
        return GetParametersRes(
            parameters=parameters, 
            status=Status(code=Code.OK, message="success")
            )


    def fit(self, ins: FitIns) -> FitRes:
        # Get the parameters from the server
        server_params = parameters_to_ndarrays(ins.parameters)
        
        # Set the parameters to the model
        set_parameters(self.net, server_params)
        
        # Train the model
        start_time = time.time()
        train(self.net, self.trainloader, epochs=ins.config['local_epochs'], verbose=True)
        end_time = time.time()
        train_time = end_time - start_time
        self.time_stats['train'].append(train_time)  
       
        # Get the updated parameters
        updated_params = get_parameters(self.net)
        
        # Serialize the updated parameters
        parameters_updated = ndarrays_to_parameters(updated_params)
        
        return FitRes(  
                parameters=parameters_updated, 
                num_examples=len(self.trainloader), 
                status=Status(code=Code.OK, message="success"),
                metrics={"pid": self.pid}
                )
    
         

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:

        server_round = ins.config["server_round"]
        
        # Deserialize the parameters to NumPy ndarray's
        ndarrays_original = parameters_to_ndarrays(ins.parameters)
        
        # Set the parameters to the model
        set_parameters(self.net, ndarrays_original)
        
        # Evaluate the model
        start_time = time.time()
        loss, accuracy = test(self.net, self.valloader, verbose=True)
        end_time = time.time()
        evaluate_time = end_time - start_time
        self.time_stats['evaluate'].append(evaluate_time)
        
        self.epoch_accuracy[server_round] = accuracy            
        if server_round == ins.config.get("total_rounds"):
            self.on_train_end()
            self._save_time_stats()  
        status = Status(code=Code.OK, message="success")
        return EvaluateRes(
            status = status,
            loss = float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)}
        )
    
    
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
    
    net = MODEL_MAP[MODEL_NAME]().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(DATASET_NAME, 
                                            CLIENT_NUMER, 
                                            BATCH_SIZE,
                                            partition_id=partition_id,
                                            federated=True
                                            )


    return FlowerClient(partition_id, net, trainloader, valloader).to_client()



def create_client_fn(batch_size, num_supernodes, model_name, dataset_name):
    """Return a *ClientApp* that Flower can call to construct clients."""
   
    def client_creation_func(context: Context) -> Client:
        # 原本 client_fn(context) 的逻辑
        context.node_config["model"] = model_name
        context.node_config["batch_size"] = batch_size
        context.node_config["num_supernodes"] = num_supernodes
        context.node_config["dataset"] = dataset_name
        return client_fn(context)

    # 给 ClientApp 的 `client_fn` 参数传一个函数，而不是一个已经实例化的 Client
    client_app = ClientApp(client_fn=client_creation_func)
    return client_app

# 在启动客户端之前，读取服务器地址
def get_server_address():
    with open("server_address.txt", "r") as f:
        return f.read().strip()

if __name__ == "__main__":
    """Create a Flower client representing a single organization."""
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DATASET_NAME = "cifar10"
    DATASET_NAME = "fashionmnist"
    num_classes = 10 if DATASET_NAME == "cifar10" or DATASET_NAME == "fashionmnist" else 100
    # net = MODEL_MAP['ResNet18'](num_classes).to(DEVICE)
    net = MODEL_MAP['LeNet5'](num_classes).to(DEVICE)
    
    
    
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--partition-id", type=int, default=0, help="Partition ID")
    parser.add_argument("--CLIENT_NUMER", type=int, default=10, help="Number of clients")
    
    args = parser.parse_args()
    partition_id = args.partition_id
    CLIENT_NUMER = args.CLIENT_NUMER
    
    # 加载数据集
    trainloader, valloader, _ = load_datasets(
        DATASET_NAME, 
        CLIENT_NUMER, 
        BATCH_SIZE,
        partition_id=partition_id,
        federated=True
    )
    
    # 获取服务器地址
    server_address = get_server_address()
    print(f"Connecting to server at {server_address}")
    
    # 创建客户端实例，提供所有必需的参数
    client = FlowerClient(
        pid=partition_id,
        net=net,
        trainloader=trainloader,
        valloader=valloader
    )
    
    # 启动客户端
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=1024*1024*1024
    )
    