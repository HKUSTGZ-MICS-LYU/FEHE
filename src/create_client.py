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
import time

 

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
        
        set_parameters(self.net, parameters)
        
        train_start = time.time()
        train(self.net, self.trainloader, epochs=config["local_epochs"], verbose=True)
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
    BATCH_SIZE = 32
    DATASET_NAME = "cifar10"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = MODEL_MAP['ResNet18']().to(DEVICE)
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
    