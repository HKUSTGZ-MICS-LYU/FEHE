import argparse
import itertools
import logging

import numpy as np
logging.basicConfig(level=logging.INFO) 
import sys
import flwr as fl
import os
from flwr.common import Metrics, Context
from flwr.client import Client, NumPyClient, ClientApp
import torch
from encryption import Enc_needed, param_decrypt, param_encrypt
from globals import MODEL_MAP
from utils import get_parameters, set_parameters
from train import train
from test import test
from dataloader import load_datasets

logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Include timestamp, level, and message
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)

class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.serialized_dataspace_log = []

    def get_parameters(self, config):
        print(f"GET PARAM: Client {self.pid}")
        return get_parameters(self.net)

    def fit(self, parameters, config):

        print(f"FIT: Client {self.pid}, Round {config['server_round']}, config: {config}")

        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=config["local_epochs"])
        updated_params = self.net.state_dict()
       
        if Enc_needed.encryption_needed.value == 1:
            _, serialized_dataspace = param_encrypt(updated_params, self.pid)
            print(f"Client {self.pid} serialized dataspace: {serialized_dataspace} MB")
            self.serialized_dataspace_log.append(serialized_dataspace)
                  
        return get_parameters(self.net), len(self.trainloader), {"pid": self.pid}

    def evaluate(self, parameters, config):        
        server_round = config["server_round"]
        print(f"EVALUATE: Client {self.pid}, server_round: {server_round}")
        
        if Enc_needed.encryption_needed.value == 1:
            print("Load weights from encrypted file")
            params_decrypted = param_decrypt(f"encrypted/aggregated_data_encrypted_{server_round}.txt")
            reshaped_params = []
            shapes = [np.shape(arr) for arr in parameters]
            current_index = 0
            for shape in shapes:
                data_result = []
                size = np.prod(shape)
                reshaped_arr = np.reshape(params_decrypted[current_index:current_index + size], shape)
                reshaped_params.append(reshaped_arr)
                current_index += size

            print(f"Load Weight: Client {self.pid} aggregated weights to the model")
            set_parameters(self.net, reshaped_params)
            loss, accuracy = test(self.net, self.valloader)
            print(f"Accuracy: Client {self.pid} accuracy: {accuracy}")
            return loss, len(self.valloader), {"accuracy": float(accuracy)}
        
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        print(f"Client {self.pid} accuracy: {accuracy}")
        return loss, len(self.valloader), {"accuracy": float(accuracy)}
    
def client_fn(context) -> Client:
    """Create a Flower client representing a single organization."""
    BATCH_SIZE = context.node_config.get("batch_size")
    CLIENT_NUMER = context.node_config.get("num_supernodes")
    MODEL_NAME = context.node_config.get("model")
    DATASET_NAME = context.node_config.get("dataset")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    net = MODEL_MAP[MODEL_NAME]().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(DATASET_NAME, 
                                            CLIENT_NUMER, 
                                            BATCH_SIZE,
                                            partition_id=partition_id,
                                            federated=True
                                            )


    print(f"Client {partition_id} created with {len(trainloader)} train and {len(valloader)} validation samples")
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


if __name__ == "__main__":
    
    """Create a Flower client representing a single organization."""
    BATCH_SIZE = 32
    DATASET_NAME = "fashionmnist"
 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = MODEL_MAP['LeNet5']().to(DEVICE)
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--partition-id", type=int, default=0, help="Partition ID")
    parser.add_argument("--CLIENT_NUMER", type=int, default=5, help="Number of clients")
    
    args = parser.parse_args()
    partition_id = args.partition_id
    CLIENT_NUMER = args.CLIENT_NUMER
    trainloader, valloader, _ = load_datasets(DATASET_NAME, 
                                            CLIENT_NUMER, 
                                            BATCH_SIZE,
                                            partition_id=partition_id,
                                            federated=True
                                            )


    print(f"Client {partition_id} created with {len(trainloader)} train and {len(valloader)} validation samples")
    

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(partition_id, net, trainloader, valloader),
        grpc_max_message_length=1024*1024*1024
    )
    