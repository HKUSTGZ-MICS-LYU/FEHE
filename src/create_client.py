import argparse
import numpy as np
import flwr as fl
from flwr.common import  Context
from flwr.client import Client, NumPyClient, ClientApp
import torch
from encryption import Enc_needed, param_decrypt, param_encrypt
from globals import MODEL_MAP
from utils import get_parameters, set_parameters
from train import train
from test import test
from dataloader import load_datasets
from Quantization import quantize_weights, dequantize_weights

 
Quantization_Bits = 8

class FlowerClient(NumPyClient):
    # The characteristics of the client are defined in the constructor
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.serialized_dataspace_log = []
        self.epoch_accuracy = {}
        self.scales = None
        self.min_vals = None

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=config["local_epochs"], verbose = False)
        updated_params = self.net.state_dict()

        # # Flatten the parameters and get the max and min values
        # flattened_params = []
        # for param in updated_params.values():
        #     flattened_params.extend(param.cpu().numpy().flatten().tolist())
        # local_max = max(flattened_params)
        # local_min = min(flattened_params)
    
        # # Get the global max and min values
        # global_max = config.get("global_max")
        # global_min = config.get("global_min")

        # # Get the quantization parameters
        # quant_max = global_max if (global_max is not None and global_min is not None) else local_max
        # quant_min = global_min if (global_max is not None and global_min is not None) else local_min

        # # Quantize the weights
        # quantized_params, scale, zero_point = quantize_weights(flattened_params, Quantization_Bits, quant_max, quant_min)
        
        if Enc_needed.encryption_needed.value == 1:
            _, serialized_dataspace, self.scales, self.min_vals = param_encrypt(updated_params, self.pid)
            self.serialized_dataspace_log.append(serialized_dataspace)
                  
        return get_parameters(self.net), len(self.trainloader), {"pid": self.pid}

    def evaluate(self, parameters, config):        
        server_round = config["server_round"]
        if Enc_needed.encryption_needed.value == 1:
            params_decrypted = param_decrypt(f"encrypted/aggregated_data_encrypted_{server_round}.txt", self.scales, self.min_vals)
            reshaped_params = []
            shapes = [np.shape(arr) for arr in parameters]
            current_index = 0
            for shape in shapes:
                size = np.prod(shape)
                reshaped_arr = np.reshape(params_decrypted[current_index:current_index + size], shape)
                reshaped_params.append(reshaped_arr)
                current_index += size

            set_parameters(self.net, reshaped_params)
            loss, accuracy = test(self.net, self.valloader)
            self.epoch_accuracy[server_round] = accuracy
            
            if server_round == config.get("total_rounds"):
                self.on_train_end()
            return loss, len(self.valloader), {"accuracy": float(accuracy)}
        
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        self.epoch_accuracy[server_round] = accuracy            
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
        
        self._save_to_csv()
    
    def _save_to_csv(self):
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


    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(partition_id, net, trainloader, valloader),
        grpc_max_message_length=1024*1024*1024
    )
    