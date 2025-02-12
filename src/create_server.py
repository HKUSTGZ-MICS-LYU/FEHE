import os
import sys
from flwr.common import *
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from quantization import Quantizer
from encryption import Enc_needed
import filedata as fd
import tenseal as ts
import socket
import time
from encryption import create_context
from utils import reshape_parameters


# Configure the fit parameters for clients 
def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 1,  
    }
    return config

# Configure the evaluation parameters for clients
def evaluate_config_factory(num_rounds: int):
    """ Return a function which configures the evaluation parameters for clients. """
    def _inner(server_round: int):
        return {
            "server_round": server_round,
            "total_rounds": num_rounds  
        }
    return _inner

class MyFlowerStrategy(FedAvg):
    """A custom Flower strategy extending FedAvg."""
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 5,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        num_rounds: int = 1,
    ) -> None:
        """A custom Flower strategy extending FedAvg."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        self.time_stats = {
            'aggregation': [],
            'quantization': [],
            'encryption': [],
            'decryption': [],
            'dequantization': [],
            'flattening': [],  # Time to flatten weights
            'statistics': [],  # Time to compute per-layer statistics
            'chunking': [],    # Time to split into chunks
            'reshaping': [],   # Time to reshape weights
        }
        self.global_max = float('-inf')  
        self.global_min = float('inf')   
        self.global_sigma = None
        self.global_mu = None
        self.all_sigma = []
        self.all_mu = []
        self.total_comm_bytes = 0
        self.num_rounds = num_rounds
        self.aggregated_path = None
        self.quantizer = Quantizer()
        self.chunk_size = 4096
        self.context = create_context()
    
    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return [], {}

    #     # Get layer structure from the first client
        
    # # 获取第一个客户端参数并打印信息
    #     first_client_params = parameters_to_ndarrays(results[0][1].parameters)
    #     print(f"\n=== Round {server_round} Layer Information ===")
    #     for idx, param in enumerate(first_client_params):
    #         print(f"Layer {idx}:")
    #         print(f"  Shape: {param.shape}")
    #         print(f"  Size: {param.size}")
    #         print(f"  Mean: {np.mean(param):.6f}")
    #         print(f"  Std: {np.std(param):.6f}")
    #         print("-" * 50)
    #     num_layers = len(first_client_params)
    #     layer_shapes = [param.shape for param in first_client_params]

    #     # Initialize storage for layer-wise processing
    #     aggregated_layers = []
    #     for layer_idx in range(num_layers):
    #         # Step 1: Flatten weights
    #         flatten_start = time.time()
    #         layer_data = []
    #         for client, fit_res in results:
    #             client_params = parameters_to_ndarrays(fit_res.parameters)
    #             layer_weights = client_params[layer_idx].flatten()
    #             layer_data.append(layer_weights)
    #         flatten_time = time.time() - flatten_start
    #         self.time_stats['flattening'].append(flatten_time)

    #         # Step 2: Compute per-layer statistics
    #         stats_start = time.time()
    #         layer_mu = np.mean([np.mean(w) for w in layer_data])
    #         layer_sigma = np.mean([np.std(w) for w in layer_data])
    #         layer_max = max([np.max(w) for w in layer_data])
    #         layer_min = min([np.min(w) for w in layer_data])
    #         stats_time = time.time() - stats_start
    #         self.time_stats['statistics'].append(stats_time)

    #         # Step 3: Quantize layer-wise
    #         quant_start = time.time()
    #         quantized_layer = []
    #         for client_weights in layer_data:
    #             quantized, params = self.quantizer.quantize_weights_unified(
    #                 client_weights,
    #                 n_bits=14,
    #                 method="naive",
    #                 global_max=layer_max,
    #                 global_min=layer_min,
    #                 mu=layer_mu,
    #                 sigma=layer_sigma,
    #                 sigma_bits=[8, 8, 8, 8, 8]
    #             )
    #             quantized_layer.append(quantized)
    #         quant_time = time.time() - quant_start
    #         self.time_stats['quantization'].append(quant_time)

    #         # Step 4: Encrypt layer-wise
    #         encrypt_start = time.time()
    #         encrypted_chunks = []
    #         for q_weights in quantized_layer:
    #             # Split into chunks, ensuring no empty chunks
    #             chunk_start = time.time()
    #             chunks = [q_weights[i * self.chunk_size:(i + 1) * self.chunk_size] 
    #                     for i in range((len(q_weights) + self.chunk_size - 1) // self.chunk_size)]
    #             chunks = [c for c in chunks if len(c) > 0]
    #             chunk_time = time.time() - chunk_start
    #             self.time_stats['chunking'].append(chunk_time)

    #             # Encrypt non-empty chunks
    #             encrypt_chunk_start = time.time()
    #             encrypted_chunks.append([ts.bfv_vector(self.context, c) for c in chunks])
    #             encrypt_chunk_time = time.time() - encrypt_chunk_start
    #             self.time_stats['encryption'].append(encrypt_chunk_time)
    #         encrypt_time = time.time() - encrypt_start
    #         self.time_stats['encryption'].append(encrypt_time)

    #         # Step 5: Aggregate encrypted chunks
    #         agg_start = time.time()
    #         agg_encrypted = [sum(chunks) for chunks in zip(*encrypted_chunks)]
    #         agg_time = time.time() - agg_start
    #         self.time_stats['aggregation'].append(agg_time)

    #         # Step 6: Decrypt
    #         decrypt_start = time.time()
    #         decrypted = [chunk.decrypt(self.context.secret_key()) for chunk in agg_encrypted]
    #         decrypted_flat = np.concatenate(decrypted) / len(results)
    #         decrypt_time = time.time() - decrypt_start
    #         self.time_stats['decryption'].append(decrypt_time)

    #         # Step 7: Dequantize
    #         dequant_start = time.time()
    #         dequantized = np.array(self.quantizer.dequantize_weights_unified(decrypted_flat, params))
    #         dequant_time = time.time() - dequant_start
    #         self.time_stats['dequantization'].append(dequant_time)

    #         # Step 8: Reshape and store
    #         reshape_start = time.time()
    #         aggregated_layers.append(dequantized.reshape(layer_shapes[layer_idx]))
    #         reshape_time = time.time() - reshape_start
    #         self.time_stats['reshaping'].append(reshape_time)

        aggregated_layers = parameters_to_ndarrays(results[0][1].parameters)
        # Convert back to Flower parameters
        return ndarrays_to_parameters(aggregated_layers), {}
    
   

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Called after evaluation. If it's the last round, we can print out the total comm.
        """
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # If we just finished the final round
        if server_round == self.num_rounds:
            print(f"[aggregate_evaluate] All {self.num_rounds} rounds completed.")
            # Print the total communication in GB
            print(f"Total communication = {self.total_comm_bytes / 1024**3} GB")
            # Create a csv file with the total communication
            total_comm_bytes_str = str(self.total_comm_bytes / 1024**3)
            with open("total_communication.csv", "w") as f:
                f.write(f"Total communication (GB),{total_comm_bytes_str}")
            f.close()
            # 保存聚合时间统计
            self._save_time_stats()
        return aggregated_metrics

    def _save_time_stats(self):
        """保存服务器端的时间统计"""
        stats_path = "server_time_stats.csv"
        try:
            with open(stats_path, "w") as f:
                f.write("operation,round,time\n")
                for operation, times in self.time_stats.items():
                    for i, time in enumerate(times):
                        f.write(f"{operation},{i},{time}\n")
            print(f"Server time statistics saved to {stats_path}")
        except Exception as e:
            print(f"Error saving server time statistics: {str(e)}")

def create_server_fn(num_rounds, min_fit_clients, min_evaluate_clients, min_available_clients) -> ServerApp:
    """
    Create a ServerApp which uses the given number of rounds.
    """
    def server_fn(context: Context, **kwargs) -> ServerAppComponents:
        config = ServerConfig(num_rounds=num_rounds)
        strategy = MyFlowerStrategy(
            fraction_fit=min_fit_clients/num_rounds,
            fraction_evaluate=min_evaluate_clients/num_rounds,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config_factory(num_rounds),
            num_rounds=num_rounds
        )         
        return ServerAppComponents(strategy=strategy, config=config)


    return ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    NUM_ROUNDS = 100
    NUM_CLIENTS = 1
    MIN_CLIENTS = 1
    
    # Get the hostname
    hostname = socket.gethostname()
    SERVER_ADDRESS = f"{hostname}:8080"
    print(f"Server will listen on {SERVER_ADDRESS}")
    
    my_strategy = MyFlowerStrategy(
        fraction_fit=MIN_CLIENTS/NUM_CLIENTS,  
        fraction_evaluate=MIN_CLIENTS/NUM_CLIENTS,  
        min_fit_clients=MIN_CLIENTS,      
        min_evaluate_clients=MIN_CLIENTS,  
        min_available_clients=MIN_CLIENTS, 
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config_factory(NUM_ROUNDS),
        num_rounds=NUM_ROUNDS
    )
    
  
    with open("server_address.txt", "w") as f:
        f.write(SERVER_ADDRESS)
    
    
    fl.server.start_server(
        server_address = SERVER_ADDRESS,  # 使用相同的地址
        config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        grpc_max_message_length = 1024 * 1024 * 1024,
        strategy = my_strategy,
    )