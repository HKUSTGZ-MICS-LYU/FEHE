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
            'dequantization': []
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
        """
        Override aggregate_fit to implement custom aggregation logic.
        server_round: The current round number.
        results: List of tuples (client, FirResults) tuples.
        failures: List of failures.
        """
        if not results:
            return [], {}
    
        
        first_client_params = parameters_to_ndarrays(results[0][1].parameters)
        
        client_flatten_weights = []
        
        for client, fit_res in results:
            pid = fit_res.metrics["pid"]
         
            client_params = parameters_to_ndarrays(fit_res.parameters)
            flattened_params = np.concatenate([param.flatten() for param in client_params])
            # get the sigma for each client
            self.all_sigma.append(np.std(flattened_params))
            self.all_mu.append(np.mean(flattened_params))
            self.global_max = max(self.global_max, np.max(flattened_params))
            self.global_min = min(self.global_min, np.min(flattened_params))
            
            client_flatten_weights.append({
                                            "pid": pid, 
                                            "weights": flattened_params
                                           })
            with open(f"client_{pid}_params.txt", "w") as f:
                for param in flattened_params:
                    f.write(f"{param}\n")
            f.close()
            
        self.global_mu = np.mean(self.all_mu)
        self.global_sigma = np.mean(self.all_sigma)
        with open("global_mu_sigma.txt", "w") as f:
            f.write(f"global_mu: {self.global_mu}\n")
            f.write(f"global_sigma: {self.global_sigma}\n")
        f.close()


        # quantize the weights
        quant_start = time.time()
        quantized_weights = []
        for client_weight in client_flatten_weights:
            quantized_weight, params = self.quantizer.quantize_weights_unified(client_weight["weights"], 
                                                                               n_bits= 8,
                                                                               method = "sigma", 
                                                                               global_max = self.global_max, 
                                                                               global_min = self.global_min,
                                                                               block_size = 128,
                                                                               sigma_bits = [14,14,14,14],
                                                                               mu = self.global_mu,
                                                                               sigma = self.global_sigma)
            quantized_weights.append({"pid": client_weight["pid"], 
                                      "weights": quantized_weight
                                      })
            with open(f"client_{client_weight['pid']}_quantized_params.txt", "w") as f:
                for param in quantized_weight:
                    f.write(f"{param}\n")
            f.close()
            
            
        quant_time = time.time() - quant_start
        self.time_stats['quantization'].append(quant_time)
            
        # encrypt the quantized weights 
        encrypte_time = time.time()
        encrypted_weights = []
        for quantized_weight in quantized_weights:
            encrypted_weight = []
            # need to splite the quantized weights into chunks, each chunk length is 4096
            weight = quantized_weight["weights"]
            num_chunks = len(weight) // self.chunk_size
            if len(weight) % self.chunk_size != 0:
                num_chunks += 1
            for i in range(num_chunks):
                chunk = weight[i * self.chunk_size: (i + 1) * self.chunk_size]
                encrypted_weight.append(ts.bfv_vector(self.context, chunk))
            encrypted_weights.append({"pid": quantized_weight["pid"], "weights": encrypted_weight})
        encrypte_time = time.time() - encrypte_time
        self.time_stats['encryption'].append(encrypte_time)
        encrypted_weights_bytes = sys.getsizeof(encrypted_weights)
        self.total_comm_bytes += encrypted_weights_bytes
        
        # aggregate the encrypted weights
        aggregate_time = time.time()
        aggregated_weights = None
        for encrypted_weight in encrypted_weights:
            if aggregated_weights is None:
                aggregated_weights = encrypted_weight["weights"]
            else:
                for i in range(len(encrypted_weight["weights"])):
                    aggregated_weights[i] += encrypted_weight["weights"][i]
        aggregate_time = time.time() - aggregate_time
        self.time_stats['aggregation'].append(aggregate_time)
        aggregate_weights_bytes_size = sys.getsizeof(aggregated_weights)
        self.total_comm_bytes += aggregate_weights_bytes_size
                
        
        # decrypt the aggregated weights
        decrypte_time = time.time()
        decrypted_weights = []
        for encrypted_weight in aggregated_weights:
            decrypted_weight = encrypted_weight.decrypt(secret_key = self.context.secret_key())
            decrypted_weights.append(decrypted_weight)
        decrypted_weights = np.concatenate(decrypted_weights)
        decrypte_time = time.time() - decrypte_time
        self.time_stats['decryption'].append(decrypte_time)

        
        
        decrypted_weights = [param / len(results) for param in decrypted_weights]
        
        with open(f"decrypted_params_{pid}.txt", "w") as f:
            for param in decrypted_weights:
                f.write(f"{param}\n")
        f.close()
        
        # dequantize the decrypted weights
        dequantize_time = time.time()
        dequantized_weights = self.quantizer.dequantize_weights_unified(decrypted_weights, params = params)
        dequantize_time = time.time() - dequantize_time
        self.time_stats['dequantization'].append(dequantize_time)

        with open(f"dequantized_params_{pid}.txt", "w") as f:
            for param in dequantized_weights:
                f.write(f"{param}\n")
        f.close()
        
        # reshape the weights
        aggregated_parameters = reshape_parameters(first_client_params, dequantized_weights)            
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters)
        return aggregated_parameters, {}
            

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
    NUM_ROUNDS = 10
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
    
    # 保存完整的服务器地址
    with open("server_address.txt", "w") as f:
        f.write(SERVER_ADDRESS)
    
    # 启动服务器时使用完整地址
    fl.server.start_server(
        server_address = SERVER_ADDRESS,  # 使用相同的地址
        config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        grpc_max_message_length = 1024 * 1024 * 1024,
        strategy = my_strategy,
    )