import os
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from encryption import Enc_needed
import filedata as fd
import tenseal as ts
import socket
import time




# Configure the fit parameters for clients 
def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 3,  
    }
    return config

# Configure the evaluation parameters for clients
def evaluate_config_factory(num_rounds: int):
    """ Return a function which configures the evaluation parameters for clients. """
    def _inner(server_round: int):
        return {
            "server_round": server_round,
            "total_rounds": num_rounds  # 注入总轮次
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
        self.aggregation_times = []
        self.global_max = None
        self.total_comm_bytes = 0
        self.num_rounds = None
        self.aggregated_path = None
    
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Override aggregate_fit to implement custom aggregation logic.
        server_round: The current round number.
        results: List of tuples (client, FirResults) tuples.
        failures: List of failures.
        """
        if len(results) < self.min_available_clients:
            print(f"Not enough clients available (have {len(results)}, need {self.min_available_clients}). Skipping round {server_round}.")
            return None
        
        # Load public key to perform computations on encrypted data
        if Enc_needed.encryption_needed.value == 1:  # Full encryption is selected
            public_key_context = ts.context_from(fd.read_data("encrypted/public_key.txt")[0])
            aggregated_weights = None
            
            for client, weights in results:
                pid = weights.metrics.get("pid")
                
                client_file_path = f"encrypted/data_encrypted_{pid}.txt"
                file_size = os.path.getsize(client_file_path)
                self.total_comm_bytes += file_size
                                
                encrypted_proto_list = fd.read_data(f"encrypted/data_encrypted_{pid}.txt")
                client_weights = []
                
 
                for encrypted_proto in encrypted_proto_list:
                    encrypted_params = ts.lazy_bfv_vector_from(encrypted_proto)
                    encrypted_params.link_context(public_key_context)
                    client_weights.append(encrypted_params)
                               
                if aggregated_weights is None:
                    aggregated_weights = client_weights
                else:
                    for i in range(len(client_weights)):
                        aggregated_weights[i] += client_weights[i]
                        
            # Write out aggregated file
            aggregated_file_path = f"encrypted/aggregated_data_encrypted_{server_round}.txt"
            serialized_weights = [param.serialize() for param in aggregated_weights]
            fd.write_data(aggregated_file_path, serialized_weights)


            # 3) Measure aggregated file size
            agg_file_size = os.path.getsize(aggregated_file_path)
            self.total_comm_bytes += agg_file_size
            
            # Continue with the aggregation with simulated data
            return super().aggregate_fit(server_round, results, failures)

        else: # No encryption is selected
            aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Called after evaluation. If it's the last round, we can print out the total comm.
        """
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        # If we just finished the final round
        if server_round == self.num_rounds:
            print(f"[aggregate_evaluate] All {self.num_rounds} rounds completed.")
            # Print the total communication in GB
            print(f"Total communication = {self.total_comm_bytes / 1e9} GB")
            # Create a csv file with the total communication
            total_comm_bytes_str = str(self.total_comm_bytes / 1e9)
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
                for round_num, t in enumerate(self.aggregation_times, 1):
                    f.write(f"aggregation,{round_num},{t}\n")
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
            on_evaluate_config_fn=evaluate_config_factory(num_rounds)
        )         
        return ServerAppComponents(strategy=strategy, config=config)


    return ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    NUM_ROUNDS = 5
    NUM_CLIENTS = 1
    MIN_CLIENTS = 1
    
    # 使用完整的主机名
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
        on_evaluate_config_fn=evaluate_config_factory(NUM_ROUNDS)
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