from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from encryption import Enc_needed
import filedata as fd
import tenseal as ts

# Configure the fit parameters for clients 
def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 30,  
    }
    return config

# Configure the evaluation parameters for clients
def evaluate_config(server_round: int):
    config = {
        "server_round": server_round,
    }
    return config

class MyFlowerStrategy(FedAvg):
    """A custom Flower strategy extending FedAvg."""
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 5,
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
        self.final_accuracies = {}
        
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Override aggregate_fit to implement custom aggregation logic.
        server_round: The current round number.
        results: List of tuples (client, FirResults) tuples.
        failures: List of failures.
        """
        print(f"\n[MyFlowerStrategy] Aggregating round {server_round} with {len(results)} results\n")
        
        if len(results) < self.min_available_clients:
            print(f"Not enough clients available (have {len(results)}, need {self.min_available_clients}). Skipping round {server_round}.")
            return None
        
        # Load public key to perform computations on encrypted data
        if Enc_needed.encryption_needed.value == 1:  # Full encryption is selected
            public_key_context = ts.context_from(fd.read_data("encrypted/public_key.txt")[0])
            aggregated_weights = None
        
            
            for client, weights in results:
                pid = weights.metrics.get("pid")
                encrypted_proto_list = fd.read_data(f"encrypted/data_encrypted_{pid}.txt")
                client_weights = []
                
                for encrypted_proto in encrypted_proto_list:
                    encrypted_params = ts.lazy_ckks_vector_from(encrypted_proto)
                    encrypted_params.link_context(public_key_context)
                    client_weights.append(encrypted_params)
                               
                if aggregated_weights is None:
                    aggregated_weights = client_weights
                else:
                    for i in range(len(client_weights)):
                        aggregated_weights[i] += client_weights[i]
                        
            # Average the weights
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] *= 1 / len(results)
            
            
            serialized_weights = [param.serialize() for param in aggregated_weights]
            fd.write_data(f"encrypted/aggregated_data_encrypted_{server_round}.txt", serialized_weights)

            # Continue with the aggregation with simulated data
            return super().aggregate_fit(server_round, results, failures)

            
        else: # No encryption is selected
            aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters


def create_server_fn(num_rounds, min_fit_clients, min_evaluate_clients, min_available_clients) -> ServerApp:
    """
    Create a ServerApp which uses the given number of rounds.
    """
    def server_fn(context: Context, **kwargs) -> ServerAppComponents:
        config = ServerConfig(num_rounds=num_rounds)
        strategy = MyFlowerStrategy(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
        )         
        return ServerAppComponents(strategy=strategy, config=config)


    return ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    my_strategy = MyFlowerStrategy(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
    fl.server.start_server(
        server_address = "localhost:8080",
        config = fl.server.ServerConfig(num_rounds=30),
        grpc_max_message_length = 1024 * 1024 * 1024,
        strategy = my_strategy,
    )