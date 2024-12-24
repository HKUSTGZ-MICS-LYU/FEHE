from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from encryption import Enc_needed
import filedata as fd
import tenseal as ts


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 3,
    }
    return config

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
        public_key_context = ts.context_from(fd.read_data("encrypted/public_key.txt")[0])
        secret_key_context = ts.context_from(fd.read_data("encrypted/secret_key.txt")[0])
        
        if Enc_needed.encryption_needed.value == 1:  # Full encryption is selected
            aggregated_parameters = []
            results_ex = None
            num = 0
           
            # Aggregate the encrypted parameters
            print(f"Aggregating encrypted parameters for round {server_round}")
            for client, weights in results:
                num += 1
                pid = weights.metrics.get("pid")
                encrypted_proto_list = fd.read_data(f"encrypted/data_encrypted_{pid}.txt")
                client_weight = []
                print("Length of encrypted_proto_list: ", len(encrypted_proto_list))
                for encrypted_proto in encrypted_proto_list:
                    encrypted_tensor = ts.lazy_ckks_tensor_from(encrypted_proto)
                    encrypted_tensor.link_context(public_key_context)
                    client_weight.append(encrypted_tensor)
                    del encrypted_tensor
                if results_ex is None:
                    results_ex = client_weight
                else:
                    for i in range(len(client_weight)):
                        results_ex[i] += client_weight[i]
            
            print(f"Aggregated {num} encrypted parameters")
            # Decrypt the aggregated parameters
            for result_ex in results_ex:
                result_ex.link_context(secret_key_context)
                result_ex = np.array(result_ex.decrypt().raw) / num
                aggregated_parameters.append(result_ex)
            
            print(f"Decrypted the aggregated parameters")
            # Save the aggregated parameters to a file
            with open(f"encrypted/encrypted_aggregated_params_{server_round}.txt", "w") as f:
                for item in aggregated_parameters:
                    for subitem in item:
                        f.write("%s\n" % subitem)
            f.close()
                      

            # As Flower framework does not support CKKS encrypted objects, aggregation is by-passed with user-defined function
            # In order to continue simulation, aggregation is performed here with in-built functions
            aggregated_parameters = super().aggregate_fit(server_round, results, failures)
            del results_ex
            
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
        config = fl.server.ServerConfig(num_rounds=3),
        grpc_max_message_length = 1024 * 1024 * 1024,
        strategy = my_strategy,
    )