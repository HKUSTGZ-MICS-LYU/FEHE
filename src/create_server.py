from collections import OrderedDict, defaultdict
import os
from flwr.common import Context, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
import torch
import encryption
import filedata as fd
import tenseal as ts
import socket
import time
from typing import Dict, List, Tuple, Optional

from quantization import Quantizer


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
            "total_rounds": num_rounds  # 注入总轮次
        }
    return _inner

class MyFlowerStrategy(FedAvg):
    """A custom Flower strategy extending FedAvg."""
    def __init__(
        self,
        num_rounds: int,
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
        self.time_stats = {
            "transmission": [],
            "quantization": [],
            "encryption": [],
            "aggregation": [],
            "decryption": [],
            "dequantization": [],
        }
        self.poly_modulus_degree = 4096
        self.plain_modulus = 1032193
        self.evaluation_metrics = []  
        self.total_comm_bytes = 0
        self.num_rounds = num_rounds
        self.aggregated_path = f"encrypted/aggregated_params.pth"
        self.quantizer = Quantizer()
        self.n_bits = 8
        self.method = "naive"
        self.context = encryption.create_context(self.poly_modulus_degree, self.plain_modulus)
    
    def _quantize_weights(self, weights: List[np.ndarray], global_max: float, global_min: float) -> Tuple[List[np.ndarray], Dict]:
        """Quantize weights using the quantizer."""
  
        quantized_weights = []
        quant_params = None
        for weight in weights:
            weight_flat = weight.flatten()
            quantized_weight, quant_params = self.quantizer.quantize_weights_unified(
                weight_flat,
                n_bits=self.n_bits,
                method=self.method,
                global_max=global_max,
                global_min=global_min
            )
            quantized_weights.append(quantized_weight)
        return quantized_weights, quant_params

    def _encrypt_weights(self, quantized_weights: List[np.ndarray]) -> List[ts.BFVVector]:
        """Encrypt quantized weights using TenSEAL."""
        encrypted_weights = []
        for weight in quantized_weights:
            # Split weights into chunks if length exceeds polynomial degree
            chunk_size = self.poly_modulus_degree
            weight_chunks = [weight[i:i + chunk_size] for i in range(0, len(weight), chunk_size)]
            
            # Encrypt each chunk
            encrypted_chunks = []
            for chunk in weight_chunks:
                encrypted_chunk = ts.bfv_vector(self.context, chunk)
                encrypted_chunks.append(encrypted_chunk)
            encrypted_weights.extend(encrypted_chunks)
        return encrypted_weights

    def _decrypt_weights(self, encrypted_weights: List[ts.BFVVector]) -> np.ndarray:
        """Decrypt encrypted weights using TenSEAL."""
        decrypted_weights = []
        for encrypted_weight in encrypted_weights:
            decrypted_chunk = encrypted_weight.decrypt(self.context.secret_key())
            decrypted_weights.extend(decrypted_chunk)
        return np.array(decrypted_weights)

    def _dequantize_weights(self, quantized_weights: np.ndarray, quant_params: Dict) -> np.ndarray:
        """Dequantize weights using the quantizer."""
        return np.array(self.quantizer.dequantize_weights_unified(quantized_weights, quant_params))

    def aggregate_fit(self, server_round: int, results, failures):
        
        # Get the transmission time and load client parameters
        transmission_time = time.time()
        client_params = [
            torch.load(f"encrypted/client_{client.metrics.get('pid')}_params.pth", weights_only=False)
            for _, client in results
        ]
        transmission_time = time.time() - transmission_time
        self.time_stats["transmission"].append(transmission_time)

        # Initialize the aggregated parameters
        quantized_layers = {}
        non_quantized_layers = defaultdict(list)
        quantization_time = time.time()
        # Pre-compile the name check
        is_quantizable = lambda name: (
            ('conv' in name or 'fc' in name)
            and 'running_mean' not in name
            and 'running_var' not in name
            and 'num_batches_tracked' not in name
        )
        # Process parameters in batch
        for params in client_params:
            for name, param in params.items():
                # Convert to numpy array once
                weight = param.cpu().detach().numpy()
                if is_quantizable(name):
                    if name not in quantized_layers:
                        quantized_layers[name] = {
                            'weights': [],
                            'shape': weight.shape
                        }
                    quantized_layers[name]['weights'].append(weight)
                else:
                    non_quantized_layers[name].append(weight)    
        quantization_time = time.time() - quantization_time
        self.time_stats["quantization"].append(quantization_time)
        

        # Process quantized layers
        final_state_dict = OrderedDict()
      
        for name, layer in quantized_layers.items():
            # Calculate global max and min
            all_weights = np.concatenate([w.flatten() for w in layer['weights']])
            global_max = np.max(all_weights)
            global_min = np.min(all_weights)
            shape = layer['shape']
            encrypted_weights = None
    
            for weight in layer['weights']:
        
                # Quantize weight
                quant_start = time.time()
                quantized_weight, quant_params = self._quantize_weights(weight, global_max, global_min)
                self.time_stats["quantization"].append(time.time() - quant_start)
        
    
                # Encrypt weights
                enc_start = time.time()
                encrypted_weight = self._encrypt_weights(quantized_weight)
                self.time_stats["encryption"].append(time.time() - enc_start)
                
                # Aggregate encrypted weights
                if encrypted_weights is None:
                    encrypted_weights = encrypted_weight
                else:
                    for i, weight in enumerate(encrypted_weight):
                        encrypted_weights[i] += weight
                
         
            # Decrypt weights
            dec_start = time.time()
            decrypted_weights = self._decrypt_weights(encrypted_weights)
            self.time_stats["decryption"].append(time.time() - dec_start)
            
            decrypted_weights = decrypted_weights / len(layer['weights'])
            
            # Dequantize weights
            dequant_start = time.time()
            dequantized_weights = self._dequantize_weights(decrypted_weights, quant_params)
            self.time_stats["dequantization"].append(time.time() - dequant_start)
            
   
            # Reshape and store in final state dict
            if dequantized_weights.size != np.prod(shape):
                raise ValueError(f"Size mismatch: expected {np.prod(shape)}, got {dequantized_weights.size}")
            final_state_dict[name] = torch.tensor(dequantized_weights.reshape(shape))
        
        # Process non-quantized layers (direct averaging)
        for name, weights in non_quantized_layers.items():
            final_state_dict[name] = torch.tensor(np.mean(weights, axis=0))
        
        # Compare the final state dict with the original state dict
       
        
        # Save aggregated parameters
        torch.save(final_state_dict, self.aggregated_path)
        
        # Call parent's aggregate method
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Called after evaluation. If it's the last round, we can print out the total comm.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Compute the average accuracy
        accuracy = []
        for _, res in results:
            if res.metrics and 'accuracy' in res.metrics:
                accuracy.append(res.metrics['accuracy'])
                
        avg_accuracy = np.mean(accuracy)
        
        self.evaluation_metrics.append({
            'server_round': server_round,
            'accuracy': avg_accuracy,
            'loss': aggregated_loss,
        })
   
        
        if server_round == self.num_rounds:
            self._save_communication()
            self._save_time_stats()
            self._save_evaluation_metrics()
            
        return aggregated_loss, aggregated_metrics
    
    def _save_evaluation_metrics(self):
        """save the evaluation metrics to a CSV file."""
        try:
            with open("evaluation_metrics.csv", "w") as f:
                f.write("round,accuracy,loss\n")
                for metrics in self.evaluation_metrics:
                    f.write(f"{metrics['server_round']},{metrics['accuracy']},{metrics['loss']}\n")
            print("Evaluation metrics saved successfully")
        except Exception as e:
            print(f"Error saving evaluation metrics: {str(e)}")

    def _save_communication(self):
        """Save the communication statistics to a CSV file."""
        total_comm_bytes_str = str(self.total_comm_bytes / 1024 ** 3)
        with open("total_communication.csv", "w") as f:
            f.write(f"Total communication (GB),{total_comm_bytes_str}")
        f.close()
        
    def _save_time_stats(self):
        """save the server time statistics to a CSV file."""
        stats_path = "server_time_stats.csv"
        with open(stats_path, "w") as f:
            f.write("operation,time\n")
            for operation, times in self.time_stats.items():
                for t in times:
                    f.write(f"{operation},{t}\n")
        f.close()
        print(f"Time statistics saved to {stats_path}")
        

def create_server_fn(num_rounds, min_fit_clients, min_evaluate_clients, min_available_clients) -> ServerApp:
    """
    Create a ServerApp which uses the given number of rounds.
    """
    def server_fn(context: Context, **kwargs) -> ServerAppComponents:
        config = ServerConfig(num_rounds=num_rounds)
        strategy = MyFlowerStrategy(
            num_rounds=num_rounds,
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
    NUM_ROUNDS = 100
    NUM_CLIENTS = 1
    MIN_CLIENTS = 1
    
    # 使用完整的主机名
    hostname = socket.gethostname()
    SERVER_ADDRESS = f"{hostname}:8080"
    print(f"Server will listen on {SERVER_ADDRESS}")
    
    my_strategy = MyFlowerStrategy(
        num_rounds=NUM_ROUNDS,
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
        strategy = my_strategy
    )
    
    
