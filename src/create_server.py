# ====================== create_server.py ======================
"""Author: Meng Xiangchen"""
"""Secure federated learning server implementation with homomorphic encryption."""

# Standard library imports
import logging
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional

# Third party imports
import argparse
import flwr as fl
import numpy as np
import tenseal as ts
import torch
from flwr.common import Context, Scalar, Metrics
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

# Local imports
from utils import encryption
from utils import filedata as fd
from utils.quantization import Quantizer
import hashlib



@dataclass
class ServerConfig:
    """Server configuration parameters."""
    num_rounds:             int = 100
    
    min_clients:            int = 20
    min_evaluate_clients:   int = 20
    min_available_clients:  int = 50
    
    poly_modulus_degree:    int = 4096
    plain_modulus:          int = 1032193
    quant_bits:             int = 8
    quant_area:             int = 5
    quant_method:           str = "sigma"
    encrypted_dir:          str = "encrypted"
    round_timeout:          Optional[float] = None
    

class SecureAggregationStrategy(FedAvg):
    """Custom aggregation strategy with secure enclave operations."""
    def __init__(self, config: ServerConfig):
        def fit_config_with_rounds(server_round: int):
            return {
                "server_round": server_round,
                "local_epochs": 1,
                "num_rounds": config.num_rounds
            }

        def evaluate_config(server_round: int) -> Dict:
            """Return a configuration with static number of rounds."""
            return {
                "server_round": server_round,
            }
        
        super().__init__(
            fraction_fit            =   config.min_clients / config.min_available_clients,
            fraction_evaluate       =    config.min_evaluate_clients / config.min_available_clients,
            min_fit_clients         =   config.min_clients,
            min_evaluate_clients    =   config.min_evaluate_clients,
            min_available_clients   =   config.min_available_clients,
            initial_parameters      =   None,
            evaluate_fn             =   None,
            on_fit_config_fn        =   fit_config_with_rounds,
            on_evaluate_config_fn   =   evaluate_config,
        )
        self.config  = config
        self.context = encryption.create_context(
            config.poly_modulus_degree,
            config.plain_modulus
        )
        self._initialize_metrics()
        Path(config.encrypted_dir).mkdir(exist_ok=True)
        self.quantizer = Quantizer()
  

    def _initialize_metrics(self) -> None:
        """Initialize metrics tracking structures."""
        self.time_metrics = {
            "transmission": [],
            "quantization": [],
            "encryption": [],
            "aggregation": [],
            "decryption": [],
            "dequantization": []
        }
        self.total_communication = 0  
        self.evaluation_log = []
    
    
    def aggregate_fit(self, server_round: int, results, failurs):
        """Secure aggregation pipeline with homomorphic encryption."""
        
        # Phase 1: Parameter Collection
        client_params = self._collect_client_parameters(results)
        
        # Phase 2: Parameter Processing
        quantized_layers, non_quantized = self._categorize_parameters(client_params)
       
        # Phase 3: Secure Aggregation
        aggregated_params = self._process_quantized_layers(quantized_layers)
        aggregated_params.update(self._process_non_quantized(non_quantized))
     
        # Phase 4: Reshape and save aggregated parameters
        reshaped_params = OrderedDict()
        for name in aggregated_params:
            shape = next(p[name].shape for p in client_params)  
            reshaped_params[name] = aggregated_params[name].view(shape)
    
        torch.save(reshaped_params, f"{self.config.encrypted_dir}/aggregated_params.pth")  
              
        # Calculate hash of aggregated parameters
        hash_object = hashlib.sha256()
        with open(f"{self.config.encrypted_dir}/aggregated_params.pth", 'rb') as f:
            hash_object.update(f.read())
        param_hash = hash_object.hexdigest()

        # Write hash to file
        with open(f"{self.config.encrypted_dir}/aggregated_params.hash", 'w') as f:
            f.write(param_hash)
            
        return super().aggregate_fit(server_round, results, failurs)

    def _collect_client_parameters(self, results) -> List[Dict]:
        """Collect and time parameter loading."""
        start = time.time()
        params = [
            torch.load(f"{self.config.encrypted_dir}/client_{client.metrics['pid']}_params.pth", weights_only=True)
            for _, client in results
        ]
        self.time_metrics["transmission"].append(time.time() - start)
        return params
    
    def _categorize_parameters(self, params: List[Dict]) -> Tuple[Dict, Dict]:
        """Categorize parameters into quantizable and non-quantizable."""
        quantized = defaultdict(list)
        non_quantized = defaultdict(list)
        
        for param_dict in params:
            for name, tensor in param_dict.items():
                target = quantized if self._is_quantizable(name) else non_quantized
                target[name].append(tensor.cpu().numpy())
  
        return quantized, non_quantized
    
    def _is_quantizable(self, name: str) -> bool:
        """
        decide whether to quantize the layer
        only quantize the weights and biases of the convolutional layer and the fully connected layer
        """
        
        # identify target layers
        is_target_layer = any(key in name.lower() for key in ('conv', 'fc', 'linear'))
        # identify weights and biases
        is_weight_or_bias = any(key in name for key in ('weight', 'bias'))
        # exclude batch normalization layers
        is_excluded = any(key in name for key in ('bn', 'batch', 'running_', 'num_batches'))
        
        return is_target_layer and is_weight_or_bias and not is_excluded
            
    def _process_quantized_layers(self, layers: Dict) -> Dict:
        """Process quantized layers with full encryption pipeline."""
        processed = {}
    
        for name, weights in layers.items():
        
            # Quantization
            q_start = time.time()
            quantized, params = self._quantize_batch(weights)
            self.time_metrics["quantization"].append(time.time() - q_start)
            
            # Encryption
            e_start = time.time()
            encrypted = [self._encrypt(w) for w in quantized]
            self.time_metrics["encryption"].append(time.time() - e_start)
        
            
            # Obtain communication size of encrypted data (in bytes)
            # use sys to get the size of the encrypted data
            self.total_communication += sum(sys.getsizeof(e) for e in encrypted)
            
            # Aggregation
            a_start = time.time()
            aggregated = []
            for i in range(len(encrypted[0])):
                aggregated.append(sum(e[i] for e in encrypted))   
            self.time_metrics["aggregation"].append(time.time() - a_start)
            
            # Obtain communication size of aggregated data (in bytes)
            self.total_communication += sys.getsizeof(aggregated) * len(encrypted)
            
            # Decryption 
            d_start = time.time()
            decrypted = self._decrypt(aggregated)
            decrypted = decrypted / len(encrypted)
            
            # Dequantization
            dequantized = self._dequantize(decrypted, params)
            self.time_metrics["decryption"].append(time.time() - d_start)
                        
            processed[name] = torch.tensor(dequantized)
        return processed

    def aggregate_evaluate(
            self, 
            server_round: int,
            results, 
            failures
        ):
        """Handle evaluation results and save metrics."""
    
    
        self.config.current_round = server_round
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Compute average accuracy
        accuracies = [res.metrics['accuracy'] for _, res in results if 'accuracy' in res.metrics]
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        # Log evaluation metrics
        self.evaluation_log.append({
            'server_round': server_round,
            'accuracy': avg_accuracy,
            'loss': aggregated_loss,
        })
        
        if server_round == self.config.num_rounds:
            self._save_metrics()
            
        return aggregated_loss, aggregated_metrics
    
    def _process_non_quantized(self, layers: Dict) -> Dict:
        """Process non-quantized layers with simple averaging."""
        return {
            name: torch.tensor(np.mean(weights, axis=0))
            for name, weights in layers.items()
        }   

    def _quantize_batch(self, weights: List[np.ndarray]) -> Tuple[List, Dict]:
        """Quantize a batch of weights."""
        flat_weights = np.concatenate([w.flatten() for w in weights])
        global_max = np.max(flat_weights)
        global_min = np.min(flat_weights)
        global_mu = np.mean(flat_weights)
        global_sigma = np.std(flat_weights)
        
        quantized = []
        quant_params = None
        for weight in weights:
            q_weight, q_params = self.quantizer.quantize_weights_unified(
                weight.flatten(),
                n_bits=self.config.quant_bits,
                sigma_bits=[self.config.quant_bits]*self.config.quant_area,
                method=self.config.quant_method,
                global_max=global_max,
                global_min=global_min,
                global_mu=global_mu,
                global_sigma=global_sigma
            )
            quantized.append(q_weight)
            quant_params = q_params
        return quantized, quant_params

    def _encrypt(self, data: np.ndarray) -> ts.BFVVector:
        """Encrypt data using TenSEAL."""
        chunk_size = self.config.poly_modulus_degree
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return [ts.bfv_vector(self.context, chunk) for chunk in chunks]

    def _decrypt(self, encrypted: List[ts.BFVVector]) -> np.ndarray:
        """Decrypt encrypted data."""
        decrypted = []
        for chunk in encrypted:
            decrypted.extend(chunk.decrypt(self.context.secret_key()))
        return np.array(decrypted)

    def _dequantize(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """Dequantize data using quantization parameters."""
        return self.quantizer.dequantize_weights_unified(quantized, params)

    

    def _save_metrics(self) -> None:
        """Save all collected metrics to files."""
        try: 
            # Save time metrics
            with open(f"{self.config.encrypted_dir}/server_time_stats.csv", "w") as f:
                f.write("operation,time\n")
                for operation, times in self.time_metrics.items():
                    for t in times:
                        f.write(f"{operation},{t}\n")
            
            # Save communication stats
            with open(f"{self.config.encrypted_dir}/total_communication.csv", "w") as f:
                f.write(f"Total communication (GB),{self.total_communication / 1024 ** 3}")
                
            logging.info("All metrics saved successfully")
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")

def create_server_app(config: ServerConfig) -> fl.server.Server:
    """Create a new server app with the custom aggregation strategy."""
    return SecureAggregationStrategy(config)

def main():
    """Main server execution flow."""
    
    hostname = socket.gethostname()
    SERVER_ADDRESS = f"{hostname}:8080"
    print(f"Server will listen on {SERVER_ADDRESS}")
    
    with open("server_address.txt", "w") as f:
        f.write(SERVER_ADDRESS)
        
    server_config = ServerConfig()
    strategy = create_server_app(server_config)
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
        grpc_max_message_length=1024**3,
    )

if __name__ == "__main__":
    main()
    