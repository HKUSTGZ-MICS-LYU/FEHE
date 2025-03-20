"""
Author: Xiangchen Meng
"""
# Standard Library
import argparse
import os
import socket
import time
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib

# Multiprocessing
from multiprocessing import Process

# Third Party Libraries
import numpy as np
import flwr as fl
import numpy as np
import tenseal as ts
import torch
import torch.backends.cudnn as cudnn
from flwr.common import Context, NDArrays, Scalar, Metrics
from flwr.server.strategy import FedAvg
from flwr.client import NumPyClient

# Local Libraries
from create_client import load_model
from utils import encryption
from utils import filedata as fd
from utils.quantization import Quantizer
import hashlib
import utils
from utils.dataloader import load_datasets
from utils.encryption import *
from utils.quantization import *
from utils import *

from models import (
    LeNet5, ResNet18, VGG, PreActResNet18, GoogLeNet, DenseNet121,
    ResNeXt29_2x64d, MobileNet, MobileNetV2, DPN92, ShuffleNetG2,
    SENet18, ShuffleNetV2, EfficientNetB0, RegNetX_200MF, SimpleDLA
)
from utils.test import test
from utils.train import train
from utils.utils import get_parameters, set_parameters



"""
Server Side Implementation
"""

@dataclass
class ServerConfig:
    """Server configuration parameters."""
    num_rounds:             int = 3  # 为演示，默认改小一些
    min_clients:            int = 2
    min_evaluate_clients:   int = 2
    min_available_clients:  int = 2
    poly_modulus_degree:    int = 4096
    plain_modulus:          int = 1032193
    quant_bits:             int = 8
    quant_area:             int = 4
    quant_method:           str = "sigma"
    encrypted_dir:          str = "encrypted"
    round_timeout:          Optional[float] = None

class SecureAggregationStrategy(FedAvg):
    """自定义聚合策略，示例版，主要展示如何把原 create_server.py 的逻辑合并进来。"""
    def __init__(self, config: ServerConfig):
        def fit_config_with_rounds(server_round: int):
            return {
                "server_round": server_round,
                "local_epochs": 10,  # 演示, 这里写1
                "num_rounds": config.num_rounds,
                "total_clients": config.min_available_clients,
                "selected_clients": config.min_clients,
            }

        def evaluate_config(server_round: int) -> Dict:
            """Return a configuration with static number of rounds."""
            return {
                "server_round": server_round,
            }
        
        super().__init__(
            fraction_fit            = config.min_clients / config.min_available_clients,
            fraction_evaluate       = config.min_evaluate_clients / config.min_available_clients,
            min_fit_clients         = config.min_clients,
            min_evaluate_clients    = config.min_evaluate_clients,
            min_available_clients   = config.min_available_clients,
            initial_parameters      = None,
            evaluate_fn             = None,
            on_fit_config_fn        = fit_config_with_rounds,
            on_evaluate_config_fn   = evaluate_config,
        )
        self.config  = config
        self.context = create_context(config.poly_modulus_degree, config.plain_modulus)
        self._initialize_metrics()
        Path(config.encrypted_dir).mkdir(exist_ok=True)
        self.quantizer = Quantizer()
        self.model_name = None
        self.dataset_name = None
        self.IID = None

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

    def aggregate_fit(self, server_round: int, results, failures):
        """服务端聚合逻辑(示例)，用于 homomorphic encryption 相关处理。"""
        
        # 识别客户端ID
        client_ids = []
        for _, client in results:
            client_ids.append(client.metrics['pid'])
            if self.model_name is None:
                self.model_name = client.metrics['model']
                self.dataset_name = client.metrics['dataset']
                self.IID = client.metrics['IID']
        
        # 写出本次聚合用到的客户端ID文件
        with open(f"{self.config.encrypted_dir}/aggregate_client_ids.txt", "w") as f:
            for client_id in client_ids:
                f.write(f"{client_id}\n")
        
        # 1. 收集客户端参数
        client_params = self._collect_client_parameters(results)
        
        # 2. 区分可量化和不可量化参数
        quantized_layers, non_quantized = self._categorize_parameters(client_params)
       
        # 3. 安全聚合
        aggregated_params = self._process_quantized_layers(quantized_layers)
        aggregated_params.update(self._process_non_quantized(non_quantized))
     
        # 4. Reshape并保存聚合结果
        reshaped_params = OrderedDict()
        for name in aggregated_params:
            shape = next(p[name].shape for p in client_params)  
            reshaped_params[name] = aggregated_params[name].view(shape)
        
        torch.save(reshaped_params, f"{self.config.encrypted_dir}/aggregated_params.pth")  
              
        # 计算并保存哈希
        hash_object = hashlib.sha256()
        with open(f"{self.config.encrypted_dir}/aggregated_params.pth", 'rb') as f:
            hash_object.update(f.read())
        param_hash = hash_object.hexdigest()
        with open(f"{self.config.encrypted_dir}/aggregated_params.hash", 'w') as f:
            f.write(param_hash)
            
        return super().aggregate_fit(server_round, results, failures)

    def _collect_client_parameters(self, results) -> List[Dict]:
        """从加密目录收集所有客户端参数。"""
        start = time.time()
        params = []
        for _, client in results:
            pid = client.metrics['pid']
            pth = f"{self.config.encrypted_dir}/client_{pid}_params.pth"
            # 演示直接用 torch.load 加载 state_dict
            model_state = torch.load(pth, map_location="cpu")
            # 注意：这里假定 model_state 就是一个 dict
            params.append(model_state)
        self.time_metrics["transmission"].append(time.time() - start)
        return params
    
    def _categorize_parameters(self, params: List[Dict]) -> Tuple[Dict, Dict]:
        """区分要量化的和不量化的层(示例)。"""
        quantized = defaultdict(list)
        non_quantized = defaultdict(list)
        
        for param_dict in params:
            for name, tensor in param_dict.items():
                target = quantized if self._is_quantizable(name) else non_quantized
                target[name].append(tensor.cpu().numpy())
        return quantized, non_quantized
    
    def _is_quantizable(self, name: str) -> bool:
        """决定是否量化某一层(示例)。"""
        is_target_layer = any(key in name.lower() for key in ('conv', 'fc', 'linear'))
        is_weight_or_bias = any(key in name for key in ('weight', 'bias'))
        is_excluded = any(key in name for key in ('bn', 'batch', 'running_', 'num_batches'))
        return is_target_layer and is_weight_or_bias and not is_excluded
            
    def _process_quantized_layers(self, layers: Dict) -> Dict:
        """对可量化层执行量化、加密、聚合、解密、反量化等操作(示例)。"""
        processed = {}
        for name, weights in layers.items():
            # 量化
            q_start = time.time()
            quantized, params = self._quantize_batch(weights)
            self.time_metrics["quantization"].append(time.time() - q_start)
            
            # 加密
            e_start = time.time()
            encrypted = [self._encrypt(w) for w in quantized]
            self.time_metrics["encryption"].append(time.time() - e_start)
            
            # 统计通信量(示例)
            self.total_communication += sum(sys.getsizeof(e) for e in encrypted)
            
            # 聚合
            a_start = time.time()
            aggregated = []
            for i in range(len(encrypted[0])):
                aggregated.append(sum(e[i] for e in encrypted))   
            self.time_metrics["aggregation"].append(time.time() - a_start)
            
            self.total_communication += sys.getsizeof(aggregated) * len(encrypted)
            
            # 解密 & 反量化
            d_start = time.time()
            decrypted = self._decrypt(aggregated)
            decrypted = decrypted / len(encrypted)
            dequantized = self._dequantize(decrypted, params)
            self.time_metrics["decryption"].append(time.time() - d_start)
            
            processed[name] = torch.tensor(dequantized)
        return processed

    def _process_non_quantized(self, layers: Dict) -> Dict:
        """对不量化的层做简单平均(示例)。"""
        return {
            name: torch.tensor(np.mean(weights, axis=0))
            for name, weights in layers.items()
        }

    def _quantize_batch(self, weights: List[np.ndarray]) -> Tuple[List, Dict]:
        """对一批权重进行量化(示例)。"""
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

    def _encrypt(self, data: np.ndarray) -> List[ts.BFVVector]:
        """使用TenSEAL对数据进行BFV加密(示例)。"""
        chunk_size = self.config.poly_modulus_degree
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return [ts.bfv_vector(self.context, chunk) for chunk in chunks]

    def _decrypt(self, encrypted: List[ts.BFVVector]) -> np.ndarray:
        """解密(示例)。"""
        decrypted = []
        for chunk in encrypted:
            decrypted.extend(chunk.decrypt(self.context.secret_key()))
        return np.array(decrypted)

    def _dequantize(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """反量化(示例)。"""
        return self.quantizer.dequantize_weights_unified(quantized, params)

    def aggregate_evaluate(self, server_round: int, results, failures):
        """聚合评估逻辑(示例)。"""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 计算平均accuracy(示例)
        accuracies = [res.metrics['accuracy'] for _, res in results if 'accuracy' in res.metrics]
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        self.evaluation_log.append({
            'server_round': server_round,
            'accuracy': avg_accuracy,
            'loss': aggregated_loss,
        })

        if server_round == self.config.num_rounds:
            self._save_metrics()
            
        return aggregated_loss, aggregated_metrics
    
    def _save_metrics(self) -> None:
        """保存日志(示例)。"""
        try:
            Path("Experiment").mkdir(exist_ok=True)

            time_pth = f"Experiment/server_time_stats.csv"
            with open(time_pth, "w") as f:
                f.write("operation,time\n")
                for operation, times in self.time_metrics.items():
                    for t in times:
                        f.write(f"{operation},{t}\n")

            comm_pth = f"Experiment/server_communication.csv"
            with open(comm_pth, "w") as f:
                f.write(f"Total communication (GB),{self.total_communication / 1024 ** 3}")

            logging.info("All metrics saved successfully.")
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")

def create_server_app(config: ServerConfig) -> fl.server.Server:
    """创建自定义策略的 fl.server.Server 实例。"""
    return SecureAggregationStrategy(config)


def run_server(args):
    """
    封装：启动服务器流程。
    """
    hostname = socket.gethostname()
    SERVER_ADDRESS = f"{hostname}:8080"
    print(f"[Server] Server will listen on {SERVER_ADDRESS}")

    # 写入文件，客户端会读取这个地址
    with open("server_address.txt", "w") as f:
        f.write(SERVER_ADDRESS)

    server_config = ServerConfig(
        num_rounds = args.num_rounds,
        min_clients = args.min_clients,
        min_evaluate_clients = args.min_clients,
        min_available_clients = args.min_available_clients,
        # 如需更多字段，请这里赋值
    )
    strategy = create_server_app(server_config)

    # 启动
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,  # 仅在 Flower <1.2 时有用; 新版可不传
        strategy=strategy,
        grpc_max_message_length=1024**3,
    )
    
    
"""
Client Side Implementation
"""


@dataclass
class ClientConfig:
    partition_id: int 
    client_number: int 
    lr: float = 0.01
    min_lr: float = 1e-6
    scheduler: str = "cosine"
    optimizer: str = "adam"
    batch_size: int = 128
    IID: bool = True
    alpha: float = 1.0
    model_name: str = "SimpleNet"
    dataset_name: str = "FASHIONMNIST"
    encrypted_dir: str = "encrypted"
    total_clients: int = field(default=2)
    selected_clients: int = field(default=2)



class SecureClient(NumPyClient):
    """整合了加密与量化处理的客户端示例。"""
    def __init__(self,
                 config: ClientConfig,
                 model: torch.nn.Module,
                 trainloader,
                 testloader,
                 valloader):
        self.config = config
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(config.encrypted_dir).mkdir(parents=True, exist_ok=True)

        self.accuracy_log: Dict[int, float] = {}
        self.time_metrics: Dict[str, List[float]] = {
            "train": [],
            "evaluate": []
        }

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """客户端本地训练 + 保存加密前的参数到文件"""
        set_parameters(self.model, parameters)

        # 更新/合并 fit_config
        full_config = {
            "lr": self.config.lr,
            "scheduler": self.config.scheduler,
            "optimizer": self.config.optimizer,
            **config
        }
        # 训练
        t0 = time.time()
        self.model = train(
            net=self.model,
            trainloader=self.trainloader,
            epochs=full_config.get("local_epochs", 1),
            config=full_config,
            current_round=full_config.get("server_round", 0),
            total_rounds=full_config.get("num_rounds", 10),
            verbose=False
        )
        train_time = time.time() - t0
        self.time_metrics["train"].append(train_time)

        # 保存到文件（明文权重），供服务器再去加载做同态加密
        params_path = f"{self.config.encrypted_dir}/client_{self.config.partition_id}_params.pth"
        torch.save(self.model.state_dict(), params_path)

        return (
            get_parameters(self.model),
            len(self.trainloader),
            {
                "pid": self.config.partition_id,
                "model": self.config.model_name,
                "dataset": self.config.dataset_name,
                "IID": self.config.IID
            }
        )

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """客户端验证环节示例，可选择是否加载服务器聚合后的参数再测。"""
        # 先判断自己是不是参加了聚合
        clients_id = []
        agg_ids_path = f"{self.config.encrypted_dir}/aggregate_client_ids.txt"
        if os.path.exists(agg_ids_path):
            with open(agg_ids_path, "r") as f:
                for line in f:
                    clients_id.append(line.strip())

        # 如果此客户端在 server 聚合列表中，则加载 aggregated_params 进行评测
        if str(self.config.partition_id) in clients_id:
            aggregated_path = f"{self.config.encrypted_dir}/aggregated_params.pth"
            hash_path = f"{self.config.encrypted_dir}/aggregated_params.hash"
            if os.path.exists(aggregated_path) and os.path.exists(hash_path):
                with open(hash_path, 'r') as f:
                    expected_hash = f.read().strip()
                with open(aggregated_path, 'rb') as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError("Parameter file hash mismatch - possible tampering detected")

                # 加载聚合后参数
                self.model.load_state_dict(torch.load(aggregated_path, map_location=self.device))
            else:
                # 如果还没聚合成功，或者文件缺失，就用传进来的 parameters
                set_parameters(self.model, parameters)
        else:
            # 不在聚合列表中，则直接用本地最新 parameters
            set_parameters(self.model, parameters)

        # 测试
        t0 = time.time()
        loss_val, accuracy = test(self.model, self.valloader, verbose=False)
        eval_time = time.time() - t0
        self.time_metrics["evaluate"].append(eval_time)

        # 记录
        round_idx = config.get("server_round", -1)
        self.accuracy_log[round_idx] = accuracy

        # 假设在最后一轮时保存各种本地统计
        if round_idx == self.config.client_number:  # 或者你想用 num_rounds=xxx
            self._finalize_training()

        return loss_val, len(self.valloader), {"accuracy": float(accuracy)}

    def _finalize_training(self):
        """在训练结束时保存日志等。"""
        print(f"\nClient {self.config.partition_id} Accuracy Report:")
        for rnd, acc in sorted(self.accuracy_log.items()):
            print(f" Round {rnd}: {acc:.4f}")

        self._save_accuracy_csv()
        self._save_time_stats()

    def _save_accuracy_csv(self):
        """保存 accuracy 到 CSV。"""
        if self.config.IID:
            csv_path = f"Experiment/{self.config.model_name}_{self.config.dataset_name}/{self.config.total_clients}_{self.config.selected_clients}/IID/client_{self.config.partition_id}_accuracy.csv"
        else:
            csv_path = f"Experiment/{self.config.model_name}_{self.config.dataset_name}/{self.config.total_clients}_{self.config.selected_clients}/NONIID/client_{self.config.partition_id}_accuracy.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        try:
            with open(csv_path, "w") as f:
                f.write("round,accuracy\n")
                for rnd, acc in sorted(self.accuracy_log.items()):
                    f.write(f"{rnd},{acc}\n")
            print(f"[Client {self.config.partition_id}] Accuracy log saved to {csv_path}")
        except Exception as e:
            print(f"[Client {self.config.partition_id}] Error saving accuracy log: {str(e)}")

    def _save_time_stats(self):
        """保存时间消耗到 CSV。"""
        if self.config.IID:
            stats_path = f"Experiment/{self.config.model_name}_{self.config.dataset_name}/{self.config.total_clients}_{self.config.selected_clients}/IID/client_{self.config.partition_id}_time_stats.csv"
        else:
            stats_path = f"Experiment/{self.config.model_name}_{self.config.dataset_name}/{self.config.total_clients}_{self.config.selected_clients}/NONIID/client_{self.config.partition_id}_time_stats.csv"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        try:
            with open(stats_path, "w") as f:
                f.write("operation,round,time\n")
                # 这里只是示例，round 索引可以自己定义
                for operation, times in self.time_metrics.items():
                    for idx, t in enumerate(times, 1):
                        f.write(f"{operation},{idx},{t}\n")
            print(f"[Client {self.config.partition_id}] Time stats saved to {stats_path}")
        except Exception as e:
            print(f"[Client {self.config.partition_id}] Error saving time stats: {str(e)}")

def get_server_address() -> str:
    """读取 server_address.txt，得到服务器地址。"""
    with open("server_address.txt") as f:
        return f.read().strip()

def run_client(args, partition_id: int):
    """
    启动某个客户端的流程。
    """
    config = ClientConfig(
        partition_id   = partition_id,
        client_number  = args.num_rounds,  # 这里仅示例，你可以自定义
        lr            = args.lr,
        min_lr        = 1e-6,
        scheduler     = "cosine",
        optimizer     = "adam",
        batch_size    = args.batch_size,
        IID           = args.iid,
        alpha         = args.alpha,
        model_name    = args.model_name,
        dataset_name  = args.dataset_name,
        # 如果还需要 total_clients / selected_clients，请在这里赋值
        total_clients = args.num_clients,
        selected_clients = args.min_clients,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    trainloader, valloader, testloader = load_datasets(
        DATASET_NAME = config.dataset_name,
        CLIENT_NUMER = config.client_number,
        BATCH_SIZE = config.batch_size,
        PARTITION_ID = config.partition_id,
        FEDERATED=True,
        IID=config.IID,
        alpha=config.alpha,
        samples_per_client=None
    )

    # 加载模型
    model = load_model(config.model_name, config.dataset_name)
    model.to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # 创建客户端
    fl_client = SecureClient(
        config      = config,
        model       = model,
        trainloader = trainloader,
        testloader  = testloader,
        valloader   = valloader
    )

    addr = get_server_address()
    print(f"[Client {partition_id}] Will connect to server {addr} ...")
    fl.client.start_numpy_client(
        server_address=addr,
        client=fl_client,
        grpc_max_message_length=1024**3
    )
    
"""
Main Function
"""


def main():
    parser = argparse.ArgumentParser(description="Unified FL Script")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["server", "client", "all"],
                        help="选择启动模式。")
    parser.add_argument("--num-clients", type=int, default=5,
                        help="客户端数量(仅当 mode=all 时有效)。")
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="全局训练总轮数。")
    parser.add_argument("--min-clients", type=int, default=2,
                        help="每轮参与训练的最少客户端数。")
    parser.add_argument("--min-available-clients", type=int, default=5,
                        help="联邦中最少可用客户端数(用于策略中 fraction_fit 等)。")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="客户端学习率。")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="客户端训练 batch size。")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Non-IID Dirichlet alpha。仅当非IID时有用。")
    parser.add_argument("--iid", action="store_true", default=True,
                        help="若指定则使用 IID 数据分割，否则非IID。")
    parser.add_argument("--model-name", type=str, default="LeNet5",
                        help="模型名称。")
    parser.add_argument("--dataset-name", type=str, default="FASHIONMNIST",
                        help="数据集名称。")

    args = parser.parse_args()

    if args.mode == "server":
        # 仅启动服务器
        print("[Main] 启动服务器...")
        run_server(args)

    elif args.mode == "client":
        # 仅启动一个客户端，partition_id 这里示例直接用 0
        print("[Main] 启动单个客户端 partition_id=0 ...")
        run_client(args, partition_id=0)

    else:
        # mode == "all"
        print("[Main] 同时启动服务器 + 多个客户端...")
        from multiprocessing import Process

        # 1) 单独进程启动服务器
        server_proc = Process(target=run_server, args=(args,))
        server_proc.start()

        # 2) 等待服务器初始化
        time.sleep(3)

        # 3) 启动多个客户端
        client_processes = []
        for pid in range(args.num_clients):
            p = Process(target=run_client, args=(args, pid))
            p.start()
            client_processes.append(p)

        # 4) 等待所有客户端结束
        for cp in client_processes:
            cp.join()

        # 若你想在所有客户端结束后再结束服务器:
        server_proc.terminate()
        print("[Main] 所有客户端已经结束，服务器进程已终止。")

if __name__ == "__main__":
    main()