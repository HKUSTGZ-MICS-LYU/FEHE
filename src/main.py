from create_server import create_server_fn
from create_client import create_client_fn
from flwr.simulation import run_simulation
import torch
import argparse
from globals import DATASET_MAP, MODEL_MAP


DEBUG = False
if DEBUG:
    import logging
    logging.basicConfig(level=logging.INFO)
    
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Flower Federated Learning Simulation")
    parser.add_argument("--model", type=str, choices=MODEL_MAP.keys(), default="Net", help="Choose the model to use")
    parser.add_argument("--dataset", type=str, choices=DATASET_MAP.keys(), default="cifar10", help="Choose the dataset to use")
    
    parser.add_argument("--min_fit_clients", type=int, default=1, help="Minimum number of clients required for training")
    parser.add_argument("--min_available_clients", type=int, default=1, help="Minimum number of clients required for training")
    parser.add_argument("--min_evaluate_clients", type=int, default=1, help="Minimum number of clients required for evaluation")
    
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of rounds for the simulation")
    parser.add_argument("--num_supernodes", type=int, default=1, help="Number of supernodes in the simulation")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs allocated per client")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs allocated per client")
    
    parser.add_argument("--server_cpus", type=int, default=1, help="Number of CPUs allocated to the server")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used for training")
    args = parser.parse_args()

    # 动态配置 GPU
    backend_config = {
        "client_resources": {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
        "server_resources": {"num_cpus": args.server_cpus},
    }
    if torch.cuda.is_available():
        backend_config["client_resources"]["num_gpus"] = 1

    # 打印配置信息
    print("Running simulation with the following configuration:")
    print(f" - Model: {args.model}, Dataset: {args.dataset}")
    print(f" - Number of supernodes: {args.num_supernodes}")
    print(f" - Number of rounds: {args.num_rounds}")
    print(f" - Minimum number of clients required for training: {args.min_fit_clients}, Minimum number of clients required for evaluation: {args.min_evaluate_clients}, Minimum number of clients required for training: {args.min_available_clients}")
    print(f" - Client resources: {backend_config['client_resources']}")
    print(f" - Server resources: {backend_config['server_resources']}")

    print("/******** Creating ServerApp ********/\n")
    server_app = create_server_fn(args.num_rounds, args.min_fit_clients, args.min_evaluate_clients, args.min_available_clients)
    
    print("/******** Creating ClientApp ********/\n")
    client_app = create_client_fn(args.batch_size, args.num_supernodes, args.model, args.dataset)

    print("/******** Running Simulation ********/\n")
    run_simulation(
        server_app=server_app,
        client_app=client_app,  
        num_supernodes=args.num_supernodes,
        backend_config=backend_config,
    )
    print("/******** Simulation Completed ********/\n")
    
    
if __name__ == "__main__":
    main()