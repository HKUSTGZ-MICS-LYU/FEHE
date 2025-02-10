import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tenseal as ts
import numpy as np
from globals import MODEL_MAP

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*14*14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def create_context():

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=4096,  # 增大多项式阶数
        coeff_mod_bit_sizes=[40, 21, 21, 21]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


class FLClient:
    def __init__(self, data, targets, test_data, test_targets, context):
        self.model = MODEL_MAP["LeNet5"]()  # 使用 LeNet5
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.data = data
        self.targets = targets
        self.test_data = test_data
        self.test_targets = test_targets
        self.context = context
        
    def evaluate(self):
        """评估模型在测试集上的准确率"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_data)
            _, predicted = torch.max(outputs.data, 1)
            total = self.test_targets.size(0)
            correct = (predicted == self.test_targets).sum().item()
        return correct / total

    # 修改本地训练逻辑
    def local_train(self, epochs=3):  # 增加epoch次数
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            # 保留本地梯度但不立即更新
    
    def get_encrypted_gradients(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad.detach().numpy().flatten()
                chunk_size = 4096
                for i in range(0, len(grad), chunk_size):
                    chunk = grad[i:i+chunk_size].tolist()
                    encrypted = ts.ckks_vector(self.context, chunk)
                    grads.append(encrypted)
        return grads
    
    def apply_aggregated_gradients(self, encrypted_aggregate):
        idx = 0
        for param in self.model.parameters():
            if param.grad is None: continue
            shape = param.grad.shape
            numel = param.grad.numel()
            
            # 使用更高精度的解密
            decrypted = []
            while len(decrypted) < numel:
                chunk = encrypted_aggregate[idx].decrypt()
                decrypted.extend([np.round(v, decimals=6) for v in chunk])  # 保留小数
                
            grad_tensor = torch.tensor(decrypted[:numel], 
                                    dtype=torch.float32).view(shape)
            param.grad = grad_tensor.clone()
            
        self.optimizer.step()
        self.optimizer.zero_grad()  # 清空梯度
       

def aggregate_gradients(all_grads):
    """
    聚合所有客户端的加密梯度
    Args:
        all_grads: 所有客户端的加密梯度列表
    Returns:
        聚合后的加密梯度
    """
    num_clients = len(all_grads)
    if num_clients == 0:
        return None
        
    # 获取第一个客户端的梯度作为初始值
    aggregated = all_grads[0]
    
    # 累加其他客户端的梯度
    for i in range(1, num_clients):
        for j in range(len(aggregated)):
            aggregated[j] += all_grads[i][j]
    
    # 计算平均值
    for grad in aggregated:
        grad *= (1 / num_clients)
        
    return aggregated

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_set = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    indices = torch.randperm(len(train_set))
    train_data = train_set.data[indices][:256]
    train_targets = train_set.targets[indices][:256]

    train_data = train_set.data[:256].unsqueeze(1).float() / 255.0
    train_targets = train_set.targets[:256]
    test_data = test_set.data[:256].unsqueeze(1).float() / 255.0
    test_targets = test_set.targets[:256]

    context = create_context()

    num_clients = 20  # 设置客户端数量
    clients = []
    data_per_client = len(train_data) // num_clients

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client = FLClient(train_data[start_idx:end_idx], train_targets[start_idx:end_idx], test_data, test_targets, context)
        clients.append(client)

    communication_rounds = 50
    local_epochs = 1

    client_acc = [[] for _ in range(num_clients)]
    mean_acc = []

    
    for round in range(communication_rounds):
        print(f"\n=== Communication Round {round+1} ===")
        
        # 每个客户端计算本地梯度
        for client in clients:
            client.local_train(epochs=local_epochs)
        
        # 收集并聚合加密梯度
        all_grads = [client.get_encrypted_gradients() for client in clients]
        aggregated = aggregate_gradients(all_grads)
        
        # 应用聚合梯度更新本地模型
        for client in clients:
            client.apply_aggregated_gradients(aggregated)
        
        # 评估
        round_acc = []
        for i, client in enumerate(clients):
            acc = client.evaluate()
            print(f"Client{i+1} Accuracy: {acc*100:.2f}%")
            client_acc[i].append(acc)
            round_acc.append(acc)
        
        mean_acc.append(sum(round_acc) / num_clients)
    
    import matplotlib.pyplot as plt
    rounds_range = list(range(1, communication_rounds + 1))
    plt.figure()
    for i in range(num_clients):
        plt.plot(rounds_range, [a * 100 for a in client_acc[i]], label=f'Client{i+1} Accuracy')
    plt.plot(rounds_range, [a * 100 for a in mean_acc], label='Mean Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Communication Rounds')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
