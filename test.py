import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tenseal as ts
import numpy as np

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
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

class FLClient:
    def __init__(self, data, targets, test_data, test_targets, context):
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.data = data
        self.targets = targets
        self.test_data = test_data
        self.test_targets = test_targets
        self.context = context
    
    def local_train(self, epochs=5):
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
    
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
    
    def get_encrypted_parameters(self):
        params = []
        for param in self.model.parameters():
            param_data = param.detach().numpy().flatten()
            chunk_size = 4096
            for i in range(0, len(param_data), chunk_size):
                chunk = param_data[i:i+chunk_size].tolist()
                encrypted = ts.ckks_vector(self.context, chunk)
                params.append(encrypted)
        return params
    
    def apply_aggregated_parameters(self, encrypted_aggregate):
        idx = 0
        for param in self.model.parameters():
            shape = param.shape
            numel = param.numel()
            decrypted = []
            while len(decrypted) < numel:
                chunk = encrypted_aggregate[idx].decrypt()
                decrypted.extend(chunk)
                idx += 1
            param_data = torch.tensor(decrypted[:numel]).view(shape)
            param.data.copy_(param_data)
            


# 服务器聚合参数
def aggregate_parameters(all_params):
    avg_params = []
    for param_chunks in zip(*all_params):
        avg = param_chunks[0].copy()
        for p in param_chunks[1:]:
            avg += p
        avg *= 1 / len(param_chunks)
        avg_params.append(avg)
    return avg_params

# 客户端加载聚合后的参数
def apply_aggregated_parameters(self, encrypted_aggregate):
    idx = 0
    for param in self.model.parameters():
        shape = param.shape
        numel = param.numel()
        decrypted = encrypted_aggregate[idx].decrypt()
        param_data = torch.tensor(decrypted[:numel]).view(shape)
        param.data.copy_(param_data)
        idx += 1
        
    def apply_aggregated_gradients(self, encrypted_aggregate):
        idx = 0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            shape = param.grad.shape
            numel = param.grad.numel()
            decrypted = []
            while len(decrypted) < numel:
                chunk = encrypted_aggregate[idx].decrypt()
                decrypted.extend(chunk)
                idx += 1
            grad_tensor = torch.tensor(decrypted[:numel]).view(shape)
            param.grad = grad_tensor.clone().detach()
        self.optimizer.step()
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_data)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == self.test_targets).sum().item()
            return correct / len(self.test_targets)

def aggregate_gradients(all_gradients):
    aggregated = []
    for grad_chunks in zip(*all_gradients):
        agg = grad_chunks[0].copy()
        for g in grad_chunks[1:]:
            agg += g
        agg *= 1/len(grad_chunks)
        aggregated.append(agg)
    return aggregated

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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
        
        for client in clients:
            client.local_train(epochs=local_epochs)
        
        all_grads = [client.get_encrypted_gradients() for client in clients]
        aggregated = aggregate_gradients(all_grads)
        
        for client in clients:
            client.apply_aggregated_gradients(aggregated)
        
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
