import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.bfv_rns import BFVContext
from src.Quantization import dequantize_weights, quantize_weights
import seaborn as sns

# 定义一个简单的全连接网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(5 * 5, 16)  # 第一层
        self.fc2 = nn.Linear(16, 10)      # 第二层

    def forward(self, x):
        x = x.view(-1, 5 * 5)  # 将输入展平为一维
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据加载到指定设备
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# 定义测试函数
def test(model, device, test_loader, criterion):
    model.eval()  # 进入评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 计算损失
            pred = output.argmax(dim=1, keepdim=True)  # 找到最大值的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # 计算平均损失
    accuracy = 100. * correct / len(test_loader.dataset)  # 计算准确率
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)\n")
    return accuracy

def get_params(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    return params



def set_params(model, params):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params:
                # If it's already a PyTorch tensor, just copy it.
                if isinstance(params[name], torch.Tensor):
                    param.copy_(params[name])
                else:
                    # Otherwise, convert to a PyTorch tensor on the correct device & dtype
                    new_tensor = torch.as_tensor(params[name], dtype=param.dtype, device=param.device)
                    param.copy_(new_tensor)
    return model



def create_context():
    global context
    n = 4096
    t = [65537]
    q = [30, 30, 30]
    context = BFVContext(q, n, t)
    # save context with pickle
    with open("context.pkl", "wb") as f:
        pickle.dump(context, f)
    f.close()
    

def param_encrypt(param_list):
    global context
    if not os.path.exists("context.pkl"):
        create_context()
    else:
        with open("context.pkl", "rb") as f:
            context = pickle.load(f)
        f.close()
        
        
    print("n:",context.n)
    print("t:", context.t[0])
    print("delta:", context.q // context.t[0])
    print("qi:",context.q_i)
    
    
    # Flatten the parameters and split them into chunks
    # Each chunk has n elements (n = 4096)
    flattened_params = []
    for param_name, param_tensor in param_list.items():
        flat_tensor = param_tensor.flatten()
        for val in flat_tensor:
            flattened_params.append(val.item())
            
    chunk_size = 4096
    global num_chunks

    num_chunks = (len(flattened_params) + chunk_size - 1) // chunk_size
    print("num_chunks:", num_chunks)
    chunked_params = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(flattened_params))
        chunked_params.append(flattened_params[start_idx:end_idx])
        
    # check if the last chunk is not full, fill it with zeros to chunk_size
    if len(chunked_params[-1]) < chunk_size:
        chunked_params[-1].extend([0] * (chunk_size - len(chunked_params[-1])))
        
    # Encrypt the data
    encrypted_params = []
    global encoded_time 
    encoded_time = 0
    global encrypted_time 
    encrypted_time = 0
    for chunk in chunked_params:
        
        encoded_time_temp = time.time()
        encoded_param = context.crt_and_encode(chunk)
        encoded_time_temp = time.time() - encoded_time_temp
        encoded_time += encoded_time_temp

        encrypted_time_temp = time.time()
        encrypted_param = context.encrypt(encoded_param)
        encrypted_time_temp = time.time() - encrypted_time_temp
        encrypted_time += encrypted_time_temp

        encrypted_params.append(encrypted_param)
        
    return encrypted_params

def param_decrypt(encrypted_params, params):
    # Decrypt the data and reconstruct the parameters
    # Reconstructed parameters are in the form of a dictionary like params
    global context
    decrypted_params = []
    global decoded_time
    decoded_time = 0
    global decrypted_time
    decrypted_time = 0
    for i, encrypted_param in enumerate(encrypted_params):
        decrypted_time_temp = time.time()
        decrypted_param = context.decrypt(encrypted_param)
        decrypted_time_temp = time.time() - decrypted_time_temp
        decrypted_time += decrypted_time_temp
        
        decoded_time_temp = time.time()
        decoded_param = context.decode_and_reconstruct(decrypted_param)
        decoded_time_temp = time.time() - decoded_time_temp
        decoded_time += decoded_time_temp
        
        decrypted_params.extend(decoded_param)
        
    # Reshape the decoded param to the original shape
    for name, param in params.items():
        params[name] = torch.tensor(decrypted_params[:param.size]).reshape(param.shape)
        decrypted_params = decrypted_params[param.size:]
        
    return params
    
    
        

# 主函数
def main():
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((5 , 5)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


    # 初始化模型、损失函数和优化器
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Train the model
    # print("0) Train and Test the model")
    # for epoch in range(10):
    #     train(model, device, train_loader, optimizer, criterion, epoch)
    # torch.save(model.state_dict(), "simple_net.pth")
    # test(model, device, test_loader, criterion)
    
    print("1) Load the model")
    model.load_state_dict(torch.load("simple_net.pth", weights_only=True))
    
    # Quantize the weights
    print("2) Quantize the weights")
    original_params = get_params(model)
    quantized_params, scales = quantize_weights(original_params, 8)

    # Encrypted the weights
    print("3) Encrypt the weights")
    encrypted_params = param_encrypt(quantized_params)
    
    # # Decrypt the weights
    print("4) Decrypt the weights")
    decrypted_params = param_decrypt(encrypted_params, quantized_params)
    
    # Dequantize the weights
    print("5) Dequantize the weights")
    dequantized_params = dequantize_weights(decrypted_params, scales)
    
    
    # Set the dequantized weights to the model
    print("6) Set the dequantized weights to the model")
    model = set_params(model, dequantized_params)
    
    # Test the model
    print("7) Test the model")
    test(model, device, test_loader, criterion)
    
    # Plot encrypt time and encoding time
    print("8) Plot encode time, encrypt time, decode time and decrypt time")
    print("encoded_time:", encoded_time)
    print("encrypted_time:", encrypted_time)
    print("decoded_time:", decoded_time)
    print("decrypted_time:", decrypted_time)
    
    # Plot the time
    import matplotlib.pyplot as plt


    # 计算每个操作的平均时间
    times = [encoded_time/num_chunks, encrypted_time/num_chunks, decoded_time/num_chunks, decrypted_time/num_chunks]
    labels = ["Encoded Time", "Encrypted Time", "Decoded Time", "Decrypted Time"]

    # 创建图形
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=sns.color_palette("viridis", 4))

    # 在每个柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加标题和标签
    plt.title("Time Breakdown for Encoding, Encrypting, Decoding, and Decrypting", fontsize=16)
    plt.xlabel("Operation", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)

    # 保存图片
    plt.savefig("time_breakdown.png", bbox_inches='tight')
    # plt.show()


        

if __name__ == "__main__":
    main()