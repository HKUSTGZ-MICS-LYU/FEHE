import torch
import math

def train(
    net, 
    trainloader, 
    epochs: int, 
    config: dict, 
    current_round: int,
    total_rounds: int,
    verbose=False
):
    """Train the network on the training set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练前添加的诊断代码
    if device.type == 'cuda':
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
        print(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print(f"GPU已用内存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1e9:.2f} GB")
        torch.cuda.empty_cache()
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # 初始化优化器
    if config.get('optimizer') == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                   lr=config.get("lr"),
                                   weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(net.parameters(), 
                                  lr=config.get("lr"), 
                                  momentum=0.9, 
                                  weight_decay=5e-4)
    
    # 根据通信轮数计算当前学习率
    if config.get('scheduler') == 'cosine':
        # 余弦退火调度器
        eta_min = 1e-6  # 最小学习率
        current_lr = eta_min + (config.get("lr") - eta_min) * (1 + math.cos(math.pi * current_round / total_rounds)) / 2
    elif config.get('scheduler') == 'step':
        # 步进调度器
        step_size = 20
        gamma = 0.1
        current_lr = config.get("lr") * (gamma ** (current_round // step_size))
    else:
        raise ValueError("Invalid scheduler type")
    
    # 设置优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    net.train()
    
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            if isinstance(batch, dict):
                image_keys = ["img", "image"]
                label_keys = ["label", "fine_label", "coarse_label"]
                
                image_key = next((k for k in image_keys if k in batch), None)
                label_key = next((k for k in label_keys if k in batch), None)
                
                if image_key and label_key:
                    images, labels = batch[image_key], batch[label_key]
                else:
                    print(f"Missing required keys in batch: {batch.keys()}")
                    continue
            else:
                try:
                    images, labels = batch
                except:
                    print(f"Unexpected batch format: {type(batch)}")
                    continue
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            loss.backward()              
            optimizer.step()
      
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Train loss {epoch_loss}, accuracy {epoch_acc}")
    return net