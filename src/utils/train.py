import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


    

def train(net, trainloader, epochs: int, config: dict, verbose=False):
    """Train the network on the training set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    initial_lr = config.get("lr", 0.1)
    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=initial_lr,
        momentum=0.9, 
        weight_decay=5e-4
    )

    scheduler_type = config.get("scheduler_step", "cosine")
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr=config.get("lr"), 
                                momentum=0.9, 
                                weight_decay=5e-4)
    
    if config.get('scheduler') == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('total_rounds', epochs),
            eta_min=config.get("min_lr", 0.001)
        )
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    net.train()
    
    for _ in range(config['server_round']):
        scheduler.step()
        
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
            
        scheduler.step()
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Train loss {epoch_loss}, accuracy {epoch_acc}")


