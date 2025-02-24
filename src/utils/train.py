import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def train(
        net, 
        trainloader, 
        epochs: int, 
        config: dict, 
        verbose=False
    ):
    """Train the network on the training set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()


    if config.get('optimizer') == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                   lr=config.get("lr"),
                                   weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(net.parameters(), 
                                  lr=config.get("lr"), 
                                  momentum=0.9, 
                                  weight_decay=5e-4)
    
    if config.get('scheduler') == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('total_rounds')
        )
    elif config.get('scheduler') == 'step':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    else:
        raise ValueError("Invalid scheduler type")
    
    net.train()
    
    for epoch in range(epochs):
        
        current_lr = optimizer.param_groups[0]['lr']
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}, Learning Rate: {current_lr:.6f}")
            
            
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
            
        # scheduler.step()
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Train loss {epoch_loss}, accuracy {epoch_acc}")
    

