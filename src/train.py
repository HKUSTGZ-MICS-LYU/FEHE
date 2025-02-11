import torch
from torch.optim.lr_scheduler import StepLR


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
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
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
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


