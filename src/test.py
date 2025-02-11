
import torch


def test(net, testloader, verbose=False):
    """Evaluate the network on the entire test set."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
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
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    
    loss /= len(testloader.dataset)
    accuracy = correct / total
    if verbose:
        print(f"Test loss {loss}, accuracy {accuracy}")
    return loss, accuracy