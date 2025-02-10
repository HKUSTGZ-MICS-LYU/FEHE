
import torch


def test(net, testloader, verbose=True):
    """Evaluate the network on the entire test set."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            if "img" in batch: 
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            elif "image" in batch: 
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            else:
                images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
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