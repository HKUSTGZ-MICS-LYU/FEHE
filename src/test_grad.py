# use the function test and train to evaluate the model

import numpy
import torch
from test import test
from train import train, train_get_loss, train_update_param
from models import LeNet5



# use fashionmnist dataset
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

model = LeNet5()
trainset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)
testloader = DataLoader(testset, batch_size=64)
for i in range(100):
    optimizer = train_get_loss(model, trainloader, epochs=1, verbose=True)
    # Extract gradients after training
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.cpu().numpy())
        else:
            gradients.append(None)
    
    # print("Gradients: ", gradients)
    

    model.zero_grad()
    # for param in model.parameters():
    #     print(param.grad)
    
    
    # Put the gradients back into the model
    for param, gradient in zip(model.parameters(), gradients):
        if gradient is not None:
            param.grad = torch.tensor(gradient)
            
    # for param in model.parameters():
    #     print(param.grad)
            
    train_update_param(model, optimizer, trainloader, epochs=i, verbose=True)
            
            

    loss, accuracy = test(model, testloader)
    print(f"Test loss: {loss}, accuracy: {accuracy}")
