import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import CNN


# Convert python image library images to PyTorch tensors

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def _set_trainset(data_path):
    return torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

def set_trainloader(data_path, batch_size=4):
    train_set = _set_trainset(data_path)
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    return trainloader

def _set_testset(data_path):
    return torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

def set_testloader(data_path, batch_size=4):
    test_set = _set_testset(data_path)
    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )
    return testloader


if __name__ == "__main__":
    data_path = './data'

    trainloader = set_trainloader(data_path, batch_size=4)
    print("Training set loaded successfully.")
    testloader = set_testloader(data_path, batch_size=4)
    print("Test set loaded successfully.")
    
    model = CNN()
    criterion = nn.CrossEntropyLoss() ## Loss function for multi-class classification
    optimizer =  torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) ## Stochastic Gradient Descent optimizer


    train_model = model.train_model(model, trainloader, criterion, optimizer, epochs=10)
    print("Training completed.")

    test_model = model.test_model(model, testloader, criterion)
    print("Testing completed.")

    #sav ethe trained mo
    torch.save(model.state_dict(), 'cifar10_cnn.pth')




