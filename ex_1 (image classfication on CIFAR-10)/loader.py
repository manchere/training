import torch
import torchvision
import torchvision.transforms as transforms


# Convert python image image library images to PyTorch tensors

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

# def pil_to_tensor(image):
#     return transforms.ToTensor()(image)

# Second step is used to normalize the images the data by specifying:
# mean and standard deviation for each channel (RGB).
# this  will convert the data from [0, 1] to [-1, 1] range.
# this helps with faster convergence during training, more stable gradients, and better performance.
# it will also make the network learm more efficiently and works better with the ReLU activation function.

# def pil_to_tensor_normalized(image):
#     return transform(image)

# def tensor_normalized(tensor):
#     normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     return normalize(tensor)

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
    _set_testset(data_path)
    return torch.utils.data.DataLoader(
        _set_testset,
        batch_size=batch_size,
        shuffle=False,
    )


if __name__ == "__main__":
    data_path = './data'
    trainloader = set_trainloader(data_path, batch_size=4)
    testloader = set_testloader(data_path, batch_size=4)

    # Example usage
    for images, labels in trainloader:
        print(images.shape, labels.shape)
        break


