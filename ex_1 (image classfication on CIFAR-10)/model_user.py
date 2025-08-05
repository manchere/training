from PIL import Image
import torchvision.transforms as transforms
import torch
from model import CNN

transforms = transforms.Compose([
    transforms.Resize((32, 32)), # resize to match the CIFAR-10 input
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

img = Image.open("put_the_image_path_here") # put the image path
img = transforms(img.convert("RGB"))  # ensure image is in RGB forma
img = img.unsqueeze(0)  # add batch dimension

model = CNN()
model.load_state_dict(torch.load('cifar10_cnn.pth'))  # Load the trained model weights
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    print(f'predicted class: {predicted.item()}')


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f'predicted class: {classes[predicted.item()]}')