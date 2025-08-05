import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1) # 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1) # 6 input channels, 16 output channels, 5x5 kernel
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # 16 input channels, 32 output channels, 5x5 kernel
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1) # 32 input channels, 64 output channels, 3x3 kernel
        
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
        
        self.fc1 = nn.Linear(64 * 2 * 2, 120) # 64 channels, 2x2 feature map size after conv4 and pooling
        self.fc2 = nn.Linear(120, 84) # 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 10)   # 84 input features, 10 output features (CIFAR-10 has 10 classes)
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 2 * 2) # Update to match output shape after 4 conv+pool layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    def train_model(self, model, trainLoader, criterion, optimizer, epochs=15):
        model.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0 # Initialize running loss
            correct = 0 
            total = 0

            for i, (images, labels) in enumerate(trainLoader):
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                
                #backward and optimize
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                #statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 100 == 99:   
                    batch_accuracy = 100 * correct / total
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainLoader)}], Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%')
                    running_loss = 0.0
        print(f'Epoch [{epoch + 1}/{epochs}] completed.')  # Add this line


    def test_model(self, model, testLoader, criterion):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testLoader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() # Count correct predictions

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(testLoader)
        print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}')
        return accuracy, avg_loss


