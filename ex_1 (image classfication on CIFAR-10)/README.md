# CIFAR-10 Image Classification with CNN

This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify images from the CIFAR-10 dataset.

## Project Structure

```
├── loader.py      # Loads and preprocesses CIFAR-10 data
├── model.py       # Defines and trains/tests the CNN model
├── data/          # CIFAR-10 dataset download location
└── README.md      # Project info
```

## How to Use

1. Install requirements:
    - Python 3.x
    - PyTorch
    - TorchVision

2. Run the main script to train and test the model:
    ```bash
    python loader.py
    ```

This will:
- Download and load the CIFAR-10 dataset
- Train the CNN model for 10 epochs (default, configurable in `loader.py`)
- Test the model
- Save the trained model as `cifar10_cnn.pth`

## Notes

- Training and test progress is printed to the console.
- You can adjust batch size and epochs in `loader.py`.
- The model and data are saved locally for reuse.
  - FC3: 84→10 neurons (output layer for 10 classes)
- **Regularization**: Dropout (50%) and ReLU activation functions

### Training Features
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Optimizer**: SGD with momentum (lr=0.001, momentum=0.9)
- **Training Monitoring**: Real-time loss and accuracy tracking
- **Model Persistence**: Automatic model saving after training

## Configuration

- **Batch Size**: 4 (configurable in data loaders)
- **Epochs**: 10 (configurable in training call)
- **Learning Rate**: 0.001
- **Momentum**: 0.9

## Requirements

- PyTorch
- TorchVision
- Python 3.x

## Notes

- The model automatically downloads CIFAR-10 dataset on first run
- Training progress is displayed every 100 batches
- The normalization strategy helps with faster convergence and stable gradients
- Model state is saved for future inference without retraining
- **Learning Rate**: 0.001
- **Momentum**: 0.9

## Requirements

- PyTorch
- TorchVision
- Python 3.x

## Notes

- The model automatically downloads CIFAR-10 dataset on first run
- Training progress is displayed every 100 batches
- The normalization strategy helps with faster convergence and stable gradients
- Model state is saved for future inference without retraining
