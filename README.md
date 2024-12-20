# MNIST Model with 1x1 Convolution in PyTorch

This project implements a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch. The model is designed to achieve 95% accuracy with less than 25,000 parameters in just one epoch.

## Requirements

- Python 3.6+
- PyTorch
- torchvision

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-model.git
   cd mnist-model
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision
   ```

## Usage

To train the model, run the following command:
```bash
python app/models/mnist_model.py
```

The model will download the MNIST dataset if it is not already present and will train for one epoch. After training, it will evaluate the model on the test dataset and print the test accuracy.

## Model Architecture

The model consists of:
- A 1x1 convolutional layer followed by a max pooling layer.
- A 3x3 convolutional layer.
- A fully connected layer that outputs the class probabilities for the 10 digit classes.

## GitHub Actions

This repository can be integrated with GitHub Actions for continuous integration and testing. Ensure to set up a workflow to validate the model's performance and parameter count.

## License

This project is licensed under the MIT License.
