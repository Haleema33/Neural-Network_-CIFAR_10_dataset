# Training Neural Networks - CIFAR-10 Classification

This project demonstrates the process of training a fully-connected neural network to classify images from the CIFAR-10 dataset using PyTorch. The notebook guides users step-by-step from data loading and preprocessing to defining, training, and evaluating a neural network model.

## Features

- Loads and preprocesses the CIFAR-10 dataset using torchvision.
- Implements a fully-connected neural network with three layers.
- Supports GPU acceleration (CUDA).
- Training and validation loops with cross-entropy loss and Adam optimizer.
- Tracks training and validation accuracy and loss across epochs.
- Plots training and validation loss curves for performance visualization.

## Project Structure

- `SOLUTION_Training Neural Networks.ipynb`: Main Jupyter notebook containing code, explanations, and results.

## Model Architecture

The neural network model (`Net` class) is defined as follows:

- **Input**: Each CIFAR-10 image (3 × 32 × 32) is flattened into a single vector of size 3072.
- **Layer 1**: Linear (3072 → 120), ReLU activation.
- **Layer 2**: Linear (120 → 84), ReLU activation.
- **Layer 3**: Linear (84 → 10), output for the 10 CIFAR-10 classes.

This architecture enables the network to learn a mapping from image pixels to classification categories.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- Jupyter Notebook

## Training and Results

- The notebook trains the model for 10 epochs, reporting training and validation loss and accuracy after each epoch.
- Typical training accuracy can reach above 60%, with validation accuracy around 50–55% (results may vary depending on parameters).
- Loss curves are plotted to visualize learning progress and possible overfitting.

## Usage

1. Clone the repository and ensure dependencies are installed.
2. Open the notebook in Jupyter and run all cells.
3. Follow the explanations in the notebook to understand each step of the workflow.

## Potential Improvements

- Experiment with deeper or convolutional architectures for improved accuracy.
- Add data augmentation (e.g., random crop, flip, rotation) for better generalization.
- Tune hyperparameters such as batch size, learning rate, and number of layers.
- Implement regularization techniques (dropout, weight decay).

## Author

Developed as a practice project for training and evaluating neural networks in PyTorch for image classification tasks.
