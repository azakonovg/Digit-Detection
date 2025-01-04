# Digit Detection

A web application for handwritten digit recognition using a PyTorch neural network trained on the MNIST dataset.

## Features

- Interactive canvas for drawing digits
- Real-time digit prediction using a convolutional neural network
- Model training with progress visualization
- Sample MNIST dataset images display
- Model weight reset functionality
- Detailed logging of dataset download and training process

## Technical Details

- Neural Network Architecture:
  - Convolutional Neural Network (CNN) with 2 convolutional layers
  - Dropout layers for regularization
  - Fully connected layers for classification
  - Trained on MNIST dataset

## Setup

1. Install dependencies:
```bash
pip install torch torchvision flask pillow numpy requests
```

2. Run the application:
```bash
python3 run.py
```

3. Open your browser and navigate to `http://127.0.0.1:5001`

## Usage

1. Draw a digit (0-9) on the canvas using your mouse or touch input
2. Click "Submit" to get the model's prediction
3. Use "Clear" to reset the canvas
4. Train the model with custom epochs using the "Train Model" button
5. Reset model weights using the "Reset Model" button
6. View sample images from the MNIST training dataset

## Model Training

- The model is trained on the MNIST dataset
- Training progress is displayed in real-time
- Model weights are automatically saved after each epoch
- Training can be customized with different numbers of epochs 