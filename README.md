# Digit Detection Web Application

A web application for handwritten digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- **Neural Network Model**:
  - Custom CNN architecture using PyTorch
  - Two convolutional layers with dropout for regularization
  - Real-time training with progress visualization
  - Model persistence (saves trained weights)
  - ~99% accuracy on MNIST dataset

- **Interactive Drawing**:
  - Canvas for drawing digits
  - Real-time digit recognition
  - Clear canvas functionality

- **Image Upload**:
  - Support for PNG, JPG, JPEG formats
  - Automatic preprocessing for prediction
  - Maintains aspect ratio during resizing

- **Training Interface**:
  - Configurable number of epochs
  - Live training progress updates
  - Loss and accuracy visualization
  - Automatic model saving

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/azakonovg/Digit-Detection.git
   cd Digit-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python3 run.py
   ```

4. Access the application at `http://127.0.0.1:5001`

## Model Architecture

The neural network consists of:
- Input layer (28x28 grayscale images)
- Conv2D layer (32 filters, 3x3 kernel)
- MaxPool2D layer
- Conv2D layer (64 filters, 3x3 kernel)
- MaxPool2D layer
- Dropout layers for regularization
- Fully connected layers (128 units)
- Output layer (10 classes)

## Usage

1. **Draw or Upload**:
   - Draw a digit on the canvas using your mouse/touch
   - Or upload an image file containing a digit

2. **Train the Model**:
   - Click the "Train Model" button
   - Set the number of epochs
   - Monitor training progress in real-time

3. **Get Predictions**:
   - Submit your drawn/uploaded digit
   - View the model's prediction

## Contributing

Feel free to submit issues and enhancement requests!

## License
MIT License 