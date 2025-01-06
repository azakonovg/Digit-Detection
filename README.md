# Digit Detection

An educational web application for handwritten digit recognition using a PyTorch neural network trained on the MNIST dataset. The application helps users understand neural networks through interactive visualization and experimentation.

## Project Goals

- Create an interactive web app for digit recognition
- Provide training mode to learn from MNIST dataset
- Visualize neural network architecture with weights and connections
- Allow users to modify network layers and neurons
- Serve as an educational tool for understanding neural networks

## Features

- Interactive canvas for drawing digits
- Real-time digit prediction using neural network
- Model training with progress visualization
- Sample MNIST dataset images display
- Model weight reset functionality
- Network architecture visualization with weight connections
- Customizable hidden layers and neurons
- Detailed logging of dataset download and training process

## Technical Details

- Neural Network Architecture:
  - Fully connected neural network with customizable layers
  - ReLU activation for hidden layers
  - Softmax activation for output layer
  - Trained on MNIST dataset
  - Real-time weight visualization

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python3 run.py
```

3. Open your browser and navigate to `http://127.0.0.1:5001`

## Usage

1. Draw digits on the canvas or view MNIST samples
2. Train the model with custom epochs
3. Watch how weights change during training
4. Modify network architecture
5. Test the model with your own drawings

## Model Training

- The model is trained on the MNIST dataset
- Training progress is displayed in real-time
- Model weights are automatically saved after each epoch
- Training can be customized with different numbers of epochs 