from flask import Flask, render_template, request, jsonify, Response
import os
import base64
import re
import traceback
from model import classifier
import torch
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import json
from torchvision import datasets
import matplotlib.pyplot as plt
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def save_base64_image(base64_string):
    try:
        # Extract the base64 data from the data URL
        image_data = re.sub('^data:image/.+;base64,', '', base64_string)
        
        # Generate a filename
        filename = f'drawing_{len(os.listdir(app.config["UPLOAD_FOLDER"]))}.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Decode and save the image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        return f'uploads/{filename}'
    except Exception as e:
        print(f"Error saving base64 image: {str(e)}")
        print(traceback.format_exc())
        raise

def process_image_for_prediction(image_path):
    # Load and preprocess the image
    image = Image.open(os.path.join('static', image_path)).convert('L')  # Convert to grayscale
    
    # Invert colors for drawn images (since MNIST has white digits on black background)
    image = Image.fromarray(255 - np.array(image))
    
    # Resize and add padding to maintain aspect ratio
    target_size = 28
    width, height = image.size
    ratio = float(target_size) / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a black background
    new_image = Image.new('L', (target_size, target_size), 0)
    
    # Paste the resized image in the center
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and std
    ])
    
    image_tensor = transform(new_image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def get_sample_mnist_images(num_samples=10):
    try:
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        
        # Get random indices
        indices = torch.randperm(len(train_dataset))[:num_samples]
        
        sample_images = []
        for idx in indices:
            image, label = train_dataset[idx]
            # Convert tensor to PIL Image
            image = transforms.ToPILImage()(image)
            
            # Save image to memory
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Convert to base64
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            sample_images.append({
                'image': img_base64,
                'label': int(label)
            })
        
        return sample_images
    except Exception as e:
        print(f"Error getting MNIST samples: {str(e)}")
        return []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Handle canvas drawing submission
            if 'image_data' in request.form:
                image_data = request.form['image_data']
                image_path = save_base64_image(image_data)
                
                # Make prediction
                image_tensor = process_image_for_prediction(image_path)
                prediction_result = classifier.predict(image_tensor, get_probabilities=True)
                
                return jsonify({
                    'success': True, 
                    'image_path': image_path,
                    'prediction': prediction_result['prediction'],
                    'probabilities': prediction_result['probabilities']
                })
            
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('index.html')

@app.route('/train')
def train():
    try:
        epochs = int(request.args.get('epochs', 5))
        print("Starting training process...")
        
        def generate_training_progress():
            try:
                for progress in classifier.train_model(epochs=epochs):
                    progress_json = json.dumps(progress)
                    yield f"data: {progress_json}\n\n"
                
                # Send final completion message
                final_message = json.dumps({
                    'status': 'completed',
                    'message': 'Training completed successfully'
                })
                yield f"data: {final_message}\n\n"
            except Exception as e:
                error_message = json.dumps({
                    'status': 'error',
                    'message': f'Training error: {str(e)}'
                })
                yield f"data: {error_message}\n\n"
            finally:
                print("Training process finished")
        
        return Response(
            generate_training_progress(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        print(f"Error during training setup: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_mnist_samples')
def get_mnist_samples():
    try:
        num_samples = int(request.args.get('num_samples', 10))
        samples = get_sample_mnist_images(num_samples)
        return jsonify({'success': True, 'samples': samples})
    except Exception as e:
        print(f"Error in get_mnist_samples route: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_accuracy')
def get_accuracy():
    try:
        print("Calculating model accuracy...")
        evaluation = classifier.evaluate_model()
        if evaluation['success']:
            print(f"Accuracy calculation successful: {evaluation['accuracy']:.2f}%")
            return jsonify({'success': True, 'accuracy': evaluation['accuracy']})
        else:
            print(f"Accuracy calculation failed: {evaluation.get('error', 'Unknown error')}")
            return jsonify({'success': False, 'error': evaluation.get('error', 'Unknown error')}), 500
    except Exception as e:
        error_msg = f"Error getting accuracy: {str(e)}"
        print(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/reset_model', methods=['POST'])
def reset_model():
    try:
        success = classifier.reset_weights()
        if success:
            # Get accuracy after reset
            evaluation = classifier.evaluate_model()
            if evaluation['success']:
                return jsonify({
                    'success': True, 
                    'message': 'Model weights reset successfully',
                    'accuracy': evaluation['accuracy']
                })
            else:
                return jsonify({
                    'success': True, 
                    'message': 'Model weights reset successfully but could not evaluate accuracy'
                })
        else:
            return jsonify({'success': False, 'error': 'Failed to reset model weights'}), 500
    except Exception as e:
        print(f"Error resetting model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_dataset_info', methods=['GET'])
def get_dataset_info():
    try:
        info = classifier.get_dataset_info()
        if info['success']:
            return jsonify({'success': True, 'info': info['info']})
        else:
            return jsonify({'success': False, 'error': info.get('error', 'Unknown error')}), 500
    except Exception as e:
        print(f"Error getting dataset info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_network_info')
def get_network_info():
    """Get information about the neural network architecture."""
    return jsonify(classifier.get_network_info())

@app.route('/get_model_weights')
def get_model_weights():
    """Get model weights for visualization."""
    try:
        # Optionally limit the number of weights per layer
        limit = request.args.get('limit', default=100, type=int)
        return jsonify(classifier.get_model_weights(limit=limit))
    except Exception as e:
        print(f"Error in get_model_weights route: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/update_hidden_size', methods=['POST'])
def update_hidden_size():
    try:
        data = request.json
        if 'layers' not in data:
            return jsonify({'success': False, 'error': 'No layer configuration provided'}), 400
            
        layers = data['layers']
        # Validate layer configuration
        if not isinstance(layers, list) or not all(isinstance(size, int) and 1 <= size <= 1000 for size in layers):
            return jsonify({'success': False, 'error': 'Invalid layer configuration. Each layer must have 1-1000 neurons'}), 400
            
        result = classifier.update_architecture(layers)
        if result['success']:
            return jsonify({'success': True, 'message': f'Network architecture updated with layers: {layers}'})
        else:
            return jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500
    except Exception as e:
        print(f"Error updating network architecture: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/update_weight', methods=['POST'])
def update_weight():
    """Update a specific weight in the model."""
    try:
        data = request.json
        print(f"Received weight update request: {data}")
        
        # Validate required parameters
        required_params = ['layer_index', 'weight_indices']
        for param in required_params:
            if param not in data:
                return jsonify({'success': False, 'error': f'Missing required parameter: {param}'}), 400
        
        # Extract parameters
        layer_index = int(data['layer_index'])
        
        # Ensure weight_indices is properly formatted
        if not isinstance(data['weight_indices'], list) or len(data['weight_indices']) != 2:
            return jsonify({'success': False, 'error': 'weight_indices must be a list of two integers'}), 400
            
        weight_indices = [int(idx) for idx in data['weight_indices']]
        
        # Check if we are directly setting a value or using an increment
        if 'new_value' in data:
            new_value = float(data['new_value'])
            print(f"Setting weight at layer {layer_index}, indices {weight_indices} to {new_value}")
            result = classifier.update_weight(layer_index, weight_indices, new_value=new_value)
        elif 'increment' in data:
            increment = float(data['increment'])
            print(f"Incrementing weight at layer {layer_index}, indices {weight_indices} by {increment}")
            result = classifier.update_weight(layer_index, weight_indices, increment=increment)
        else:
            return jsonify({'success': False, 'error': 'Either new_value or increment must be provided'}), 400
        
        print(f"Update result: {result}")
        
        if result['success']:
            return jsonify({
                'success': True, 
                'new_weight': result['new_weight'],
                'accuracy': result['accuracy']
            })
        else:
            return jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500
            
    except Exception as e:
        error_msg = f"Error updating weight: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())  # Print full traceback
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/run_training_steps', methods=['POST'])
def run_training_steps():
    try:
        data = request.get_json()
        steps = data.get('steps', 10)
        
        # Validate input
        try:
            steps = int(steps)
            if steps < 1 or steps > 100:
                return jsonify({'success': False, 'error': 'Steps must be between 1 and 100'}), 400
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid steps value'}), 400
        
        # Run the training steps
        print(f"Running {steps} training steps...")
        result = classifier.train_steps(steps=steps)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        import traceback
        error_msg = f"Error during training steps: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        epochs = data.get('epochs', 1)
        batch_size = data.get('batch_size', 64)
        
        # Start training in a separate thread
        def train_thread():
            try:
                print(f"Starting training with {epochs} epochs...")
                for progress in classifier.train_model(epochs=epochs, batch_size=batch_size):
                    pass  # Progress is already logged in the console
                print("Training completed")
            except Exception as e:
                print(f"Error in training thread: {str(e)}")
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Training started'})
    except Exception as e:
        error_msg = f"Error starting training: {str(e)}"
        print(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
