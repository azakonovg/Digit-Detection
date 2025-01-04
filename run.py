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
                prediction = classifier.predict(image_tensor)
                
                return jsonify({
                    'success': True, 
                    'image_path': image_path,
                    'prediction': prediction
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
        evaluation = classifier.evaluate_model()
        if evaluation['success']:
            return jsonify({'success': True, 'accuracy': evaluation['accuracy']})
        else:
            return jsonify({'success': False, 'error': evaluation.get('error', 'Unknown error')}), 500
    except Exception as e:
        print(f"Error getting accuracy: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
