import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import requests
import time
import urllib.request
import gzip
import numpy as np
from urllib.error import URLError
import struct
from PIL import Image
import ssl
import certifi
import sys

def log_debug(message):
    """Print debug message to stderr to ensure it's visible in the console"""
    print(f"\n[DEBUG] {message}", file=sys.stderr, flush=True)

class MNIST_Custom(datasets.VisionDataset):
    """Custom MNIST dataset implementation with direct download."""
    
    resources = [
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
         'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
         'd53e105ee54ea749a09fcbcd1e9432'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
         '9fb629c4189551a2d022fa330f9573f3'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
         'ec29112dd5afa0611ce80d1b7f02629c')
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    dataset_info = {
        'train_size': 60000
    }

    def __init__(self, root, train=True, transform=None, target_transform=None):
        log_debug("=== Initializing MNIST_Custom Dataset ===")
        log_debug(f"Root directory: {root}")
        log_debug(f"Train mode: {train}")
        log_debug(f"Training set size: {self.dataset_info['train_size']} images")
        
        super(MNIST_Custom, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        self.train = train
        
        log_debug("Checking if dataset exists...")
        if not self._check_exists():
            log_debug("Dataset not found. Starting download...")
            self.download()
        else:
            log_debug("Dataset already exists.")

        if self.train:
            data_file = self.training_file
            log_debug(f"Loading training data from {data_file}")
        else:
            data_file = self.test_file
            log_debug(f"Loading test data from {data_file}")

        try:
            log_debug(f"Loading data from {os.path.join(self.processed_folder, data_file)}")
            self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
            log_debug(f"Data loaded successfully. Shape: {self.data.shape}, Targets shape: {self.targets.shape}")
        except Exception as e:
            log_debug(f"Error loading data: {str(e)}")
            raise

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                          self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                          self.test_file)))

    def download(self):
        """Download and process the MNIST dataset"""
        log_debug("=== Starting MNIST Dataset Download ===")
        log_debug(f"Raw data folder: {self.raw_folder}")
        log_debug(f"Processed data folder: {self.processed_folder}")
        
        if self._check_exists():
            log_debug("Dataset already downloaded and processed.")
            log_debug("=== Download Process Complete ===")
            return

        log_debug("Creating necessary directories...")
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        log_debug("Directories created successfully.")

        # Download files
        total_files = len(self.resources)
        downloaded_files = 0
        log_debug(f"Downloading {total_files} files...")
        
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            
            if os.path.exists(file_path):
                log_debug(f"File already exists: {filename}")
                downloaded_files += 1
                continue
                
            log_debug(f"Downloading file {downloaded_files + 1}/{total_files}: {filename}")
            try:
                log_debug(f"Attempting primary download from: {url}")
                self._download_file(url, file_path)
                downloaded_files += 1
                log_debug(f"Successfully downloaded: {filename}")
            except Exception as e:
                log_debug(f"Primary download failed: {str(e)}")
                alt_url = url.replace('ossci-datasets.s3.amazonaws.com/mnist', 
                                    'yann.lecun.com/exdb/mnist')
                try:
                    log_debug(f"Attempting alternative download from: {alt_url}")
                    self._download_file(alt_url, file_path)
                    downloaded_files += 1
                    log_debug(f"Successfully downloaded {filename} from alternative source")
                except Exception as e2:
                    log_debug(f"Alternative download failed: {str(e2)}")
                    log_debug("=== Download Process Failed ===")
                    raise RuntimeError(
                        f"Failed to download {filename}. Tried both:\n"
                        f"1. {url}\n"
                        f"2. {alt_url}\n"
                        f"Please check your internet connection or manually download "
                        f"the file and place it in: {self.raw_folder}")

        log_debug(f"Downloaded {downloaded_files}/{total_files} files successfully.")

        # Process and save as torch files
        try:
            log_debug("Processing downloaded files...")
            log_debug("Reading training data...")
            training_set = (
                self._read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte.gz')),
                self._read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte.gz'))
            )
            log_debug("Training data processed successfully.")
            
            log_debug("Reading test data...")
            test_set = (
                self._read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte.gz')),
                self._read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte.gz'))
            )
            log_debug("Test data processed successfully.")
        except Exception as e:
            log_debug("=== Processing Failed ===")
            raise RuntimeError(f"Error processing downloaded files: {str(e)}")

        try:
            log_debug("Saving processed files...")
            log_debug(f"Saving training data to {self.training_file}")
            with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
                torch.save(training_set, f)
            
            log_debug(f"Saving test data to {self.test_file}")
            with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
                torch.save(test_set, f)
            
            log_debug("=== Download and Processing Complete ===")
            log_debug(f"Dataset is ready for use!")
        except Exception as e:
            log_debug("=== Saving Failed ===")
            raise RuntimeError(f"Error saving processed files: {str(e)}")

    def _download_file(self, url, filepath):
        """Download a file with proper SSL context and robust error handling"""
        log_debug(f"Initiating download from: {url}")
        log_debug(f"Target path: {filepath}")
        
        # Create SSL context with certificate verification
        log_debug("Setting up SSL context...")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        log_debug("SSL context created successfully")
        
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                log_debug(f"Download attempt {attempt + 1}/{max_retries}")
                # Try to download with requests first
                log_debug("Using requests library for download...")
                response = requests.get(url, verify=True, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                log_debug(f"Total file size: {total_size/1024/1024:.1f} MB")
                
                block_size = 8192
                downloaded = 0
                
                log_debug("Starting file download...")
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            downloaded += len(chunk)
                            f.write(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f'\rProgress: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent:.1f}%)', end='', file=sys.stderr, flush=True)
                
                log_debug(f"File downloaded successfully: {filepath}")
                return
            
            except (requests.exceptions.RequestException, IOError) as e:
                log_debug(f"Download attempt {attempt + 1} failed")
                log_debug(f"Error: {str(e)}")
                
                if attempt < max_retries - 1:
                    log_debug(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Last attempt: try urllib as fallback
                    log_debug("All requests attempts failed. Trying urllib as fallback...")
                    try:
                        log_debug("Setting up urllib opener...")
                        opener = urllib.request.build_opener()
                        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                        urllib.request.install_opener(opener)
                        
                        log_debug("Downloading with urllib...")
                        urllib.request.urlretrieve(
                            url,
                            filepath,
                            context=ssl_context
                        )
                        log_debug("Successfully downloaded file using fallback method")
                        return
                    except Exception as e2:
                        log_debug("Fallback download failed")
                        log_debug(f"Error: {str(e2)}")
                        raise RuntimeError(
                            f"Failed to download after {max_retries} attempts and fallback.\n"
                            f"Primary error: {str(e)}\n"
                            f"Fallback error: {str(e2)}")

    def _read_image_file(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            buf = f.read()
            data = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols))
            return data

    def _read_label_file(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            buf = f.read()
            labels = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
            return labels

class DigitNet(nn.Module):
    def __init__(self, hidden_sizes=[128]):
        super(DigitNet, self).__init__()
        
        # Create a list to store all layers
        self.layers = nn.ModuleList()
        
        # Input layer: 28x28=784 nodes (flattened image) -> first hidden layer
        prev_size = 28 * 28
        
        # Create hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer: last hidden layer -> 10 nodes (digits 0-9)
        self.layers.append(nn.Linear(prev_size, 10))
        
        # Store layer information
        self.layer_info = [
            {
                'name': 'Input Layer',
                'type': 'Flatten',
                'input_shape': '28x28',
                'output_shape': '784'
            }
        ]
        
        # Add hidden layers info
        total_params = 0
        for i, hidden_size in enumerate(hidden_sizes):
            layer = self.layers[i]
            # Calculate parameters:
            # Weights: input_size * output_size
            # Biases: output_size
            weights_params = layer.in_features * layer.out_features
            bias_params = layer.out_features
            params = weights_params + bias_params
            total_params += params
            self.layer_info.append({
                'name': f'Hidden Layer {i+1}',
                'type': 'Linear + ReLU',
                'input_shape': str(layer.in_features),
                'output_shape': str(layer.out_features),
                'parameters': params,
                'weights': weights_params,
                'biases': bias_params
            })
        
        # Add output layer info
        output_layer = self.layers[-1]
        # Calculate output layer parameters
        output_weights_params = output_layer.in_features * output_layer.out_features
        output_bias_params = output_layer.out_features
        output_params = output_weights_params + output_bias_params
        total_params += output_params
        self.layer_info.append({
            'name': 'Output Layer',
            'type': 'Linear + LogSoftmax',
            'input_shape': str(output_layer.in_features),
            'output_shape': '10',
            'parameters': output_params,
            'weights': output_weights_params,
            'biases': output_bias_params
        })
        
        # Store total parameters
        self.total_params = total_params
        
    def get_network_info(self):
        """Get information about the neural network architecture."""
        return {
            'layers': self.layer_info,
            'total_parameters': self.total_params
        }
        
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        
        # Pass through hidden layers with ReLU activation
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Output layer with log softmax
        x = F.log_softmax(self.layers[-1](x), dim=1)
        return x

class DigitClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.hidden_sizes = [128]  # Default hidden layer configuration
        self.model = DigitNet(hidden_sizes=self.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.model_path = 'static/model/digit_model.pth'
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Initialize model with random weights or load if exists
        if os.path.exists(self.model_path):
            try:
                # Try to load the model state and metadata
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Check if it's the old format (with fc1 and fc2)
                if isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
                    print("Converting old model format to new architecture...")
                    # Create a temporary model with single hidden layer
                    temp_model = DigitNet(hidden_sizes=[128]).to(self.device)
                    # Map old layer names to new ones
                    new_state_dict = {
                        'layers.0.weight': checkpoint['fc1.weight'],
                        'layers.0.bias': checkpoint['fc1.bias'],
                        'layers.1.weight': checkpoint['fc2.weight'],
                        'layers.1.bias': checkpoint['fc2.bias']
                    }
                    temp_model.load_state_dict(new_state_dict)
                    # Save in new format with metadata
                    self.save_model()
                    print("Converted and saved model in new format")
                else:
                    # Check if the checkpoint has metadata
                    if isinstance(checkpoint, dict) and 'hidden_sizes' in checkpoint:
                        self.hidden_sizes = checkpoint['hidden_sizes']
                        self.model = DigitNet(hidden_sizes=self.hidden_sizes).to(self.device)
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        print("No architecture metadata found, using default architecture")
                        # Save current model with metadata
                        self.save_model()
                
                print(f"Loaded model with architecture: {self.hidden_sizes}")
                self.optimizer = optim.Adam(self.model.parameters())
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Initializing with random weights")
        
        # Training data setup
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.dataset_info = MNIST_Custom.dataset_info

    def save_model(self):
        """Save model with architecture metadata"""
        try:
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'hidden_sizes': self.hidden_sizes
            }
            torch.save(checkpoint, self.model_path)
            print(f"Saved model with architecture: {self.hidden_sizes}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def update_architecture(self, hidden_sizes):
        """Update the model with new hidden layer sizes."""
        try:
            print(f"Updating model architecture with hidden sizes: {hidden_sizes}")
            # Create new model with the specified architecture
            new_model = DigitNet(hidden_sizes=hidden_sizes).to(self.device)
            
            # Initialize with random weights
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
            
            new_model.apply(init_weights)
            
            # Update model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters())
            self.hidden_sizes = hidden_sizes
            
            # Save the new model with metadata
            self.save_model()
            
            return {'success': True}
        except Exception as e:
            error_msg = f"Error updating model architecture: {str(e)}"
            print(error_msg)
            return {'success': False, 'error': error_msg}

    def get_network_info(self):
        """Get information about the neural network architecture."""
        return self.model.get_network_info()

    def evaluate_model(self):
        """Evaluate the model on the test dataset and return accuracy."""
        try:
            # Load test dataset
            test_dataset = MNIST_Custom(root='./data', train=False, transform=self.transform)
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
            
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = 100. * correct / total
            return {'success': True, 'accuracy': accuracy}
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return {'success': False, 'error': str(e)}

    def train_model(self, epochs=5, batch_size=64):
        try:
            # Load MNIST dataset using custom implementation
            print("Loading MNIST dataset...")
            train_dataset = MNIST_Custom(root='./data', train=True, transform=self.transform)
            
            # Create data loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            print(f"Starting training for {epochs} epochs...")
            self.model.train()
            training_progress = []
            last_save_time = time.time()
            
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    try:
                        # Move data to device and clear gradients
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        # Forward pass
                        output = self.model(data)
                        loss = F.nll_loss(output, target)
                        
                        # Backward pass
                        loss.backward()
                        self.optimizer.step()
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        running_loss += loss.item()
                        
                        # Report progress more frequently
                        if batch_idx % 50 == 0:
                            accuracy = 100. * correct / total
                            avg_loss = running_loss / (batch_idx + 1)
                            
                            progress = {
                                'epoch': epoch + 1,
                                'batch': batch_idx,
                                'loss': avg_loss,
                                'accuracy': accuracy,
                                'total_batches': len(train_loader)
                            }
                            training_progress.append(progress)
                            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                            yield progress
                            
                            # Save model periodically (every 5 minutes)
                            current_time = time.time()
                            if current_time - last_save_time > 300:  # 300 seconds = 5 minutes
                                try:
                                    self.save_model()
                                    print(f"Periodic save at epoch {epoch + 1}, batch {batch_idx}")
                                    last_save_time = current_time
                                except Exception as e:
                                    print(f"Warning: Could not save periodic checkpoint: {str(e)}")
                            
                            # Clear memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                        # Try to save model on error
                        try:
                            self.save_model()
                            print(f"Saved model after error in batch {batch_idx}")
                        except Exception as save_error:
                            print(f"Could not save model after error: {str(save_error)}")
                        continue
                
                # Save model after each epoch
                try:
                    self.save_model()
                    print(f"Saved model after epoch {epoch + 1}")
                except Exception as e:
                    print(f"Error saving model after epoch: {str(e)}")
            
            # Final save attempt and evaluation
            try:
                self.save_model()
                print("Training completed successfully")
                
                # Evaluate model after training
                evaluation = self.evaluate_model()
                if evaluation['success']:
                    print(f"Final model accuracy on test set: {evaluation['accuracy']:.2f}%")
                    return {'status': 'success', 'message': f"Training completed successfully. Test accuracy: {evaluation['accuracy']:.2f}%"}
                
                return {'status': 'success', 'message': 'Training completed successfully'}
            except Exception as e:
                print(f"Error saving final model: {str(e)}")
                return {'status': 'partial_success', 'message': 'Training completed but could not save final model'}
        
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def predict(self, image_tensor):
        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor.to(self.device))
                pred = output.argmax(dim=1, keepdim=True)
                return pred.item()
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    def reset_weights(self):
        """Reset the model weights to random values"""
        def weight_reset(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.reset_parameters()
        
        try:
            self.model.apply(weight_reset)
            # Delete the saved model file if it exists
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            print("Model weights reset to random values")
            
            # Evaluate model after reset
            evaluation = self.evaluate_model()
            if evaluation['success']:
                print(f"Model accuracy after reset: {evaluation['accuracy']:.2f}%")
            
            return True
        except Exception as e:
            print(f"Error resetting weights: {str(e)}")
            return False

    def get_dataset_info(self):
        """Return information about the MNIST dataset."""
        try:
            return {
                'success': True,
                'info': self.dataset_info
            }
        except Exception as e:
            print(f"Error getting dataset info: {str(e)}")
            return {'success': False, 'error': str(e)}

# Initialize the classifier
print("Initializing DigitClassifier...")
classifier = DigitClassifier() 