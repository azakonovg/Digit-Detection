from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import base64
import re
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Handle canvas drawing submission
            if 'image_data' in request.form:
                image_data = request.form['image_data']
                image_path = save_base64_image(image_data)
                return jsonify({'success': True, 'image_path': image_path})
            
            # Handle file upload
            elif 'image' in request.files:
                file = request.files['image']
                if file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    return render_template('index.html', uploaded_image=f'uploads/{filename}')
                else:
                    return jsonify({'success': False, 'error': 'Invalid file type or no file selected'})
            
            return jsonify({'success': False, 'error': 'No image data or file provided'})
        
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('index.html', uploaded_image=None)

if __name__ == '__main__':
    app.run(debug=True)
