# Digit Detection Web App

A web application that allows users to draw or upload handwritten digits for recognition. The application provides two main functionalities:
1. Drawing digits directly in the browser
2. Uploading image files containing digits

## Features
- Interactive canvas for drawing digits
- File upload support for images (PNG, JPG, JPEG)
- Real-time drawing preview
- Mobile-friendly with touch support
- Clean and modern UI

## Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd digit-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Requirements
- Python 3.x
- Flask
- Other dependencies listed in requirements.txt

## Project Structure
- `run.py`: Main Flask application
- `templates/`: HTML templates
- `static/uploads/`: Directory for uploaded images
- `requirements.txt`: Python dependencies

## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
MIT License 