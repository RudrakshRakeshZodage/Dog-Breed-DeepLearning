from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from predict import predict_breed

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        try:
            prediction = predict_breed(img_path)
            if prediction.startswith('Error'):
                return jsonify({'error': prediction}), 500
            # Mock additional predictions for demo (replace with actual model output)
            predictions = [
                {'breed': f'{prediction}_sample1', 'confidence': '95', 'sample_image': ''},
                {'breed': f'{prediction}_sample2', 'confidence': '4', 'sample_image': ''},
                {'breed': f'{prediction}_sample3', 'confidence': '1', 'sample_image': ''},
            ]
            return jsonify({'predictions': predictions})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_sample_image', methods=['GET'])
def get_sample_image():
    # Placeholder for sample image (replace with actual sample image endpoint)
    sample_data = {
        'image_data': 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80',
        'predictions': [
            {'breed': 'Labrador_Retriever', 'confidence': '95', 'sample_image': ''},
            {'breed': 'Golden_Retriever', 'confidence': '4', 'sample_image': ''},
            {'breed': 'German_Shepherd', 'confidence': '1', 'sample_image': ''},
        ]
    }
    return jsonify(sample_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)