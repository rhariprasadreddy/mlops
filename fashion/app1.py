from flask import Flask, request, jsonify, render_template_string
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import mlflow
import mlflow.pytorch

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the production model
model_name = "FashionMNIST-BestModel"
model_stage = "Production"
model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_stage}")
model.eval()

# Flask app setup
app = Flask(__name__)

# Transform matching training preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# FashionMNIST class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Simple HTML form
html_form = '''
<!doctype html>
<title>FashionMNIST Predictor</title>
<h2>Upload a FashionMNIST Image</h2>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=image>
  <input type=submit value=Predict>
</form>
{% if prediction %}
  <h3>Prediction: {{ prediction }}</h3>
  <h4>Confidence: {{ confidence }}</h4>
{% endif %}
'''

@app.route('/')
def index():
    return render_template_string(html_form)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template_string(html_form, prediction="No image uploaded")

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('L')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()

    return render_template_string(html_form, prediction=label, confidence=round(confidence, 4))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)