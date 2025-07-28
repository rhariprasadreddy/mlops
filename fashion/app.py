import os
import base64
from io import BytesIO

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template
import mlflow.pytorch

app = Flask(__name__)

# Set MLflow tracking URI (adjust if using remote MLflow server)
mlflow.set_tracking_uri("http://localhost:5000")

# Load model from MLflow registry
model_name = "FashionMNIST-BestModel"
stage = "Production"
client = mlflow.tracking.MlflowClient()
model_uri = f"models:/{model_name}/{stage}"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Class labels
classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    image = Image.open(file.stream).convert("RGB")
    image_transformed = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs.data, 1)
        label = classes[predicted.item()]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template("result.html", label=label, image_data=img_str)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
