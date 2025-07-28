from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "Welcome to Fashion MNIST API."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "features" not in data:
        return {"error": "Missing 'features' key in request"}, 400

    features = np.array(data["features"]).reshape(1, -1)
    model = joblib.load("model.pkl")
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
