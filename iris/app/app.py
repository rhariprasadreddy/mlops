from flask import Flask, request, jsonify
import pandas as pd
import mlflow

# Load model from MLflow registry
try:
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    model = mlflow.pyfunc.load_model(
        "models:/IrisBestModel/Production"
    )
except Exception as e:
    print(
        "Failed to load model: {}".format(str(e))
    )
    model = None

# Class label mapping
class_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# Initialize Flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "Iris Predictor API is running! Use POST /predict to get results."


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        input_json = request.get_json()
        input_df = pd.DataFrame([input_json])

        # Ensure column order
        input_df = input_df[
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ]
        ]

        prediction = model.predict(input_df)
        class_id = int(prediction[0])
        class_name = class_map[class_id]

        return jsonify({
            "predicted_class_id": class_id,
            "predicted_class_name": class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8001)
