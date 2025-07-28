from flask import Flask, request, jsonify
import pandas as pd
import mlflow

app = Flask(__name__)


try:
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    model = mlflow.pyfunc.load_model("models:/HousingBestModel/Production")
except Exception as e:
    print("Model load failed:", str(e))
    model = None


@app.route("/", methods=["GET"])
def home():
    return "Housing price prediction API is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
