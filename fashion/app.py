from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "Fashion API is up."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]
        model = joblib.load("model.pkl")
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
