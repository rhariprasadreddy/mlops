from flask import Flask, request, jsonify
import pandas as pd
import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load model from MLflow registry
model = mlflow.pyfunc.load_model("models:/HousingBestModel/Production")

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.get_json()
        input_df = pd.DataFrame([input_json])

        # Ensure feature order matches training
        input_df = input_df[
            [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
        ]

        prediction = model.predict(input_df)
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8000)
