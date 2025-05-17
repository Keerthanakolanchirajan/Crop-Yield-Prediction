from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load trained model
model_path = "/content/drive/MyDrive/crop_yield_model.pkl"
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file '{model_path}' not found.")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON request
        data = request.get_json()

        # Convert inputs to a DataFrame for model compatibility
        input_df = pd.DataFrame({
            'Crop': [data['Crop']],
            'Crop_Year': [data['Crop_Year']],
            'Season': [data['Season']],
            'State': [data['State']],
            'Area': [data['Area']],
            'Production': [data['Production']],
            'Annual_Rainfall': [data['Annual_Rainfall']],
            'Fertilizer': [data['Fertilizer']],
            'Pesticide': [data['Pesticide']]
        })

        # Ensure data is correctly formatted before prediction
        input_df.fillna(0, inplace=True)

        # Perform prediction
        predicted_yield = model.predict(input_df)

        # Return JSON response
        return jsonify({'predicted_yield': round(predicted_yield[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
