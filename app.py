# app.py

from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('loan_approval_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()

        # Assuming the input_data is a dictionary with the same features as your training data
        features = pd.DataFrame(input_data, index=[0])

        # Make a prediction using the loaded model
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
