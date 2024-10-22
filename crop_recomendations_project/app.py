from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Define the directory where the model is stored
model_directory = os.path.join(os.getcwd(), "model")
model_file = "crop_recommendation_model.pkl"

# Build the full path for the model file
model_path = os.path.join(model_directory, model_file)

# Log the model loading path to the console
print(f"Looking for model at: {model_path}")

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)  # Load your trained model from the file
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model is located correctly.")

# Feature renaming function
def rename_features(features):
    rename_mapping = {
        'temperature': 'Temperature',  # Ensure that 'temperature' is renamed to 'Temperature'
        'ph': 'pH_Value',
        'nitrogen': 'Nitrogen',
        'humidity': 'Humidity',
        'phosphorus': 'Phosphorus',
        'potassium': 'Potassium',
        'rainfall': 'Rainfall',
        # Add any other features that need renaming
    }
    
    # Rename the columns using the mapping
    features = features.rename(columns=rename_mapping)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = request.form.to_dict()
        
        # Debugging: Print the incoming data to check its format
        print("Received input data:")
        print(input_data)

        # Convert the input data to a DataFrame
        features = pd.DataFrame([input_data])

        # Debugging: Print the DataFrame after conversion
        print("Converted DataFrame:")
        print(features)

        # Ensure the feature names are correct before making predictions
        features = rename_features(features)

        # Debugging: Print the DataFrame after renaming the features
        print("Renamed DataFrame:")
        print(features)

        # Make the prediction
        prediction = model.predict(features)
        
        # Convert numerical prediction back to the corresponding crop label
        crop_mapping = {
            1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya', 
            7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes', 
            12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram',
            17: 'MungBean', 18: 'MothBeans', 19: 'PigeonPeas', 20: 'KidneyBeans', 
            21: 'ChickPea', 22: 'Coffee'
        }

        predicted_crop = crop_mapping[prediction[0]]  # Map the numeric result to the crop name
        
        # Return the prediction result
        return render_template('result.html', prediction=predicted_crop)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({
            "error": "An error occurred during prediction.",
            "details": str(e)  # Provide more error details for debugging
        })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
