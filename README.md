

# Machine Condition Prediction using Random Forest

**Madhan Raj M**
**2nd Year, Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

## Project Overview

This project is focused on predicting the condition of a machine using a **Random Forest Classifier**. The prediction is based on various input parameters like temperature, vibration, oil quality, RPM, and other relevant sensor data. It is designed to help in early detection of machine faults and support preventive maintenance.

I worked on this project as part of my course on Data Analysis in Mechanical Engineering. It gave me a hands-on experience in applying machine learning to real-world mechanical systems.

---

## Tools and Libraries Required

Before running the code, make sure all the necessary Python libraries are installed. You can do this using the following command:

```bash
pip install -r requirements.txt
```

This will install all the dependencies listed in the `requirements.txt` file.

---

## Key Files Used for Prediction

The following files are essential for running the prediction script:

* `random_forest_model.pkl`: This file contains the trained Random Forest model.
* `scaler.pkl`: A scaler object that was used to normalize input features during training.
* `selected_features.pkl`: This file includes the exact list and order of features used to train the model.

These files should be in the same directory as your prediction script, or the correct file paths should be provided.

---

## How the Prediction Process Works

1. **Loading the Required Files**
   Load the trained model, feature scaler, and feature list using `joblib.load()`.

2. **Input Data Preparation**
   Create a pandas DataFrame with only one row containing your machine’s current values for all features.

3. **Data Preprocessing**
   The input values need to be scaled using the loaded `scaler` so they match the format used during model training.

4. **Making a Prediction**
   Use `.predict()` to determine the machine's condition.
   You can also use `.predict_proba()` to see the probabilities for each possible class.

---

## How to Run a Prediction

Here is a basic template for running a prediction:

```python
import joblib
import pandas as pd

# Load saved model and related files
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Sample input values
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange columns in the correct order
new_data = new_data[selected_features]

# Apply scaling
scaled_data = scaler.transform(new_data)

# Get predictions
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Machine Condition:", prediction[0])
print("Class Probabilities:", prediction_proba[0])
```

---

## Important Notes

* Always make sure your input includes the **exact same features** as the model was trained on.
* Values should be within the typical operating range.
* Keep the feature order the same as used during training to avoid incorrect results.

---

## Optional: Retraining the Model

If you want to retrain or update the model with new data:

* Follow the same preprocessing and scaling steps.
* Use consistent feature selection.
* Save the new model and preprocessing objects using `joblib`.

---

## Applications of This Project

* Identifying whether a machine is operating normally or if there are early signs of faults.
* Can be applied in manufacturing environments, predictive maintenance systems, or smart industrial setups using sensor data.
