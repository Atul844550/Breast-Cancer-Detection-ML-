import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model
model = tf.keras.models.load_model("breast_cancer_model.h5")

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Streamlit App
st.title("Breast Cancer Prediction App")
st.markdown("""
### **About the Prediction**
This app predicts whether a tumor is **Malignant (M)** or **Benign (B)** based on input features.

- **Malignant (M):** Cancerous tumor (needs medical attention)
- **Benign (B):** Non-cancerous tumor (less harmful)
""")

# Define feature names based on dataset
feature_labels = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
    "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]

# Collect user inputs
inputs = []
st.markdown("### **Enter Feature Values Below:**")
for label in feature_labels:
    value = st.number_input(f"{label}", min_value=0.0, max_value=2000.0, step=0.01)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    # Check if all inputs are zero (default case)
    if all(value == 0.0 for value in inputs):
        st.warning("Please enter valid feature values before making a prediction.")
    else:
        # Convert input to numpy array and reshape
        input_data = np.array(inputs).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Scale the input data

        # Make prediction
        prediction = model.predict(input_data)[0][0]
        diagnosis = "Malignant (M) - Cancerous" if prediction > 0.5 else "Benign (B) - Non-cancerous"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display result
        st.subheader("Prediction Result")
        st.write(f"**Diagnosis:** {diagnosis}")
        st.write(f"**Confidence Level:** {confidence:.2%}")
