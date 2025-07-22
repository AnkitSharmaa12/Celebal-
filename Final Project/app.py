import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
from joblib import load
import os
import pandas as pd 


try:
    scaler = load("scaler.joblib")
    models = {
        "Random Forest": load("random_forest_model.joblib"),
        "SVM": load("svm_model.joblib"),
        # "Gradient Boosting": load("gb_model.joblib") # this is not used as after so many tries it
        # was unable to train even with gpu.
    }
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler file: {e}. Please ensure 'scaler.joblib', 'random_forest_model.joblib', and 'svm_model.joblib' are in the correct directory.")
    st.stop() 

# Class labels
label_dict = {
    0: "daisy",
    1: "dandelion",
    2: "rose",
    3: "sunflower",
    4: "tulip"
}

# --- APP CONTENT ---

st.title("🌼 Flower Species Recognition Dashboard")
st.write("Upload a flower image and select a model to classify the species.")

# Model selection
model_choice = st.selectbox("Select Classification Model", list(models.keys()))

# Image uploader
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    st.write("---") 
    st.subheader("Processing Image and Prediction")

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Flower Image", use_column_width=True)

    # Convert to bytes and then to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) # 1 for color image

    # Resize image for model input
    image_resized = cv2.resize(image, (128, 128))

    # Convert to grayscale and extract HOG features
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # Ensure HOG parameters match those used during model training
    features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)

    # Scale features
    features_scaled = scaler.transform([features])

    # Make prediction
    model = models[model_choice]
    prediction = model.predict(features_scaled)[0]

    # Display result
    st.success(f"🌺 Predicted Flower Species: **{label_dict[prediction].capitalize()}**")

    st.write("---") 
    st.subheader("Prediction Confidence")

    # Check if the model has predict_proba method
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get the confidence for the predicted class
        predicted_confidence = probabilities[prediction] * 100

        st.write(f"Confidence for **{label_dict[prediction].capitalize()}**: **{predicted_confidence:.2f}%**")
        st.progress(float(predicted_confidence / 100))

        st.write("") 

      
        confidence_df = pd.DataFrame({
            'Species': [label_dict[i].capitalize() for i in range(len(label_dict))],
            'Probability': probabilities
        }).set_index('Species')

        st.bar_chart(confidence_df)
        st.caption("Distribution of probabilities across all flower species.")

    else:
        st.info("This model does not provide prediction probabilities (e.g., SVM without `probability=True` during training).")

st.write("---") 
st.write("Built with Streamlit from a Celebel Data Science Intern.")
