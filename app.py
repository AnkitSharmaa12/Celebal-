import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
from joblib import load
import os

# Load scaler and models
scaler = load("scaler.joblib")
models = {
    "Random Forest": load("random_forest_model.joblib"),
    "SVM": load("svm_model.joblib"),
    # "Gradient Boosting": load("gb_model.joblib")
}

# Class labels 
label_dict = {
    0: "daisy",
    1: "dandelion",
    2: "rose",
    3: "sunflower",
    4: "tulip"
}

# App title
st.title("🌼 Flower Species Recognition Dashboard")
st.write("Upload a flower image and select a model to classify the species.")

# Model selection
model_choice = st.selectbox("Select Classification Model", list(models.keys()))

# Image uploader
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (128, 128))

    st.image(image, caption="Uploaded Flower Image", use_column_width=True)

    # Convert to grayscale and extract HOG features
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)

    # Scale features
    features_scaled = scaler.transform([features])

    # Make prediction
    model = models[model_choice]
    prediction = model.predict(features_scaled)[0]

    # Display result
    st.success(f"🌺 Predicted Flower Species: **{label_dict[prediction].capitalize()}**")
