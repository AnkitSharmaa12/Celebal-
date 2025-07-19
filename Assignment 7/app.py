import streamlit as st
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import os

# Constants
IMG_SIZE = (112, 112)

# Load your model
@st.cache_resource
def load_yoga_model():
    model = load_model("yoga_pose_model.keras")  # Already correctly named
    return model

# Define label map (ensure it matches your training labels)
label_map = {
    0: 'downdog',
    1: 'goddess',
    2: 'plank',
    3: 'tree',
    4: 'warrior2'
}

def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🧘 Yoga Pose Classifier")
st.write("Upload an image of a yoga pose to predict the asana.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Image could not be decoded. Please upload a valid image file.")
        st.stop()

    # Display uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_yoga_model()

    # Preprocess image
    input_image = preprocess_image(image)

    # Prediction with error handling
    try:
        preds = model.predict(input_image)[0]
        pred_class = np.argmax(preds)
        pred_label = label_map[pred_class]
        confidence = preds[pred_class]

        st.success(f"🧘 Predicted Pose: **{pred_label}** with {confidence:.2%} confidence")

        # Show bar chart of all class probabilities
        st.subheader("Class Confidence Scores")
        conf_scores = pd.DataFrame({
            "Pose": list(label_map.values()),
            "Confidence": preds
        })
        st.bar_chart(conf_scores.set_index("Pose"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
