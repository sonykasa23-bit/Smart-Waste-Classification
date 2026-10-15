
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Classification")
st.write(
    "Upload a waste image and let the CNN model "
    "predict its category."
)

MODEL_PATH = Path(__file__).parent / "waste_classification_model.keras"
LABEL_PATH = Path(__file__).parent / "class_names.json"

# Load the trained CNN
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load class labels
with open(LABEL_PATH, "r") as file:
    class_names = json.load(file)

# Image upload
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        width=300
    )

    if st.button("Predict Waste Category"):
        # Prepare image exactly as during training
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict category
        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        st.success(
            f"Predicted Category: {predicted_class.title()}"
        )
        st.metric("Prediction Score", f"{confidence:.2f}%")

        st.subheader("Category Scores")
        for index, name in enumerate(class_names):
            st.write(f"{name.title()}: {predictions[index] * 100:.2f}%")
            st.progress(float(predictions[index]))
