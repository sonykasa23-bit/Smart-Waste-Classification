
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Classification")
st.write(
    "Upload a waste image to classify it into one "
    "of six categories using a trained CNN model."
)

MODEL_PATH = "best_smart_waste_model.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Add the trained model to the repository.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded waste image", width=350)

    if st.button("Classify Waste", type="primary"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize((224, 224))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.success(f"Predicted category: {predicted_class.title()}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.subheader("Prediction scores")
        for name, score in zip(CLASS_NAMES, predictions):
            st.write(f"**{name.title()}** — {score * 100:.2f}%")
            st.progress(float(score))

st.caption(
    "AI-based waste classification. Predictions may be incorrect "
    "for unclear images or waste items outside the training categories."
)
