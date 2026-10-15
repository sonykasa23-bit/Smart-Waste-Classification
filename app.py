
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️"
)

st.title("Smart Waste Classification ♻️")
st.write("Upload a waste image to predict its category.")

# Model location
MODEL_PATH = Path(__file__).parent / "waste_classification_model.keras"

# Keep the training class order
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Load the CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file is missing. Upload waste_classification_model.keras.")
    st.stop()

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", width=300)

    if st.button("Predict"):
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        if len(predictions) != len(class_names):
            st.error("Model classes do not match the class list.")
            st.stop()

        index = int(np.argmax(predictions))

        st.success(
            f"Predicted Category: {class_names[index].title()}"
        )
        st.metric(
            "Confidence",
            f"{float(predictions[index]) * 100:.2f}%"
        )