
import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "waste_classification_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "class_names.json"
)

DEFAULT_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# --------------------------------------------------
# 3. LOAD CLASS NAMES
# --------------------------------------------------
if os.path.isfile(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            loaded_classes = json.load(f)

        if isinstance(loaded_classes, list) and len(loaded_classes) == 6:
            CLASS_NAMES = loaded_classes
        else:
            CLASS_NAMES = DEFAULT_CLASSES

    except (json.JSONDecodeError, OSError):
        CLASS_NAMES = DEFAULT_CLASSES
else:
    CLASS_NAMES = DEFAULT_CLASSES

# --------------------------------------------------
# 4. LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

st.title("♻️ Smart Waste Classification")
st.subheader("AI-Based Waste Detection Using CNN")

st.write(
    "Upload an image of a waste item. "
    "The trained CNN model will predict its category."
)

st.info(
    "Supported categories: Cardboard, Glass, Metal, "
    "Paper, Plastic and Trash."
)

if not os.path.isfile(MODEL_PATH):
    st.error("Trained model file not found.")
    st.write("Place this file beside app.py:")
    st.code("waste_classification_model.keras")
    st.stop()

try:
    model = load_model(MODEL_PATH)

    # Get the model's expected input dimensions
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        st.error("This app expects a model with a single image input.")
        st.stop()

    if len(input_shape) != 4:
        st.error(f"Unexpected model input shape: {input_shape}")
        st.stop()

    if input_shape[1] is None or input_shape[2] is None:
        IMG_SIZE = (150, 150)
    else:
        IMG_SIZE = (int(input_shape[1]), int(input_shape[2]))

    st.sidebar.write("Model input shape:", str(input_shape))
    st.sidebar.write("Image size used:", f"{IMG_SIZE[0]} × {IMG_SIZE[1]}")

except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# 5. IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Waste Image",
            width=350
        )

        if st.button("Classify Waste", type="primary"):
            with st.spinner("Analyzing image..."):

                # Resize image to the dimensions expected by the model
                resized_image = image.resize(IMG_SIZE)

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                # Add batch dimension
                image_array = np.expand_dims(image_array, axis=0)

                # Predict
                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                predictions = np.asarray(predictions)

                if predictions.ndim != 2 or predictions.shape[0] != 1:
                    st.error(
                        "The model returned an unexpected prediction shape."
                    )
                    st.stop()

                predictions = predictions[0]

                if len(predictions) != len(CLASS_NAMES):
                    st.error(
                        "The model output count does not match the six "
                        "waste categories. Check the model and class_names.json."
                    )
                    st.write("Model outputs:", len(predictions))
                    st.write("Class names:", CLASS_NAMES)
                    st.stop()

                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.success(
                f"Predicted Waste Category: {predicted_class.title()}"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.subheader("Prediction Scores")

            for name, score in zip(CLASS_NAMES, predictions):
                score = float(score)

                st.write(f"**{name.title()}** — {score * 100:.2f}%")

                st.progress(
                    min(1.0, max(0.0, score))
                )

            st.caption(
                "Confidence is the model's predicted score, not a guarantee "
                "that the classification is correct."
            )

    except Exception as error:
        st.error("Unable to process this image.")
        st.code(str(error))

# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------
st.divider()

st.markdown("""
### About This Project

**Project:** Smart Waste Classification Using CNN

**Technologies:** Python, TensorFlow, CNN, Streamlit

**Purpose:** AI-assisted classification of common waste materials.
""")
