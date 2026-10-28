
import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------
# 1. Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# ----------------------------------
# 2. File Paths
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "waste_classification_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "class_names.json"
)

IMG_SIZE = (224, 224)

# ----------------------------------
# 3. Class Names
# ----------------------------------
DEFAULT_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            saved_classes = json.load(file)

        if isinstance(saved_classes, list) and len(saved_classes) == 6:
            CLASS_NAMES = saved_classes
        else:
            CLASS_NAMES = DEFAULT_CLASSES

    except (json.JSONDecodeError, OSError):
        CLASS_NAMES = DEFAULT_CLASSES
else:
    CLASS_NAMES = DEFAULT_CLASSES

# ----------------------------------
# 4. Load Trained Model
# ----------------------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


# ----------------------------------
# 5. Website Interface
# ----------------------------------
st.title("♻️ Smart Waste Classification")

st.markdown(
    """
    ### AI-Based Waste Detection Using CNN

    Upload an image of a waste item. The trained deep learning
    model will predict its category.
    """
)

st.info(
    "Supported categories: Cardboard, Glass, Metal, Paper, Plastic and Trash."
)

# ----------------------------------
# 6. Verify Model File
# ----------------------------------
if not os.path.isfile(MODEL_PATH):
    st.error("Trained model file not found!")

    st.write("Expected model filename:")
    st.code("waste_classification_model.keras")

    st.write("Expected repository structure:")
    st.code(
        "Smart_Waste_Classification/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "├── class_names.json\n"
        "└── waste_classification_model.keras"
    )

    st.warning(
        "Add the trained model to your GitHub repository's "
        "main folder, then redeploy the app."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.stop()

# ----------------------------------
# 7. Upload Image
# ----------------------------------
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

                resized_image = image.resize(IMG_SIZE)

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                # Predict using the trained model
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(predictions) != len(CLASS_NAMES):
                    st.error(
                        "The number of model outputs does not match "
                        "the six waste categories. Check your model."
                    )
                    st.stop()

                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.success(
                f"Predicted Waste Category: {predicted_class.title()}"
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            # Show scores for all classes
            st.subheader("Prediction Scores")

            for name, score in zip(CLASS_NAMES, predictions):
                st.write(f"**{name.title()}** — {score * 100:.2f}%")
                st.progress(
                    min(1.0, max(0.0, float(score)))
                )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the classification is correct."
            )

    except Exception as error:
        st.error("Unable to process this image.")
        st.code(str(error))

# ----------------------------------
# 8. Footer
# ----------------------------------
st.divider()

st.markdown(
    """
    **Project:** Smart Waste Classification Using CNN

    **Technologies:** Python, TensorFlow, CNN, Streamlit

    **Purpose:** AI-assisted classification of common waste materials.
    """
)
