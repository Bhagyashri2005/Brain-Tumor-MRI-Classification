import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_resnet50.keras")


model = load_model()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert to NumPy
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ResNet50 preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    # Predicted class
    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 NeuroScan AI")

    st.caption(
        "Brain MRI Classification System"
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("**Architecture:** ResNet50")
    st.write("**Approach:** Transfer Learning")
    st.write("**Input Size:** 224 × 224")

    st.divider()

    st.subheader("🧬 Classes")

    st.write("🟣 Glioma")
    st.write("🟠 Meningioma")
    st.write("🟢 No Tumor")
    st.write("🔵 Pituitary")

    st.divider()

    st.subheader("📊 Test Performance")

    st.metric(
        "Accuracy",
        "88.21%"
    )

    st.metric(
        "Precision",
        "88.18%"
    )

    st.metric(
        "Recall",
        "88.21%"
    )

    st.metric(
        "F1-Score",
        "87.98%"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🧠 NeuroScan AI")

st.subheader(
    "Brain Tumor MRI Image Classification"
)

st.write(
    "Upload a brain MRI image and use the trained "
    "ResNet50 model to classify it into one of four categories."
)

st.info(
    "🔬 Deep Learning • Medical Imaging • "
    "4-Class Classification"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.header("🖼️ MRI Preview")

    # --------------------------------------------------------
    # IMAGE + INFORMATION
    # --------------------------------------------------------

    image_col, info_col = st.columns(
        [1.3, 1],
        gap="large"
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

    with info_col:

        st.subheader(
            "📋 Image Information"
        )

        st.metric(
            "Image Size",
            f"{image.size[0]} × {image.size[1]}"
        )

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Format:** {image.format}"
        )

        st.write(
            f"**Color Mode:** {image.mode}"
        )

        st.write(
            "**Model Input:** 224 × 224"
        )


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔍 Analyze MRI",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing MRI with ResNet50..."
        ):

            predicted_class, confidence, probabilities = (
                predict_image(image)
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.header("📋 Analysis Result")

        display_class = (
            predicted_class
            .replace("_", " ")
            .title()
        )

        st.success(
            f"### 🧠 Predicted Class: {display_class}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.header(
            "📊 Class Probability Distribution"
        )

        probability_cols = st.columns(4)

        for i, probability in enumerate(
            probabilities
        ):

            class_name = (
                class_names[i]
                .replace("_", " ")
                .title()
            )

            with probability_cols[i]:

                st.metric(
                    class_name,
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.header(
            "🤖 Model Information"
        )

        model_col1, model_col2, model_col3 = st.columns(3)

        with model_col1:

            st.metric(
                "Architecture",
                "ResNet50"
            )

        with model_col2:

            st.metric(
                "Test Accuracy",
                "88.21%"
            )

        with model_col3:

            st.metric(
                "Input Size",
                "224 × 224"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 NeuroScan AI • Brain Tumor MRI Image Classification"
)

st.caption(
    "Powered by TensorFlow • Keras • ResNet50 • Streamlit"
)