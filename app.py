import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Animal Detection",
    page_icon="🐾",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

classes = ["kutya", "lepke", "tyuk"]

st.title("🐾 AI Animal Detection System")
st.markdown("Upload an image and let AI identify the animal.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    with col2:
        st.success(
            f"Prediction: {classes[predicted_class]}"
        )
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{classes[i]} : {prob*100:.2f}%")
        st.progress(float(prob))