import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)


def preprocess_uploaded_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("Brain Tumor Detection from MRI")
st.markdown("Upload an MRI image. The model will predict whether a brain tumor is present.")


uploaded_file = st.file_uploader("Upload MRI Image")

if uploaded_file is not None:
    try:
     
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded MRI Image", width=250)

        with st.spinner("Analyzing..."):
            processed_img = preprocess_uploaded_image(img)
            prediction = model.predict(processed_img)[0][0]
            confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
            label = "Tumor Detected (YES)" if prediction > 0.5 else "No Tumor Detected (NO)"
            color = "red" if prediction > 0.5 else "green"

       
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 10px; background-color: {color}; color: white; font-size: 20px; text-align: center;'>
            <strong>{label}</strong><br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please try again with a correct image format.")
