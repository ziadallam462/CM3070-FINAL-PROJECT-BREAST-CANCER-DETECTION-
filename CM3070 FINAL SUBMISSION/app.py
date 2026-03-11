import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("finetuned_balanced_resnet.h5")  
    return model

model = load_model()

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('L')  
    image = image.resize((100, 100))
    image = np.array(image)
    image = cv2.merge([image, image, image])
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    
    return image

st.title("🩺 Breast Cancer Detection from Mammograms")
st.write("Upload a mammogram image to classify as **Normal** or **Cancer**.")

uploaded_file = st.file_uploader("Upload a mammogram image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    processed_img = preprocess_image(uploaded_file)
    prediction = model.predict(processed_img)[0][0]
    label = "Cancer" if prediction > 0.20042022 else "Normal"
    confidence = prediction if prediction > 0.20042022 else 1 - prediction
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2f}")

