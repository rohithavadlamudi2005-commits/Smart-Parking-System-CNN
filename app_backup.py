import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("Model/smart_parking_cnn.keras")

st.title("🚗 Smart Parking System")

uploaded_file = st.file_uploader(
    "Upload Parking Slot Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((64, 64))
    img = np.array(img) / 255.0

    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("🔴 Occupied Parking Slot")
    else:
        st.success("🟢 Free Parking Slot")