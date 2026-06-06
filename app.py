import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image

st.set_page_config(page_title="Smart Parking System")

st.title("🚗 Smart Parking System")

# Paths
ANNOTATION_FILE = "Dataset/annotations.xml"
IMAGE_FOLDER = "Dataset/images"

# Load CNN Model
model = tf.keras.models.load_model(
    "Model/smart_parking_cnn.keras"
)

st.success("CNN Model Loaded Successfully")

# Read XML
tree = ET.parse(ANNOTATION_FILE)
root = tree.getroot()

# Get image names
image_names = []

for img in root.findall("image"):
    image_names.append(
        os.path.basename(img.attrib["name"])
    )

# Dropdown
selected_image = st.selectbox(
    "Select Parking Lot Image",
    image_names
)

# Image path
image_path = os.path.join(
    IMAGE_FOLDER,
    selected_image
)

if os.path.exists(image_path):

    # Display image
    image = Image.open(image_path)

    st.image(
        image,
        caption=selected_image,
        use_container_width=True
    )

    # Read image using OpenCV
    img_cv = cv2.imread(image_path)

    free_count = 0
    occupied_count = 0

    # Find selected image in XML
    for image_tag in root.findall("image"):

        if os.path.basename(
            image_tag.attrib["name"]
        ) == selected_image:

            for poly in image_tag.findall("polygon"):

                points = poly.attrib["points"]

                coords = []

                for p in points.split(";"):
                    x, y = map(float, p.split(","))
                    coords.append(
                        (int(x), int(y))
                    )

                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]

                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)

                crop = img_cv[
                    y_min:y_max,
                    x_min:x_max
                ]

                if crop.size == 0:
                    continue

                crop = cv2.resize(
                    crop,
                    (64, 64)
                )

                crop = crop / 255.0

                crop = np.expand_dims(
                    crop,
                    axis=0
                )

                prediction = model.predict(
                    crop,
                    verbose=0
                )[0][0]

                if prediction > 0.5:
                    occupied_count += 1
                else:
                    free_count += 1

            break

    st.subheader("🚗 Parking Statistics")

    st.success(
        f"🟢 Free Slots: {free_count}"
    )

    st.error(
        f"🔴 Occupied Slots: {occupied_count}"
    )

    st.info(
        f"📊 Total Slots: {free_count + occupied_count}"
    )

else:
    st.error("Image Not Found")