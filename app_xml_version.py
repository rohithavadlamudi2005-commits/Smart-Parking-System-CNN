import streamlit as st
import os
import xml.etree.ElementTree as ET
from PIL import Image
import tensorflow as tf

st.title("🚗 Smart Parking System")

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
    image_names.append(os.path.basename(img.attrib["name"]))

selected_image = st.selectbox(
    "Select Parking Lot Image",
    image_names
)

image_path = os.path.join(
    IMAGE_FOLDER,
    selected_image
)

if os.path.exists(image_path):

    image = Image.open(image_path)

    st.image(
        image,
        caption=selected_image,
        use_container_width=True
    )

    # XML Statistics
    for img in root.findall("image"):

        if os.path.basename(img.attrib["name"]) == selected_image:

            free_count = 0
            occupied_count = 0

            for poly in img.findall("polygon"):

                if poly.attrib["label"] == "free_parking_space":
                    free_count += 1
                else:
                    occupied_count += 1

            st.subheader("Parking Statistics")

            st.write("Total Slots:", free_count + occupied_count)
            st.write("Free Slots:", free_count)
            st.write("Occupied Slots:", occupied_count)

            break

else:
    st.error("Image Not Found")