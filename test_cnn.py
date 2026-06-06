import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

# Load model
model = tf.keras.models.load_model(
    "Model/smart_parking_cnn.keras"
)

# Read image
img = cv2.imread("Dataset/images/0.png")

# Read XML
tree = ET.parse("Dataset/annotations.xml")
root = tree.getroot()

for image_tag in root.findall("image"):

    if image_tag.attrib["id"] == "0":

        poly = image_tag.findall("polygon")[0]

        points = poly.attrib["points"]

        coords = []

        for p in points.split(";"):
            x, y = map(float, p.split(","))
            coords.append((int(x), int(y)))

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        crop = img[y_min:y_max, x_min:x_max]

        crop = cv2.resize(crop, (64, 64))
        crop = crop / 255.0

        crop = np.expand_dims(crop, axis=0)

        prediction = model.predict(crop)

        print("Prediction:", prediction[0][0])

        if prediction[0][0] > 0.5:
            print("Occupied")
        else:
            print("Free")

        break