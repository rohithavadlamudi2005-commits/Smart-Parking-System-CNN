import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

model = tf.keras.models.load_model(
    "Model/smart_parking_cnn.keras"
)

img = cv2.imread("Dataset/images/0.png")

tree = ET.parse("Dataset/annotations.xml")
root = tree.getroot()

free_count = 0
occupied_count = 0

for image_tag in root.findall("image"):

    if image_tag.attrib["id"] == "0":

        for poly in image_tag.findall("polygon"):

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

            pred = model.predict(crop, verbose=0)[0][0]

            if pred > 0.5:
                occupied_count += 1
            else:
                free_count += 1

        break

print("Free Slots:", free_count)
print("Occupied Slots:", occupied_count)
print("Total Slots:", free_count + occupied_count)