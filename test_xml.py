import xml.etree.ElementTree as ET

tree = ET.parse("Dataset/annotations.xml")
root = tree.getroot()

for img in root.findall("image"):

    if img.attrib["id"] == "0":

        for poly in img.findall("polygon")[:1]:
            print(poly.attrib)

        break