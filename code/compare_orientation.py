from PIL import Image, ImageDraw, ImageOps
import xml.etree.ElementTree as ET

image_path = r"data\EdgeVision Dataset\EdgeVision-Dataset\images\Image_01035.jpg"
xml_path = r"data\EdgeVision Dataset\EdgeVision-Dataset\labels\voc_xml\Image_01035.xml"

CLASS_COLORS = {
    "BikeWithRider": "blue",
    "NoHelmet": "red",
    "Helmet": "green"
}

# Load upright image
img = Image.open(image_path)
img = ImageOps.exif_transpose(img)

# Load annotations
tree = ET.parse(xml_path)
root = tree.getroot()

draw = ImageDraw.Draw(img)

for obj in root.findall("object"):
    name = obj.find("name").text
    box = obj.find("bndbox")

    xmin = int(float(box.find("xmin").text))
    ymin = int(float(box.find("ymin").text))
    xmax = int(float(box.find("xmax").text))
    ymax = int(float(box.find("ymax").text))

    # Look up the color from dictionary, default to white if class isn't found
    color = CLASS_COLORS.get(name, "white")

    draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=5)
    draw.text((xmin, ymin), name, fill=color)

img.save("code/Image_01035_comparison.jpg")
print("Saved comparison image.")