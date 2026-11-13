from PIL import Image
import os

paths = [
    r"data\EdgeVision Dataset\EdgeVision-Dataset\images\Image_01035.jpg",
]

for path in paths:
    img = Image.open(path)

    print("\nFile:", path)
    print("Dimensions:", img.size)
    print("EXIF orientation:", img.getexif().get(274, "Missing"))
    print("Format:", img.format)