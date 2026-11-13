from PIL import Image, ImageOps

image_path = r"data\EdgeVision Dataset\EdgeVision-Dataset\images\Image_01035.jpg"

img = Image.open(image_path)

# Apply the EXIF orientation to the actual pixels
upright = ImageOps.exif_transpose(img)

# Save a separate copy
upright.save("code/Image_01035_upright.jpg")

print("Original size:", img.size)
print("Corrected size:", upright.size)
print("Upright copy saved successfully.")