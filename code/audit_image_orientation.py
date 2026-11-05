from PIL import Image
import os
import csv

IMAGE_DIR = r"data\EdgeVision Dataset\EdgeVision-Dataset\images"
OUTPUT_FILE = r"data\image_orientation_audit.csv"

orientation_names = {
    1: "Normal",
    2: "Mirrored",
    3: "Rotated 180",
    4: "Mirrored vertically",
    5: "Mirrored and rotated",
    6: "Rotated 90 clockwise",
    7: "Mirrored and rotated",
    8: "Rotated 90 counterclockwise"
}

results = []

for filename in os.listdir(IMAGE_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(IMAGE_DIR, filename)

    try:
        with Image.open(path) as img:

            orientation = img.getexif().get(274, 1)

            results.append([
                filename,
                img.size[0],
                img.size[1],
                orientation,
                orientation_names.get(orientation, "Unknown")
            ])

    except Exception as e:
        results.append([filename, "", "", "", f"Error: {e}"])

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Filename",
        "Width",
        "Height",
        "EXIF Orientation",
        "Orientation Description"
    ])

    writer.writerows(results)

print("Images checked:", len(results))

rotated = [r for r in results if r[3] in (3, 6, 8)]
mirrored = [r for r in results if r[3] in (2, 4, 5, 7)]

print("Rotated images:", len(rotated))
print("Mirrored images:", len(mirrored))
print("Normal images:", len(results) - len(rotated) - len(mirrored))

print("Audit saved to:", OUTPUT_FILE)