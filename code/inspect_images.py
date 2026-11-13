
from pathlib import Path
from PIL import Image, ImageDraw
import random

root = Path(r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset")
images_dir = root / "images"
labels_dir = root / "labels" / "yolo"
output_dir = root / "inspection"
output_dir.mkdir(parents=True, exist_ok=True)

class_names = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

colors = {
    0: "blue",
    1: "red",
    2: "green"
}

image_files = list(images_dir.glob("*.jpg"))
random.seed(42)
samples = random.sample(image_files, min(10, len(image_files)))

for image_path in samples:
    label_path = labels_dir / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Missing label:", image_path.name)
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height

        color = colors.get(class_id, "yellow")
        name = class_names.get(class_id, str(class_id))

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 15)), name, fill=color)

    output_path = output_dir / image_path.name
    image.save(output_path)
    print("Saved:", output_path)

print("Inspection images saved in:", output_dir)
