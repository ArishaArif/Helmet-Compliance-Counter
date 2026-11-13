from pathlib import Path
from PIL import Image, ImageDraw
import random

root = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset"
)

images_dir = root / "images"
labels_dir = root / "labels" / "yolo"
output_dir = root / "inspection_mixed"
output_dir.mkdir(exist_ok=True)

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
# Find images containing all three classes.
matches = []

for label_file in labels_dir.glob("*.txt"):
    if label_file.name.lower() == "classes.txt":
        continue

    found_classes = set()

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if parts and parts[0].isdigit():
            found_classes.add(int(parts[0]))

    if found_classes == {0, 1, 2}:
        matches.append(label_file)

# Choose up to 5 matching images.
random.seed(42)
selected = random.sample(matches, min(5, len(matches)))

for label_file in selected:
    image_path = None

    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / (label_file.stem + extension)
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5 or not parts[0].isdigit():
            continue

        class_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height

        label = class_names.get(class_id, str(class_id))
        
        # Using the defined dictionary colors instead of hardcoded red
        box_color = colors.get(class_id, "yellow")
        
        draw.rectangle((x1, y1, x2, y2), outline=box_color, width=3)
        draw.text((x1, max(0, y1 - 15)), label, fill=box_color)

    output_path = output_dir / f"{label_file.stem}_mixed.jpg"
    image.save(output_path)
    print("Saved:", output_path)

print("Total images containing all three classes:", len(matches))
print("Images inspected:", len(selected))