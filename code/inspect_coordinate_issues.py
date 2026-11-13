from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"D:\HelmetComplianceCounter\cleaned_dataset")
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
OUTPUT_DIR = Path(r"D:\HelmetComplianceCounter\documentation\coordinate_review")

OUTPUT_DIR.mkdir(exist_ok=True)

colors = {
    0: "blue",
    1: "red",
    2: "green"
}

for filename in ["Image_02080", "Image_02187"]:
    image_path = IMAGE_DIR / f"{filename}.jpg"
    label_path = LABEL_DIR / f"{filename}.txt"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    img_width, img_height = image.size

    for line in label_path.read_text().splitlines():
        class_id, x, y, w, h = map(float, line.split())

        x1 = (x - w / 2) * img_width
        y1 = (y - h / 2) * img_height
        x2 = (x + w / 2) * img_width
        y2 = (y + h / 2) * img_height

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=colors[int(class_id)],
            width=3
        )

    image.save(OUTPUT_DIR / f"{filename}_review.jpg")

print("Previews saved to:", OUTPUT_DIR)