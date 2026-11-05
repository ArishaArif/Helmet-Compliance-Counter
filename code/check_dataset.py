
from pathlib import Path

root = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset"
)

images_dir = root / "images"
labels_dir = root / "labels" / "yolo"

images = {
    p.stem for p in images_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
}

labels = {
    p.stem for p in labels_dir.glob("*.txt")
    if p.name.lower() != "classes.txt"
}

print("Total images:", len(images))
print("Total label files:", len(labels))
print("Images without labels:", len(images - labels))
print("Labels without images:", len(labels - images))
