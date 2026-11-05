
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\HelmetComplianceCounter\cleaned_dataset")

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

class_names = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

class_counts = Counter()
total_annotations = 0
invalid_annotations = 0

images = list(IMAGE_DIR.glob("*"))
labels = list(LABEL_DIR.glob("*.txt"))

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

print("Images:", len(images))
print("Labels:", len(labels))

print("\nMissing labels:", len(image_stems - label_stems))
print("Missing images:", len(label_stems - image_stems))

for label_file in labels:

    for line_number, line in enumerate(
        label_file.read_text().splitlines(), start=1
    ):

        parts = line.split()

        if len(parts) != 5:
            invalid_annotations += 1
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            invalid_annotations += 1
            continue

        if class_id not in class_names or w <= 0 or h <= 0:
            invalid_annotations += 1
            continue

        class_counts[class_id] += 1
        total_annotations += 1

print("\n========== CLASS DISTRIBUTION ==========")

for class_id, name in class_names.items():
    count = class_counts[class_id]
    percentage = (count / total_annotations * 100) if total_annotations else 0

    print(f"{name}: {count} ({percentage:.2f}%)")

print("\nTotal annotations:", total_annotations)
print("Invalid annotations:", invalid_annotations)
