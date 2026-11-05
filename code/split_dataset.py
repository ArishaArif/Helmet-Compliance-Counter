from pathlib import Path
from collections import Counter
import random
import shutil
import math

ROOT = Path(r"D:\HelmetComplianceCounter")

SOURCE = ROOT / "cleaned_dataset"
OUTPUT = ROOT / "dataset_split"

IMAGE_DIR = SOURCE / "images"
LABEL_DIR = SOURCE / "labels"

SEED = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10
}

CLASS_NAMES = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Collect images
images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

# Verify image-label pairs
for image in images:
    label = LABEL_DIR / f"{image.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(
            f"Missing label for {image.name}"
        )

if OUTPUT.exists():
    raise FileExistsError(
        "dataset_split already exists. "
        "Rename or remove it before running this script again."
    )

# Shuffle reproducibly
random.Random(SEED).shuffle(images)

total = len(images)

train_end = int(total * 0.70)
val_end = train_end + int(total * 0.20)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

# Create output folders
for split in splits:
    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# Statistics
split_counts = {}
class_counts = {}

for split, split_images in splits.items():

    counts = Counter()
    annotations = 0

    for image in split_images:

        label = LABEL_DIR / f"{image.stem}.txt"

        # Copy image
        shutil.copy2(
            image,
            OUTPUT / "images" / split / image.name
        )

        # Read and prepare labels
        cleaned_lines = []

        for line in label.read_text().splitlines():

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid annotation in {label.name}"
                )

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class ID in {label.name}"
                )

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError(
                    f"Non-finite coordinate in {label.name}"
                )

            # Convert center coordinates to box corners
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            # Clip box to image boundaries
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            new_w = x2 - x1
            new_h = y2 - y1

            if new_w <= 0 or new_h <= 0:
                raise ValueError(
                    f"Invalid box after clipping in {label.name}"
                )

            new_x = (x1 + x2) / 2
            new_y = (y1 + y2) / 2

            cleaned_lines.append(
                f"{class_id} {new_x:.6f} {new_y:.6f} "
                f"{new_w:.6f} {new_h:.6f}"
            )

            counts[class_id] += 1
            annotations += 1

        # Save label
        output_label = (
            OUTPUT / "labels" / split / label.name
        )

        output_label.write_text(
            "\n".join(cleaned_lines) +
            ("\n" if cleaned_lines else "")
        )

    split_counts[split] = len(split_images)
    class_counts[split] = counts

# Create YOLO configuration
yaml_content = f"""path: {OUTPUT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: BikeWithRider
  1: NoHelmet
  2: Helmet
"""

(OUTPUT / "data.yaml").write_text(yaml_content)

# Print summary
print("\n========== DATASET SPLIT SUMMARY ==========")

for split in splits:
    print(f"\n{split.upper()}")
    print("Images:", split_counts[split])

    for class_id, name in CLASS_NAMES.items():
        print(
            f"{name}: {class_counts[split][class_id]}"
        )

print("\nTotal images:", sum(split_counts.values()))
print("Random seed:", SEED)
print("\nDataset saved to:", OUTPUT)
print("YOLO configuration:", OUTPUT / "data.yaml")