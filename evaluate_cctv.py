
from pathlib import Path
import shutil
import yaml

# Project paths
ROOT = Path(r"D:\HelmetComplianceCounter")

SOURCE_IMAGES = ROOT / "fine_tune_dataset" / "images" / "val"
SOURCE_LABELS = ROOT / "fine_tune_dataset" / "labels" / "val"

OUTPUT = ROOT / "cctv_evaluation"
OUTPUT_IMAGES = OUTPUT / "images" / "val"
OUTPUT_LABELS = OUTPUT / "labels" / "val"

PATTERN = "*WhatsApp-Video-2026-03-25-at-15_24_49*"

# Find the held-out CCTV images
images = sorted(SOURCE_IMAGES.glob(PATTERN))

if not images:
    raise SystemExit("ERROR: No matching CCTV images found.")

print(f"Found {len(images)} CCTV images.")

# Create separate output folders
OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# Copy images and corresponding labels
copied = 0
missing = []

for image_path in images:
    label_path = SOURCE_LABELS / (image_path.stem + ".txt")

    if not label_path.exists():
        missing.append(image_path.name)
        continue

    shutil.copy2(image_path, OUTPUT_IMAGES / image_path.name)
    shutil.copy2(label_path, OUTPUT_LABELS / label_path.name)
    copied += 1

# Verify
print(f"\nSuccessfully copied: {copied}")
print(f"Missing labels: {len(missing)}")

if missing:
    print("\nMissing label files:")
    for name in missing:
        print(name)

if copied != 101 or missing:
    raise SystemExit(
        "\nERROR: Expected exactly 101 complete image-label pairs."
    )

# Create evaluation YAML
data = {
    "path": str(OUTPUT).replace("\\", "/"),
    "train": "images/val",
    "val": "images/val",
    "names": {
        0: "BikeWithRider",
        1: "NoHelmet",
        2: "Helmet"
    }
}

yaml_path = OUTPUT / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("\nEvaluation dataset ready!")
print(f"Images: {OUTPUT_IMAGES}")
print(f"Labels: {OUTPUT_LABELS}")
print(f"YAML: {yaml_path}")
