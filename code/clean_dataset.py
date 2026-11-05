
from pathlib import Path
from PIL import Image
import shutil
import csv

# Original dataset
ROOT = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset"
)

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels" / "yolo"

# Separate cleaned dataset
OUTPUT_DIR = Path(r"D:\HelmetComplianceCounter\cleaned_dataset")
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# EXIF orientations representing rotations
ROTATED_ORIENTATIONS = {3, 6, 8}

excluded_images = []
removed_annotations = []
copied_images = 0
copied_labels = 0
total_annotations_kept = 0

image_extensions = {".jpg", ".jpeg", ".png"}

# Process every image
for image_path in IMAGE_DIR.iterdir():

    if image_path.suffix.lower() not in image_extensions:
        continue

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Missing label:", image_path.name)
        continue

    # Check EXIF orientation
    with Image.open(image_path) as img:
        orientation = img.getexif().get(274, 1)

    # Exclude rotated images and their labels
    if orientation in ROTATED_ORIENTATIONS:

        excluded_images.append([
            image_path.name,
            orientation,
            "Rotated EXIF orientation"
        ])

        continue

    # Read original YOLO annotations
    lines = label_path.read_text().splitlines()
    valid_lines = []

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) != 5:
            valid_lines.append(line)
            continue

        try:
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            valid_lines.append(line)
            continue

        # Remove only annotations with non-positive dimensions
        if width <= 0 or height <= 0:

            removed_annotations.append([
                image_path.name,
                line_number,
                line,
                "Non-positive bounding-box dimensions"
            ])

            continue

        valid_lines.append(line)

    # Copy image without modifying it
    shutil.copy2(
        image_path,
        OUTPUT_IMAGES / image_path.name
    )

    # Save cleaned annotations
    output_label = OUTPUT_LABELS / label_path.name
    output_label.write_text(
        "\n".join(valid_lines) + ("\n" if valid_lines else "")
    )

    copied_images += 1
    copied_labels += 1
    total_annotations_kept += len(valid_lines)

# Save exclusion report
with open(OUTPUT_DIR / "exclusion_report.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow([
        "Image",
        "EXIF Orientation",
        "Reason"
    ])
    writer.writerows(excluded_images)

# Save removed annotation report
with open(OUTPUT_DIR / "removed_annotations.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow([
        "Image",
        "Line Number",
        "Original Annotation",
        "Reason"
    ])
    writer.writerows(removed_annotations)

print("\n========== CLEANING SUMMARY ==========")
print("Images retained:", copied_images)
print("Labels retained:", copied_labels)
print("Rotated images excluded:", len(excluded_images))
print("Invalid annotations removed:", len(removed_annotations))
print("Annotations retained:", total_annotations_kept)

print("\nCleaned dataset:", OUTPUT_DIR)
print("Exclusion report:", OUTPUT_DIR / "exclusion_report.csv")
print("Removed annotations report:", OUTPUT_DIR / "removed_annotations.csv")
