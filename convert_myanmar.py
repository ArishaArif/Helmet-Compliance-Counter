
import csv
import os
import shutil
from pathlib import Path
from collections import Counter

# ==================================================
# PATHS
# ==================================================

SOURCE_IMAGES = Path(r"D:\myanmar\part_7\part_7")
SOURCE_ANNOTATIONS = Path(r"D:\myanmar\annotation\annotation")
SPLIT_FILE = Path.home() / "Downloads" / "data_split.csv"

OUTPUT = Path(r"D:\HelmetComplianceCounter\video_dataset")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# ==================================================
# YOLO CLASSES
# ==================================================

CLASSES = [
    "H0_N1",
    "H0_N2",
    "H0_N3",
    "H0_N4",
    "H1_N0",
    "H1_N1",
    "H1_N2",
    "H2_N0",
    "H2_N1",
]

CLASS_MAP = {
    "DHelmet": "H1_N0",
    "DNoHelmet": "H0_N1",
    "DHelmetP1Helmet": "H2_N0",
    "DHelmetP1HelmetP2NoHelmet": "H2_N1",
    "DHelmetP1NoHelmet": "H1_N1",
    "DHelmetP1NoHelmetP2NoHelmet": "H1_N2",
    "DNoHelmetP0NoHelmet": "H0_N2",
    "DNoHelmetP1Helmet": "H1_N1",
    "DNoHelmetP1NoHelmet": "H0_N2",
    "DNoHelmetP1NoHelmetP2Helmet": "H1_N2",
    "DNoHelmetP1NoHelmetP2NoHelmet": "H0_N3",
    "DNoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": "H0_N4",
}

CLASS_IDS = {
    name: index
    for index, name in enumerate(CLASSES)
}

# ==================================================
# INITIAL CHECKS
# ==================================================

if not SOURCE_IMAGES.is_dir():
    raise FileNotFoundError(SOURCE_IMAGES)

if not SOURCE_ANNOTATIONS.is_dir():
    raise FileNotFoundError(SOURCE_ANNOTATIONS)

if not SPLIT_FILE.is_file():
    raise FileNotFoundError(SPLIT_FILE)

# ==================================================
# READ OFFICIAL SPLITS
# ==================================================

video_splits = {}

with open(SPLIT_FILE, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        video_id = row["video_id"].strip()

        if not video_id.startswith("Yangon_II_"):
            continue

        sequence_folder = SOURCE_IMAGES / video_id

        # Only include downloaded sequences.
        if not sequence_folder.is_dir():
            continue

        official_split = row["Set"].strip().lower()

        if official_split not in ("training", "validation", "test"):
            raise ValueError(
                f"Unexpected split: {official_split}"
            )

        split = {
            "training": "train",
            "validation": "val",
            "test": "test",
        }[official_split]

        video_splits[video_id] = split

if len(video_splits) != 28:
    raise ValueError(
        f"Expected 28 sequences, found {len(video_splits)}"
    )

print(f"Sequences found: {len(video_splits)}")

# ==================================================
# CHECK OUTPUT DIRECTORIES
# ==================================================

for split in ("train", "val", "test"):
    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# Remove old generated dataset files only.
for split in ("train", "val", "test"):
    for folder in (
        OUTPUT / "images" / split,
        OUTPUT / "labels" / split,
    ):
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()

# ==================================================
# PREPARE LABEL STORAGE
# ==================================================

label_lines = {}
class_counts = Counter()
unknown_labels = Counter()

processed_videos = 0
processed_frames = 0

for video_id, split in video_splits.items():

    image_folder = SOURCE_IMAGES / video_id
    annotation_file = SOURCE_ANNOTATIONS / f"{video_id}.csv"

    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)

    # Prepare empty labels for all 100 frames.
    for frame_id in range(1, 101):
        key = (split, f"{video_id}_{frame_id:02d}")
        label_lines[key] = []

    # Read annotations.
    with open(
        annotation_file,
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            original_label = row["label"].strip()

            if original_label not in CLASS_MAP:
                unknown_labels[original_label] += 1
                continue

            class_name = CLASS_MAP[original_label]
            class_id = CLASS_IDS[class_name]

            frame_id = int(row["frame_id"])

            if not 1 <= frame_id <= 100:
                raise ValueError(
                    f"Invalid frame {frame_id} in {video_id}"
                )

            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Invalid box in {video_id}, frame {frame_id}"
                )

            # Convert pixel XYWH to normalized YOLO XYWH.
            x_center = (x + w / 2) / IMAGE_WIDTH
            y_center = (y + h / 2) / IMAGE_HEIGHT

            width = w / IMAGE_WIDTH
            height = h / IMAGE_HEIGHT

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
                and 0 <= x_center - width / 2
                and x_center + width / 2 <= 1
                and 0 <= y_center - height / 2
                and y_center + height / 2 <= 1
            ):
                raise ValueError(
                    f"Out-of-bounds box in {video_id}, "
                    f"frame {frame_id}: {x}, {y}, {w}, {h}"
                )

            key = (split, f"{video_id}_{frame_id:02d}")

            label_lines[key].append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            class_counts[class_name] += 1

    processed_videos += 1

    print(f"Read {video_id} -> {split}")

# ==================================================
# STOP IF UNKNOWN LABELS EXIST
# ==================================================

if unknown_labels:
    print("\nUnknown annotation labels detected:")

    for name, count in unknown_labels.items():
        print(f"{name}: {count}")

    raise ValueError(
        "Conversion stopped because some labels are unmapped."
    )

# ==================================================
# CREATE IMAGE HARD LINKS
# ==================================================

for video_id, split in video_splits.items():

    for frame_id in range(1, 101):

        filename = f"{frame_id:02d}.jpg"

        source = SOURCE_IMAGES / video_id / filename

        if not source.is_file():
            raise FileNotFoundError(source)

        destination = (
            OUTPUT / "images" / split /
            f"{video_id}_{filename}"
        )

        # Hard links avoid duplicating image storage.
        os.link(source, destination)

        processed_frames += 1

    print(f"Linked images: {video_id}")

# ==================================================
# WRITE YOLO LABEL FILES
# ==================================================

for (split, image_stem), lines in label_lines.items():

    label_file = (
        OUTPUT / "labels" / split /
        f"{image_stem}.txt"
    )

    with open(label_file, "w", encoding="utf-8") as file:

        if lines:
            file.write("\n".join(lines) + "\n")

# ==================================================
# CREATE data.yaml
# ==================================================

yaml_content = [
    "path: D:/HelmetComplianceCounter/video_dataset",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "",
    "names:",
]

for class_id, class_name in enumerate(CLASSES):
    yaml_content.append(
        f"  {class_id}: {class_name}"
    )

with open(
    OUTPUT / "data.yaml",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(yaml_content) + "\n")

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========== CONVERSION SUMMARY ==========")

print(f"Videos processed: {processed_videos}")
print(f"Frames processed: {processed_frames}")
print(f"YOLO classes: {len(CLASSES)}")

print("\nAnnotations per class:")

for class_name in CLASSES:
    print(
        f"{class_name}: {class_counts[class_name]}"
    )

print("\nImages and labels per split:")

for split in ("train", "val", "test"):

    image_count = len(
        list((OUTPUT / "images" / split).glob("*.jpg"))
    )

    label_count = len(
        list((OUTPUT / "labels" / split).glob("*.txt"))
    )

    print(
        f"{split}: {image_count} images, "
        f"{label_count} labels"
    )

print("\nNo unknown annotation labels.")
print("Conversion completed successfully.")
