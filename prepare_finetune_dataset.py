
from pathlib import Path
import shutil
import re
from collections import Counter, defaultdict

PROJECT = Path(r"D:\HelmetComplianceCounter")
EDGE = PROJECT / "dataset_split"
CCTV = Path(r"C:\Users\U.S.A Trader\Downloads\Helmet vs No Helmet.v3i.yolov11")
OUT = PROJECT / "fine_tune_dataset"

HOLDOUT_SOURCE = "WhatsApp-Video-2026-03-25-at-15_24_49_mp4"

# CCTV IDs -> EdgeVision IDs
CLASS_MAP = {
    0: 0,  # bike -> BikeWithRider
    1: 2,  # helmet -> Helmet
    2: 1,  # no helmet -> NoHelmet
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if OUT.exists():
    raise SystemExit(
        f"STOP: {OUT} already exists. Nothing was changed.\n"
        "Rename or move that folder before running this script again."
    )

for path in [EDGE, CCTV]:
    if not path.exists():
        raise SystemExit(f"Missing folder: {path}")

# Create output structure
for split in ["train", "val", "test"]:
    (OUT / "images" / split).mkdir(parents=True)
    (OUT / "labels" / split).mkdir(parents=True)

edge_counts = Counter()
cctv_counts = Counter()
source_counts = defaultdict(Counter)
object_counts = Counter()


def copy_image_and_label(image, label, destination_split, prefix=""):
    new_name = prefix + image.name
    new_label_name = Path(new_name).stem + ".txt"

    dest_image = OUT / "images" / destination_split / new_name
    dest_label = OUT / "labels" / destination_split / new_label_name

    if dest_image.exists() or dest_label.exists():
        raise RuntimeError(f"Filename collision: {new_name}")

    shutil.copy2(image, dest_image)
    return dest_label


# 1. Copy original EdgeVision dataset without changing labels
print("\nCOPYING EDGEVISION DATASET")

for split in ["train", "val", "test"]:
    image_dir = EDGE / "images" / split
    label_dir = EDGE / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        raise SystemExit(f"Missing EdgeVision folders for split: {split}")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image in images:
        label = label_dir / (image.stem + ".txt")

        if not label.exists():
            raise RuntimeError(f"Missing EdgeVision label: {label}")

        dest_label = copy_image_and_label(image, label, split)
        shutil.copy2(label, dest_label)

        edge_counts[split] += 1

    print(f"{split}: {edge_counts[split]} images")


# 2. Add CCTV dataset, remapping classes
print("\nADDING CCTV DATASET")

for original_split in ["train", "valid", "test"]:
    image_dir = CCTV / original_split / "images"
    label_dir = CCTV / original_split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise SystemExit(f"Missing CCTV folders: {original_split}")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image in images:
        label = label_dir / (image.stem + ".txt")

        if not label.exists():
            raise RuntimeError(f"Missing CCTV label: {label}")

        source = re.sub(r"_mp4-.*$", "_mp4", image.stem)

        if source == image.stem:
            raise RuntimeError(
                f"Could not identify source video for: {image.name}"
            )

        destination_split = (
            "val" if source == HOLDOUT_SOURCE else "train"
        )

        dest_label = copy_image_and_label(
            image, label, destination_split, prefix="cctv_"
        )

        remapped_lines = []

        for line_number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()

            try:
                old_class = int(parts[0])
            except (ValueError, IndexError):
                raise RuntimeError(
                    f"Invalid label: {label}, line {line_number}"
                )

            if old_class not in CLASS_MAP:
                raise RuntimeError(
                    f"Unexpected class {old_class} in {label}"
                )

            parts[0] = str(CLASS_MAP[old_class])
            remapped_lines.append(" ".join(parts))
            object_counts[CLASS_MAP[old_class]] += 1

        dest_label.write_text(
            "\n".join(remapped_lines) + ("\n" if remapped_lines else ""),
            encoding="utf-8"
        )

        cctv_counts[destination_split] += 1
        source_counts[source][destination_split] += 1


# 3. Write the combined dataset YAML
yaml_text = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: BikeWithRider
  1: NoHelmet
  2: Helmet
"""

(OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")


# 4. Print results
print("\n" + "=" * 55)
print("DATASET PREPARATION COMPLETE")
print("=" * 55)

for split in ["train", "val", "test"]:
    edge_n = edge_counts[split]
    cctv_n = cctv_counts[split]
    total = edge_n + cctv_n

    print(
        f"{split.upper():5} | EdgeVision: {edge_n:4} "
        f"| CCTV: {cctv_n:4} | TOTAL: {total:4}"
    )

print("\nCCTV IMAGES BY SOURCE")
for source, counts in sorted(source_counts.items()):
    print(
        f"{source} | train: {counts['train']} "
        f"| val: {counts['val']}"
    )

print("\nCCTV OBJECT COUNTS AFTER REMAPPING")
print(f"BikeWithRider (0): {object_counts[0]}")
print(f"NoHelmet      (1): {object_counts[1]}")
print(f"Helmet        (2): {object_counts[2]}")

print("\nEXPECTED TOTALS")
print("Train: 1971")
print("Val:    572")
print("Test:   237")
print("Total: 2780")

actual = {
    split: edge_counts[split] + cctv_counts[split]
    for split in ["train", "val", "test"]
}

expected = {"train": 1971, "val": 572, "test": 237}

if actual != expected:
    print("\nWARNING: Actual counts differ from expected counts.")
    print("Please share this output before training.")
else:
    print("\nAll image counts match the expected totals.")

print("\nOutput:", OUT)
