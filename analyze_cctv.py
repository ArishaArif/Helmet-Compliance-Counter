
from pathlib import Path
from collections import Counter

BASE = Path(r"C:\Users\U.S.A Trader\Downloads\Helmet vs No Helmet.v3i.yolov11")

splits = ["train", "valid", "test"]
counts = Counter()
images_with_class = Counter()
empty_files = 0

for split in splits:
    label_dir = BASE / split / "labels"
    split_counts = Counter()

    for file in label_dir.glob("*.txt"):
        seen = set()

        for line in file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue

            class_id = int(parts[0])
            counts[class_id] += 1
            split_counts[class_id] += 1
            seen.add(class_id)

        if not seen:
            empty_files += 1

        for class_id in seen:
            images_with_class[class_id] += 1

    print(f"\n{split.upper()}:")
    for class_id in range(3):
        print(f"  {class_id}: {split_counts[class_id]} objects")

print("\nTOTAL OBJECTS:")
for class_id, name in {
    0: "bike",
    1: "helmet",
    2: "no helmet"
}.items():
    print(f"{name}: {counts[class_id]} objects")
    print(f"  Images containing class: {images_with_class[class_id]}")

print("\nEmpty annotation files:", empty_files)
