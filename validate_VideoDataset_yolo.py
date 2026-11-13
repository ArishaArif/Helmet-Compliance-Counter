
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\HelmetComplianceCounter\video_dataset_cleaned")

CLASS_NAMES = [
    "H0_N1", "H0_N2", "H0_N3", "H0_N4",
    "H1_N0", "H1_N1", "H1_N2",
    "H2_N0", "H2_N1"
]

total_boxes = 0
total_invalid = 0

for split in ["train", "val", "test"]:
    label_dir = ROOT / "labels" / split
    class_counts = Counter()
    invalid = []

    for file in label_dir.glob("*.txt"):
        with open(file, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                parts = line.split()

                if len(parts) != 5:
                    invalid.append((file.name, line_number, "Expected 5 values"))
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    invalid.append((file.name, line_number, "Non-numeric value"))
                    continue

                if not 0 <= class_id < len(CLASS_NAMES):
                    invalid.append((file.name, line_number, "Invalid class ID"))
                    continue

                if not (
                    0 <= x <= 1 and
                    0 <= y <= 1 and
                    0 < w <= 1 and
                    0 < h <= 1
                ):
                    invalid.append((file.name, line_number, "Invalid coordinates"))
                    continue

                class_counts[class_id] += 1
                total_boxes += 1

    total_invalid += len(invalid)

    print(f"\n{split.upper()} VALIDATION")
    print("-" * 45)
    print("Valid boxes:", sum(class_counts.values()))
    print("Invalid boxes:", len(invalid))

    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{name}: {class_counts[class_id]}")

    if invalid:
        print("\nFirst 10 invalid annotations:")
        for item in invalid[:10]:
            print(item)

print("\n" + "=" * 45)
print("TOTAL VALID BOXES:", total_boxes)
print("TOTAL INVALID BOXES:", total_invalid)
