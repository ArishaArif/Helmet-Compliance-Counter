from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\HelmetComplianceCounter\cleaned_dataset")
LABEL_DIR = ROOT / "labels"

class_names = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

issues = Counter()
examples = []

for label_file in LABEL_DIR.glob("*.txt"):
    for line_number, line in enumerate(
        label_file.read_text().splitlines(), start=1
    ):
        parts = line.split()

        if len(parts) != 5:
            issues["Incorrect number of values"] += 1
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            issues["Non-numeric values"] += 1
            continue

        if class_id not in class_names:
            issues["Invalid class ID"] += 1
            continue

        if not all(0 <= value <= 1 for value in (x, y, w, h)):
            issues["Coordinates outside 0-1"] += 1
            if len(examples) < 10:
                examples.append(
                    f"{label_file.name}, line {line_number}: {line}"
                )

        if w <= 0 or h <= 0:
            issues["Non-positive width or height"] += 1

print("========== COORDINATE VALIDATION ==========")
print("Label files checked:", len(list(LABEL_DIR.glob("*.txt"))))

if not issues:
    print("No coordinate issues found!")
else:
    for issue, count in issues.items():
        print(f"{issue}: {count}")

if examples:
    print("\nFirst 10 problematic annotations:")
    for example in examples:
        print(example)