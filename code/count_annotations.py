
from pathlib import Path
from collections import Counter

# Location of the YOLO label files
root = Path(r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset\labels\yolo")

counts = Counter()
label_files = 0

for file in root.glob("*.txt"):
    if file.name.lower() in ("classes.txt", "readme.txt"):
        continue

    label_files += 1

    for line in file.read_text().splitlines():
        parts = line.split()

        if parts and parts[0].isdigit():
            counts[int(parts[0])] += 1

print("Label files found:", label_files)
print("BikeWithRider (class 0):", counts[0])
print("NoHelmet (class 1):", counts[1])
print("Helmet (class 2):", counts[2])
print("Total annotated objects:", sum(counts.values()))
