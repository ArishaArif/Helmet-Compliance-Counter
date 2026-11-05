
from pathlib import Path
from collections import Counter

labels_dir = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset\labels\yolo"
)

# Count which classes appear together in each image.
combinations = Counter()

for file in labels_dir.glob("*.txt"):
    if file.name.lower() == "classes.txt":
        continue

    classes_in_image = set()

    for line in file.read_text().splitlines():
        parts = line.split()

        if parts and parts[0].isdigit():
            classes_in_image.add(int(parts[0]))

    if classes_in_image:
        combinations[tuple(sorted(classes_in_image))] += 1

class_names = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

print("Class combinations across images:\n")

for combination, count in combinations.most_common():
    names = [class_names[c] for c in combination]
    print(f"{' + '.join(names)}: {count} images")
