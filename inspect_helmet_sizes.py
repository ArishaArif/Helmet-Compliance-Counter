
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"D:\HelmetComplianceCounter\cctv_evaluation")
IMAGES = ROOT / "images" / "val"
LABELS = ROOT / "labels" / "val"

classes = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

sizes = {0: [], 1: [], 2: []}

for label_file in LABELS.glob("*.txt"):
    image_file = IMAGES / (label_file.stem + ".jpg")

    if not image_file.exists():
        image_file = IMAGES / (label_file.stem + ".png")

    if not image_file.exists():
        continue

    image = cv2.imread(str(image_file))
    if image is None:
        continue

    h, w = image.shape[:2]
    scale = min(320 / w, 320 / h)

    with open(label_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            box_w = float(parts[3]) * w * scale
            box_h = float(parts[4]) * h * scale

            if cls in sizes:
                sizes[cls].append((box_w, box_h))

for cls, name in classes.items():
    boxes = sizes[cls]

    print(f"\n{name}: {len(boxes)} objects")

    if not boxes:
        continue

    areas = [w * h for w, h in boxes]

    print(f"Average width:  {np.mean([x[0] for x in boxes]):.1f} px")
    print(f"Average height: {np.mean([x[1] for x in boxes]):.1f} px")
    print(f"Median area:    {np.median(areas):.1f} px")

    for threshold in [8, 12, 16, 24]:
        count = sum(
            1 for w, h in boxes
            if w < threshold or h < threshold
        )
        print(f"Smaller than {threshold}px in either dimension: "
              f"{count}/{len(boxes)}")
