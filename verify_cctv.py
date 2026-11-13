
import cv2
from pathlib import Path

BASE = Path(r"C:\Users\U.S.A Trader\Downloads\Helmet vs No Helmet.v3i.yolov11")

image_path = BASE / "train/images/ANPR-cam-Shahra-E-Faisal-Aisha-Bawani_mp4-0002_jpg.rf.2051fefd4016807e08aef60072f98fad.jpg"
label_path = BASE / "train/labels/ANPR-cam-Shahra-E-Faisal-Aisha-Bawani_mp4-0002_jpg.rf.2051fefd4016807e08aef60072f98fad.txt"

names = {
    0: "bike",
    1: "helmet",
    2: "no helmet"
}

colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

height, width = image.shape[:2]

with open(label_path, "r") as file:
    for line in file:
        parts = line.split()
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        color = colors[class_id]
        name = names[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, name, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

output_path = BASE / "verified_sample.jpg"
cv2.imwrite(str(output_path), image)

print("Saved:", output_path)
