
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\HelmetComplianceCounter\myanmar_video_dataset")

splits = ["train", "val", "test"]
total_images = 0
total_labels = 0

print("MYANMAR VIDEO DATASET VERIFICATION")
print("=" * 55)

for split in splits:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    total_images += len(images)
    total_labels += len(labels)

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    if missing_labels:
        print("Examples:", list(missing_labels)[:5])

    if missing_images:
        print("Examples:", list(missing_images)[:5])

print("\n" + "=" * 55)
print("TOTAL IMAGES:", total_images)
print("TOTAL LABEL FILES:", total_labels)

yaml_path = ROOT / "data.yaml"
print("data.yaml exists:", yaml_path.exists())

if yaml_path.exists():
    print("\nDATA.YAML CONTENT")
    print(yaml_path.read_text(encoding="utf-8"))
