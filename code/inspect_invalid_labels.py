
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

root = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset"
)

images_dir = root / "images"
labels_dir = root / "labels" / "yolo"
output_dir = root / "inspection_invalid"
output_dir.mkdir(exist_ok=True)

files_to_inspect = [
    "Image_00566.txt",
    "Image_00958.txt",
    "Image_00960.txt",
    "Image_01035.txt",
    "Image_01037.txt",
    "Image_01045.txt",
    "Image_01059.txt",
    "Image_01060.txt",
    "Image_02080.txt",
    "Image_02187.txt",
]

CLASS_INFO = {
    0: ("BikeWithRider", "blue"),
    1: ("NoHelmet", "red"),
    2: ("Helmet", "green"),
}

for filename in files_to_inspect:

    label_file = labels_dir / filename
    image_path = None

    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / (label_file.stem + extension)

        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print("Image not found:", filename)
        continue

    # Read original image dimensions and EXIF orientation.
    original = Image.open(image_path)
    original_width, original_height = original.size
    orientation = original.getexif().get(274, 1)

    # Apply EXIF orientation to the image pixels.
    image = ImageOps.exif_transpose(original).convert("RGB")

    corrected_width, corrected_height = image.size
    draw = ImageDraw.Draw(image)

    print(f"\nProcessing: {filename}")
    print("EXIF orientation:", orientation)
    print("Original dimensions:", original_width, original_height)
    print("Corrected dimensions:", corrected_width, corrected_height)

    for line_number, line in enumerate(
        label_file.read_text().splitlines(), start=1
    ):

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue

        if class_id not in CLASS_INFO:
            continue

        class_name, color = CLASS_INFO[class_id]

        # Convert normalized YOLO coordinates
        # into original pixel coordinates.
        x_center = x * original_width
        y_center = y * original_height
        box_width = w * original_width
        box_height = h * original_height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        # Transform bounding-box coordinates to match
        # the EXIF-corrected image orientation.
        if orientation == 3:
            x1, x2 = original_width - x2, original_width - x1
            y1, y2 = original_height - y2, original_height - y1

        elif orientation == 6:
            x1, y1, x2, y2 = (
                original_height - y2,
                x1,
                original_height - y1,
                x2
            )

        elif orientation == 8:
            x1, y1, x2, y2 = (
                y1,
                original_width - x2,
                y2,
                original_width - x1
            )

        else:
            # No coordinate transformation for orientation 1.
            # Other mirrored orientations are not transformed here.
            if orientation not in (1, None):
                print(
                    f"Warning: Orientation {orientation} "
                    f"not supported for {filename}"
                )

        # Draw the transformed bounding box.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=4
        )

        # Add matching class-name text.
        text = f"{class_name} (line {line_number})"

        text_x = max(0, min(x1, corrected_width - 1))
        text_y = max(0, min(y1, corrected_height - 1))

        draw.text(
            (text_x, text_y),
            text,
            fill=color,
            stroke_width=1,
            stroke_fill="black"
        )

    output_path = output_dir / f"{label_file.stem}_review.jpg"

    image.save(output_path)

    print("Saved:", output_path)

print("\nInspection complete.")
print("Review folder:", output_dir)
