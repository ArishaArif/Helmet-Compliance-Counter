
from pathlib import Path

labels_dir = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset\labels\yolo"
)

valid_classes = {0, 1, 2}
total_annotations = 0
invalid_lines = []

for file in labels_dir.glob("*.txt"):
    if file.name.lower() == "classes.txt":
        continue

    for line_number, line in enumerate(
        file.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        total_annotations += 1

        if len(parts) != 5:
            invalid_lines.append(
                f"{file.name}, line {line_number}: expected 5 values"
            )
            continue

        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            invalid_lines.append(
                f"{file.name}, line {line_number}: non-numeric value"
            )
            continue

        if class_id not in valid_classes:
            invalid_lines.append(
                f"{file.name}, line {line_number}: invalid class ID"
            )
        elif not all(0 <= value <= 1 for value in (x, y, width, height)):
            invalid_lines.append(
                 f"{file.name}, line {line_number}: "
                 f"class={class_id}, x={x}, y={y}, "
                 f"width={width}, height={height}"
            )
        elif width <= 0 or height <= 0:
            invalid_lines.append(
                f"{file.name}, line {line_number}: non-positive box size"
            )

print("Total annotations checked:", total_annotations)
print("Invalid annotations:", len(invalid_lines))

for problem in invalid_lines[:20]:
    print(problem)

if len(invalid_lines) > 20:
    print("Only the first 20 problems are shown.")
