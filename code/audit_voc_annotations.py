
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

root = Path(
    r"D:\HelmetComplianceCounter\data\EdgeVision Dataset\EdgeVision-Dataset"
)

xml_dir = root / "labels" / "voc_xml"

files_checked = 0
objects_checked = 0
issues = []
class_counts = Counter()

for xml_file in sorted(xml_dir.glob("*.xml")):
    files_checked += 1

    try:
        tree = ET.parse(xml_file)
        annotation = tree.getroot()
    except ET.ParseError as error:
        issues.append(f"{xml_file.name}: XML parsing error: {error}")
        continue

    size = annotation.find("size")

    if size is None:
        issues.append(f"{xml_file.name}: missing image size")
        continue

    try:
        image_width = int(size.findtext("width"))
        image_height = int(size.findtext("height"))
    except (TypeError, ValueError):
        issues.append(f"{xml_file.name}: invalid image dimensions")
        continue

    if image_width <= 0 or image_height <= 0:
        issues.append(f"{xml_file.name}: non-positive image dimensions")
        continue

    for index, obj in enumerate(annotation.findall("object"), start=1):
        objects_checked += 1
        class_name = obj.findtext("name", default="UNKNOWN")
        class_counts[class_name] += 1

        box = obj.find("bndbox")

        if box is None:
            issues.append(
                f"{xml_file.name}, object {index}: missing bounding box"
            )
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            issues.append(
                f"{xml_file.name}, object {index}: invalid box coordinates"
            )
            continue

        if xmax <= xmin or ymax <= ymin:
            issues.append(
                f"{xml_file.name}, object {index} ({class_name}): "
                f"zero or negative box size"
            )
            continue

        if (
            xmin < 0 or ymin < 0
            or xmax > image_width or ymax > image_height
        ):
            issues.append(
                f"{xml_file.name}, object {index} ({class_name}): "
                f"box extends outside image boundaries"
            )

print("XML files checked:", files_checked)
print("Objects checked:", objects_checked)
print("Classes found:")

for name, count in sorted(class_counts.items()):
    print(f"  {name}: {count}")

print("\nStructural issues found:", len(issues))

for issue in issues[:40]:
    print(" -", issue)

if len(issues) > 40:
    print(f"... and {len(issues) - 40} more issues.")

report_path = root / "voc_audit_report.txt"
report_path.write_text(
    "\n".join([
        f"XML files checked: {files_checked}",
        f"Objects checked: {objects_checked}",
        f"Structural issues: {len(issues)}",
        "",
        *issues,
    ]),
    encoding="utf-8",
)

print("\nFull issue report saved to:", report_path)
