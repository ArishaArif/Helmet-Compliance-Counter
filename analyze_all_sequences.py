
import csv
from pathlib import Path
from collections import defaultdict

annotation_dir = Path(r"D:\myanmar\annotation\annotation")
image_dir = Path(r"D:\myanmar\part_7\part_7")

class_map = {
    "DHelmet": "H1_N0",
    "DNoHelmet": "H0_N1",
    "DHelmetP1Helmet": "H2_N0",
    "DHelmetP1HelmetP2NoHelmet": "H2_N1",
    "DHelmetP1NoHelmet": "H1_N1",
    "DHelmetP1NoHelmetP2NoHelmet": "H1_N2",
    "DNoHelmetP0NoHelmet": "H0_N2",
    "DNoHelmetP1Helmet": "H1_N1",
    "DNoHelmetP1NoHelmet": "H0_N2",
    "DNoHelmetP1NoHelmetP2Helmet": "H1_N2",
    "DNoHelmetP1NoHelmetP2NoHelmet": "H0_N3",
    "DNoHelmetP0NoHelmetP1NoHelmet": "H0_N3",
    "DNoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": "H0_N4",
}

rare_classes = ["H0_N3", "H0_N4", "H1_N2", "H2_N0", "H2_N1"]

print("\nSEQUENCE | STATUS | ROWS | TRACKS | " + " | ".join(rare_classes))
print("-" * 115)

for file in sorted(annotation_dir.glob("*.csv")):
    counts = defaultdict(int)
    tracks = set()
    rows = 0

    with open(file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            label = row["label"]

            if label not in class_map:
                continue

            mapped = class_map[label]
            if mapped in rare_classes:
                counts[mapped] += 1

            tracks.add(row["track_id"])

    status = "DOWNLOADED" if (image_dir / file.stem).is_dir() else "MISSING"

    values = [str(counts[c]) for c in rare_classes]

    print(
        f"{file.stem} | {status} | {rows} | "
        f"{len(tracks)} | " + " | ".join(values)
    )
