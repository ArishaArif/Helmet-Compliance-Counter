from pathlib import Path

LABEL_DIR = Path(
    r"D:\HelmetComplianceCounter\cleaned_dataset\labels"
)

results = []

for label_file in LABEL_DIR.glob("*.txt"):

    counts = {
        0: 0,  # BikeWithRider
        1: 0,  # NoHelmet
        2: 0   # Helmet
    }

    for line in label_file.read_text().splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if class_id in counts:
            counts[class_id] += 1

    results.append({
        "image": label_file.stem,
        "bikes": counts[0],
        "no_helmet": counts[1],
        "helmet": counts[2],
        "total": sum(counts.values())
    })

# Highest number of BikeWithRider annotations
print("\n========== MOST RIDERS ==========")

for item in sorted(
    results,
    key=lambda x: x["bikes"],
    reverse=True
)[:10]:
    print(
        f"{item['image']}: "
        f"{item['bikes']} riders"
    )

# Highest total annotations
print("\n========== MOST TOTAL ANNOTATIONS ==========")

for item in sorted(
    results,
    key=lambda x: x["total"],
    reverse=True
)[:10]:
    print(
        f"{item['image']}: "
        f"{item['total']} annotations "
        f"(Bikes: {item['bikes']}, "
        f"NoHelmet: {item['no_helmet']}, "
        f"Helmet: {item['helmet']})"
    )

# Highest NoHelmet and Helmet counts
for key, title in [
    ("no_helmet", "MOST NO-HELMET ANNOTATIONS"),
    ("helmet", "MOST HELMET ANNOTATIONS")
]:
    print(f"\n========== {title} ==========")

    for item in sorted(
        results,
        key=lambda x: x[key],
        reverse=True
    )[:5]:
        print(f"{item['image']}: {item[key]}")