
from pathlib import Path
import re

BASE = Path(r"C:\Users\U.S.A Trader\Downloads\Helmet vs No Helmet.v3i.yolov11")

for split in ["train", "valid", "test"]:
    folder = BASE / split / "images"
    files = list(folder.glob("*"))
    sources = {}

    for f in files:
        source = re.sub(r"-\d+_jpg.*", "", f.name)
        sources[source] = sources.get(source, 0) + 1

    print(f"\n{split.upper()} ({len(files)} images)")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")
