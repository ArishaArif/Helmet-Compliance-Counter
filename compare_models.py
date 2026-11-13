
from pathlib import Path
import csv
import shutil
import sys

from ultralytics import YOLO

PROJECT = Path(r"D:\HelmetComplianceCounter")

ORIGINAL_MODEL = (
    PROJECT / "runs" / "helmet_yolov8n" / "weights" / "best.pt"
)

FINETUNED_MODEL = (
    PROJECT / "runs" / "detect" / "runs"
    / "helmet_yolov8n_cctv_finetune"
    / "weights" / "best.pt"
)

DATA_YAML = PROJECT / "fine_tune_dataset" / "data.yaml"
DATASET = PROJECT / "fine_tune_dataset"

COMPARISON_DATA = PROJECT / "comparison_data" / "cctv_holdout"
RESULTS_DIR = PROJECT / "runs" / "model_comparison"
CSV_PATH = RESULTS_DIR / "comparison_results.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODELS = {
    "Original": ORIGINAL_MODEL,
    "Fine-tuned": FINETUNED_MODEL,
}


def check_files():
    required = [ORIGINAL_MODEL, DATA_YAML]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    if not FINETUNED_MODEL.exists():
        raise FileNotFoundError(
            "Fine-tuned best.pt not found.\n"
            "Wait until training finishes before running this script."
        )


def prepare_cctv_holdout():
    source_images = DATASET / "images" / "val"
    source_labels = DATASET / "labels" / "val"

    images = sorted(
        p for p in source_images.iterdir()
        if p.is_file()
        and p.name.startswith("cctv_")
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) != 101:
        raise RuntimeError(
            f"Expected 101 held-out CCTV images, found {len(images)}."
        )

    image_out = COMPARISON_DATA / "images"
    label_out = COMPARISON_DATA / "labels"

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for image in images:
        label = source_labels / (image.stem + ".txt")

        if not label.exists():
            raise RuntimeError(f"Missing label: {label}")

        shutil.copy2(image, image_out / image.name)
        shutil.copy2(label, label_out / label.name)

    yaml_path = COMPARISON_DATA / "data.yaml"

    yaml_path.write_text(
        f"""path: {COMPARISON_DATA.as_posix()}
train: images
val: images
test: images

names:
  0: BikeWithRider
  1: NoHelmet
  2: Helmet
""",
        encoding="utf-8"
    )

    print(f"CCTV holdout prepared: {len(images)} images")
    return yaml_path


def evaluate_model(model_name, model_path, dataset_name,
                   data_yaml, split):

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
        project=str(RESULTS_DIR),
        name=f"{model_name.lower().replace('-', '_')}_{dataset_name}",
        exist_ok=True
    )

    box = metrics.box
    speed = metrics.speed

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "preprocess_ms": speed.get("preprocess", 0),
        "inference_ms": speed.get("inference", 0),
        "postprocess_ms": speed.get("postprocess", 0),
    }

    print(f"Precision:       {row['precision']:.4f}")
    print(f"Recall:          {row['recall']:.4f}")
    print(f"mAP50:           {row['mAP50']:.4f}")
    print(f"mAP50-95:        {row['mAP50_95']:.4f}")
    print(f"Inference/image: {row['inference_ms']:.2f} ms")

    return row


def main():
    check_files()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cctv_yaml = prepare_cctv_holdout()

    rows = []

    evaluations = [
        ("EdgeVision_Test", DATA_YAML, "test"),
        ("CCTV_Holdout", cctv_yaml, "val"),
    ]

    for model_name, model_path in MODELS.items():
        for dataset_name, yaml_path, split in evaluations:
            row = evaluate_model(
                model_name,
                model_path,
                dataset_name,
                yaml_path,
                split
            )
            rows.append(row)

    fields = [
        "model",
        "dataset",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results saved to:\n{CSV_PATH}")


if __name__ == "__main__":
    main()
