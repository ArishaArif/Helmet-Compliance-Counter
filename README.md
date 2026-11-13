# Helmet Compliance Counter

A computer-vision application for detecting helmet compliance in motorcycle traffic images and videos. The system uses YOLOv8 for object detection, ByteTrack for video tracking, and Flask for its web interface and processing backend.

## Project Overview

Helmet Compliance Counter identifies two helmet-status classes:

- **Helmet** — helmet detected.
- **NoHelmet** — no-helmet detection.

The underlying models also contain a `BikeWithRider` class, but the application filters its displayed and counted results to Helmet and NoHelmet.

The application supports individual and multiple image uploads, video processing, annotated results, tracking IDs, compliance summaries, and downloadable reports.

## Technology Stack

- Python
- YOLOv8 / Ultralytics
- OpenCV
- ByteTrack
- Flask
- SQLite / PostgreSQL (Neon)
- HTML, CSS and JavaScript
- ReportLab for PDF exports

## Model Checkpoints

Two trained checkpoints are included in the `models/` directory:

| File | Purpose |
|---|---|
| `cctv_finetuned_best.pt` | Current deployment model, fine-tuned for CCTV footage |
| `edgevision_backup_best.pt` | Backup model trained on the EdgeVision dataset |

The application uses environment variables to select the model:

```env
IMAGE_MODEL=models/cctv_finetuned_best.pt
VIDEO_MODEL=models/cctv_finetuned_best.pt
```

The fine-tuned checkpoint is the current active model. The EdgeVision checkpoint is retained as a backup.

## Model Evaluation

Results below come from separate evaluation datasets and should not be compared as though they were the same test.

### EdgeVision held-out test set

Original EdgeVision model:

| Metric | Result |
|---|---:|
| Precision | 93.8% |
| Recall | 86.5% |
| mAP50 | 93.0% |
| mAP50-95 | 66.3% |

Fine-tuned model on the same EdgeVision test set:

| Metric | Result |
|---|---:|
| Precision | 94.7% |
| Recall | 82.3% |
| mAP50 | 90.6% |
| mAP50-95 | 62.5% |

### CCTV holdout evaluation

The fine-tuned model was evaluated on 101 held-out CCTV images at 640-pixel inference size.

| Metric | Result |
|---|---:|
| Precision | 61.4% |
| Recall | 61.7% |
| mAP50 | 69.5% |
| mAP50-95 | 35.4% |

Per-class CCTV mAP50:

| Class | mAP50 |
|---|---:|
| BikeWithRider | 94.7% |
| NoHelmet | 63.5% |
| Helmet | 50.2% |

CCTV performance has improved substantially compared with the original EdgeVision model, but helmet-status detection still requires improvement. These results are not a claim that the system has achieved 80–95% precision on CCTV footage.

For further details, see [the technical report](documentation/PROJECT_TECHNICAL_REPORT.md).

## Application Features

- Login system with administrator and analyst roles.
- Password hashing and login-attempt throttling.
- CSRF protection and security response headers.
- Upload of up to 20 JPEG/PNG images in one request.
- Single-video upload (MP4, AVI, MOV or MKV).
- Image and video inference using YOLOv8.
- ByteTrack tracking for video processing.
- Track-level majority-vote helmet classification.
- Live annotated video preview.
- Helmet and NoHelmet detection summaries.
- Detection history and per-job results.
- CSV, TXT, XML, JSON and PDF report exports.
- SQLite local database with optional PostgreSQL/Neon support.

## Project Structure

```text
HelmetComplianceCounter/
├── code/                  # Dataset audit and preparation scripts
├── documentation/         # Project
