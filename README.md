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
├── documentation/         # Project records and technical report
├── models/                # Active and backup model checkpoints
├── static/                # Static frontend assets
├── templates/             # Flask HTML templates
├── app.py                 # Flask application
├── requirements.txt
├── .gitignore
└── README.md
```

Datasets, uploaded files, generated media, virtual environments and training runs are not included in the source repository.

## Local Setup

### 1. Clone the repository

```cmd
git clone https://github.com/ArishaArif/Helmet-Compliance-Counter.git
cd Helmet-Compliance-Counter
```

### 2. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
SECRET_KEY=replace_with_a_long_random_secret
IMAGE_MODEL=models/cctv_finetuned_best.pt
VIDEO_MODEL=models/cctv_finetuned_best.pt
```

For PostgreSQL, also configure:

```env
DATABASE_URL=your_postgresql_connection_string
```

When `DATABASE_URL` is not configured, the application uses a local SQLite database.

Never commit `.env` or publish database credentials.

### 5. Create an administrator account

```cmd
python app.py create-admin admin@example.com
```

The application prompts for the administrator password.

### 6. Start the application

```cmd
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Important Limitations

- CCTV helmet and no-helmet detection is not yet consistently reliable.
- ByteTrack IDs are temporary tracking identities and do not guarantee persistent real-world identities.
- A track-level count is not necessarily a count of unique people across an entire video.
- Video processing speed depends on hardware, resolution, inference size and tracking overhead.
- The current processing pipeline does not guarantee real-time playback at the source video's original frame rate.
- Compliance percentages are calculated from the detections recorded by the application; they should not be interpreted as independently verified traffic statistics.
- Production deployment still requires verification of model paths, database connectivity, persistent file storage, worker configuration and resource limits.

## Documentation

See `documentation/PROJECT_TECHNICAL_REPORT.md` for dataset auditing, cleaning decisions, model training, evaluation comparisons, technical decisions and unresolved issues.

## Project Status

The Flask application, dataset preparation pipeline, model training and initial evaluations are implemented. CCTV fine-tuning has improved performance on the CCTV holdout set, but reliable helmet-status detection, stable tracking, video speed and production deployment remain areas for further work.
