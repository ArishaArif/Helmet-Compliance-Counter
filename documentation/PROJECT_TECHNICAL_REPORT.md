# Helmet Compliance Counter — Technical Project Report

**Project:** Helmet Compliance Counter  
**Repository:** https://github.com/ArishaArif/Helmet-Compliance-Counter  
**Framework:** YOLOv8n, Ultralytics, OpenCV, Flask and ByteTrack  
**Report scope:** Dataset preparation, annotation auditing, training, CCTV fine-tuning, evaluation and application implementation.

---

## 1. Project Objective

Helmet Compliance Counter is a computer-vision application intended to detect helmet compliance among motorcycle riders in traffic images and videos.

The detection model uses three classes:

| Class ID | Class | Meaning |
|---|---|---|
| 0 | BikeWithRider | Motorcycle with a rider |
| 1 | NoHelmet | No-helmet head/rider region |
| 2 | Helmet | Helmet region |

The application displays and counts Helmet and NoHelmet detections. The BikeWithRider class remains part of the trained model but is filtered from the application's helmet-status results.

The intended workflow is:

1. An analyst signs in.
2. The analyst uploads images or a traffic video.
3. YOLOv8 processes the input.
4. ByteTrack tracks detections across video frames.
5. Helmet-status detections are recorded.
6. The application presents summaries and annotated results.
7. The analyst exports a report.

A critical distinction is that a detection is not automatically a verified person identity. Tracking IDs can change, and helmet-status detections must be associated correctly before they can support reliable unique-rider statistics.

## 2. Original EdgeVision Dataset

**Dataset:** EdgeVision Dataset, V1  
**Source:** Mendeley Data  
**Licence stated by source:** CC BY 4.0  
**Images:** 2,392  
**Annotations:** 8,275  
**Formats:** YOLO, Pascal VOC XML and COCO  
**Training format selected:** YOLO

### 2.1 Original class distribution

| Class | Instances | Percentage |
|---|---:|---:|
| BikeWithRider | 3,793 | 45.8% |
| NoHelmet | 2,810 | 34.0% |
| Helmet | 1,672 | 20.2% |
| **Total** | **8,275** | **100%** |

These values represent annotated objects, not unique riders or the number of images containing each class.

Initial pairing checks found 2,392 images and 2,392 matching YOLO label files, with no missing pairs.

## 3. Dataset Quality Auditing

### 3.1 Invalid YOLO annotations

The label-validation process identified ten problematic annotations:

- `Image_00566.txt`: a Helmet annotation with non-positive height.
- `Image_00958`
- `Image_00960`
- `Image_01035`
- `Image_01037`
- `Image_01045`
- `Image_01059`
- `Image_01060`
- `Image_02080`
- `Image_02187`

The other nine files contained coordinates outside the normalized 0–1 range.

### 3.2 Pascal VOC audit

The XML audit examined 2,392 XML files and 8,275 objects.

It reported 79 structural flags:
- 78 boxes extending beyond image boundaries.
- One zero-size box in `Image_00566`.

The boundary cases were visually reviewed. Boxes extending beyond image boundaries can represent partially visible objects, so they were not automatically removed.

The zero-height Helmet annotation in `Image_00566` was removed in the cleaned copy.

### 3.3 EXIF orientation

The orientation audit identified 34 images with EXIF orientation values 3, 6 or 8.

`Image_01035` was investigated:
- Original dimensions: 4624 × 2604 pixels.
- EXIF orientation: 6.
- Displayed orientation: 90 degrees clockwise.

The image and annotation coordinates had to be interpreted using the original pixel orientation. Rotating image pixels without transforming the corresponding boxes could misalign annotations.

**Decision:** Exclude the 34 affected images and matching labels from the cleaned training dataset rather than risk inconsistent image/annotation transformations.

### 3.4 Boundary-overflow exceptions

Two remaining BikeWithRider annotations were visually reviewed:

| Image | Width |
|---|---:|
| Image_02080 | 1.002463 |
| Image_02187 | 1.001825 |

They were retained as visually reviewed boundary-overflow cases.

The cleaned dataset should therefore not be described as having every coordinate strictly within the normalized 0–1 range.

### 3.5 Crowded-image review

Crowded-image analysis used BikeWithRider annotations as a proxy for rider density.

Examples:
- `Image_00672`: 10 BikeWithRider boxes, 20 total annotations.
- `Image_00705`: 7 BikeWithRider boxes, 17 total annotations.
- `Image_00720`: 8 BikeWithRider boxes, 17 total annotations.

Total boxes are not equivalent to unique rider counts because one rider can have multiple associated annotations.

## 4. Dataset Cleaning

The original dataset was preserved. A separate `cleaned_dataset` was created.

The cleaning procedure:
1. Excluded the 34 images with EXIF orientation values 3, 6 or 8.
2. Removed the single zero-height annotation.
3. Retained visually reviewed boundary-overflow annotations.
4. Copied the retained images and labels.
5. Generated exclusion and removal reports.

### 4.1 Cleaning results

| Item | Result |
|---|---:|
| Original images | 2,392 |
| Orientation-affected images excluded | 34 |
| Retained images | 2,358 |
| Invalid annotations removed | 1 |
| Retained annotations | 8,152 |
| Image/label pairs | 2,358 |
| Missing image/label pairs | 0 |

Approximately 98.58% of the original images were retained.

### 4.2 Cleaned class distribution

| Class | Instances | Percentage |
|---|---:|---:|
| BikeWithRider | 3,735 | 45.82% |
| NoHelmet | 2,785 | 34.16% |
| Helmet | 1,632 | 20.02% |
| **Total** | **8,152** | **100%** |

Verification found no missing pairs, invalid class IDs or non-positive box dimensions. The two documented boundary-overflow exceptions remained.

Cleaning reports:
- `cleaned_dataset/exclusion_report.csv`
- `cleaned_dataset/removed_annotations.csv`

## 5. Original Model Training

**Model:** YOLOv8n  
**Framework:** Ultralytics 8.4.202  
**Training device:** CPU  
**Starting weights:** Pretrained YOLOv8n  
**Dataset:** Cleaned EdgeVision dataset

Recorded configuration:

| Setting | Value |
|---|---|
| Epochs | 50 |
| Image size | 416 |
| Batch size | 4 |
| Device | CPU |
| Workers | 0 |
| Early-stopping patience | 10 |
| Training script | `code/train.py` |
| Run directory | `runs/helmet_yolov8n` |

Training was interrupted by a laptop shutdown. The run contained `last.pt` and `best.pt`. Training was resumed using the saved `last.pt` checkpoint and Ultralytics resume functionality. Output confirmed continuation at epoch 22 of 50.

## 6. Original EdgeVision Model Evaluation

### 6.1 Validation results

The training validation output covered 471 images and 1,620 instances.

| Metric | Result |
|---|---:|
| Precision | 91.2% |
| Recall | 88.1% |
| mAP50 | 92.2% |
| mAP50–95 | 67.0% |

Per-class validation results:

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| BikeWithRider | 92.4% | 94.2% | 96.7% | 83.6% |
| NoHelmet | 88.6% | 82.9% | 88.5% | 55.6% |
| Helmet | 92.6% | 87.1% | 91.3% | 61.9% |

### 6.2 Independent test results

The held-out EdgeVision test split contained:
- 237 images.
- 850 instances.

| Metric | Result |
|---|---:|
| Precision | 93.8% |
| Recall | 86.5% |
| mAP50 | 93.0% |
| mAP50–95 | 66.3% |

These are EdgeVision test results, not CCTV results.

## 7. CCTV Dataset Preparation and Fine-Tuning

The project subsequently incorporated CCTV-oriented data to improve performance on distant traffic footage.

The additional Roboflow dataset investigated contained 761 images and the classes helmet, license_plate and motorcyclist.

The combined fine-tuning dataset used the following class mapping:

| CCTV class | EdgeVision class |
|---|---|
| Bike | BikeWithRider |
| No helmet | NoHelmet |
| Helmet | Helmet |

### 7.1 Fine-tuning dataset composition

| Split | EdgeVision | CCTV | Total |
|---|---:|---:|---:|
| Train | 1,650 | 321 | 1,971 |
| Validation | 471 | 101 | 572 |
| Test | 237 | 0 | 237 |
| **Total** | **2,358** | **422** | **2,780** |

The 101 CCTV validation images were held out from training. They came from a distinct source video within the same collection, so they are a useful holdout but not a fully independent external dataset.

The EdgeVision test split remained separate.

### 7.2 Fine-tuning configuration

The fine-tuned model started from the original EdgeVision-trained YOLOv8n checkpoint.

Recorded command:

```cmd
yolo detect train model=runs\helmet_yolov8n\weights\best.pt data=fine_tune_dataset\data.yaml epochs=15 imgsz=320 batch=4 device=cpu workers=0 patience=5 project=runs name=helmet_yolov8n_cctv_finetune
```

Training stopped early after six epochs, approximately 1.188 hours, with the best checkpoint recorded at epoch 1.

The resulting checkpoint is:

`runs/detect/runs/helmet_yolov8n_cctv_finetune/weights/best.pt`

## 8. Model Comparison

### 8.1 EdgeVision test comparison

| Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Original | 92.81% | 82.16% | 90.15% | 63.87% |
| Fine-tuned | 94.74% | 82.33% | 90.56% | 62.55% |

Fine-tuning preserved broadly similar EdgeVision performance. Precision and mAP50 increased slightly, while mAP50–95 decreased slightly.

### 8.2 CCTV holdout comparison at 320

| Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Original | 4.05% | 5.95% | 0.55% | 0.07% |
| Fine-tuned | 48.01% | 59.16% | 53.41% | 28.21% |

The fine-tuned model substantially improved performance on the CCTV holdout compared with the original EdgeVision model.

### 8.3 Fine-tuned CCTV evaluation at 640

On the same 101 CCTV images, at 640-pixel inference size:

| Metric | Result |
|---|---:|
| Precision | 61.4% |
| Recall | 61.7% |
| mAP50 | 69.5% |
| mAP50–95 | 35.4% |

Per-class results:

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| BikeWithRider | 76.7% | 93.1% | 94.7% | 64.9% |
| NoHelmet | 46.1% | 57.7% | 63.5% | 22.4% |
| Helmet | 61.5% | 34.4% | 50.2% | 19.0% |

The 640-pixel evaluation improved overall CCTV mAP50 from 53.4% to 69.5% compared with the 320-pixel evaluation.

However, Helmet recall remained low at 34.4%. Helmet-status detection is still a significant limitation.

### 8.4 Original CCTV model at 640

| Metric | Result |
|---|---:|
| Precision | 4.24% |
| Recall | 16.4% |
| mAP50 | 1.44% |
| mAP50–95 | 0.19% |

The fine-tuned model substantially outperformed the original model on the CCTV holdout at this resolution.

**Interpretation:** The fine-tuned checkpoint is the more appropriate current model for CCTV deployment. It is not yet sufficiently reliable to claim production-grade helmet-compliance detection.

## 9. Small-object Investigation

CCTV helmet and no-helmet regions are substantially smaller than motorcycle/rider boxes.

At 320-pixel inference size, the ground-truth object-size analysis found:

| Class | Objects | Average width | Average height |
|---|---:|---:|---:|
| BikeWithRider | 173 | 17.1 px | 45.5 px |
| NoHelmet | 26 | 5.4 px | 12.8 px |
| Helmet | 135 | 6.5 px | 14.3 px |

Most Helmet and NoHelmet objects occupied very few pixels. This reduces the visual information available to the detector.

Increasing inference size to 640 improved evaluation results, but increased computational cost substantially.

## 10. Application Implementation

The Flask application is implemented in `app.py`.

### 10.1 Authentication and access control

Implemented functionality includes:
- User login and logout.
- Administrator and analyst roles.
- Password hashing using Werkzeug.
- Administrator-only user creation.
- Login-attempt throttling.
- Session-based access control.
- CSRF validation for POST requests.

### 10.2 Upload validation

The application supports:
- Up to 20 images per image-upload request.
- JPEG and PNG images.
- A 10 MB maximum per image.
- A 40-million-pixel image limit.
- One video per video-upload request.
- MP4, AVI, MOV and MKV extensions.
- Video container and OpenCV readability checks.
- A 600 MB Flask request-size limit.

Images are decoded and re-encoded for processing, dropping their original metadata.

### 10.3 Detection and tracking

Image inference uses YOLOv8 prediction.

Video inference uses:

```python
model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    conf=CONF,
    imgsz=VIDEO_IMGSZ,
    verbose=False
)
```

Only Helmet and NoHelmet detections are extracted for the application's recorded results.

Video tracks are confirmed after a minimum of three detections. A majority vote across a track's observed labels determines its final helmet-status label.

The application saves representative crops and annotated previews.

### 10.4 Dashboard and reporting

Implemented backend functionality includes:
- Image and video jobs.
- Job progress and status.
- Live annotated video-frame streaming.
- Current tracked detection summaries.
- Detection history.
- Per-job access checks.
- CSV, TXT, XML, JSON and PDF exports.
- Input file SHA-256 hashes.

### 10.5 Database

The application supports:
- SQLite as the local fallback.
- PostgreSQL when `DATABASE_URL` is configured.
- User, model-run and detection tables.
- Parameterized SQL queries.

The PostgreSQL configuration uses Psycopg and its connection pool. The deployment environment must include all required PostgreSQL packages.

## 11. Active and Backup Models

Two trained checkpoints are retained:

| Repository path | Purpose |
|---|---|
| `models/cctv_finetuned_best.pt` | Active deployment checkpoint |
| `models/edgevision_backup_best.pt` | Backup EdgeVision checkpoint |

The active deployment configuration currently selects the fine-tuned checkpoint for both image and video inference:

```env
IMAGE_MODEL=models/cctv_finetuned_best.pt
VIDEO_MODEL=models/cctv_finetuned_best.pt
```

The backup is retained so the original EdgeVision model can be restored or compared without retraining.

The model paths must be checked against the deployment platform's working directory and persistent-file configuration.

## 12. Known Issues and Limitations

### Detection quality

- CCTV Helmet recall remains low.
- NoHelmet precision and recall remain below the desired target.
- Small distant helmet regions are difficult to detect.
- The model's performance differs substantially between EdgeVision-style images and CCTV footage.

### Tracking and counting

- ByteTrack IDs can change when a rider disappears or is occluded.
- A track ID does not prove a persistent real-world identity.
- The current implementation counts confirmed tracks, not necessarily unique riders across the entire recording.
- Association of helmet-status detections with the correct motorcycle/rider needs further validation.

### Video performance

- Inference and tracking overhead can reduce processing speed.
- The current pipeline does not guarantee playback at the source video's original frame rate.
- Larger inference sizes improve some detection metrics but increase CPU processing time.
- Live preview updates are sampled from processed frames.

### Deployment

- Model paths must be configured correctly.
- The PostgreSQL connection pool dependency must be installed when cloud PostgreSQL is used.
- Uploaded files and generated media require suitable persistent storage.
- Production hosting must support the application's background video-processing workload.
- Secret keys and database credentials must be configured outside the Git repository.

## 13. Reporting and Evaluation Rules

The following distinctions must be preserved in future reports:

1. Precision, recall, mAP50 and mAP50–95 are different metrics.
2. mAP is not ordinary classification accuracy.
3. Validation results must not be presented as independent test results.
4. EdgeVision and CCTV evaluations must be labelled separately.
5. Results at 320 and 640 pixels must identify their inference sizes.
6. Tracking IDs must not automatically be treated as unique people.
7. Compliance percentages are only as reliable as the underlying detections and rider associations.

## 14. Current Project Status

### Implemented

- EdgeVision dataset audit and cleaning.
- YOLOv8n training and independent test evaluation.
- CCTV dataset preparation and class mapping.
- CCTV fine-tuning and model comparison.
- Flask authentication and access control.
- Image and video upload processing.
- YOLOv8 inference and ByteTrack integration.
- Track-level classification and detection summaries.
- Live annotated previews.
- Multiple report export formats.
- SQLite and PostgreSQL code paths.

### Remaining work

- Improve Helmet and NoHelmet CCTV detection.
- Reduce duplicate detections and tracking-ID changes.
- Improve the relationship between helmet-status boxes and tracked riders.
- Improve video playback timing and processing speed.
- Verify the complete deployment configuration.
- Test cloud database connectivity and persistent media storage.
- Validate the complete application on representative traffic footage.

## 15. Final Assessment

The project progressed from an audited EdgeVision object-detection dataset to a Flask application with image/video processing, tracking, summaries and report exports.

The original model performed strongly on the EdgeVision test split. Fine-tuning substantially improved CCTV performance, and inference at 640 pixels improved the CCTV holdout results further.

The fine-tuned model is therefore selected as the current deployment model, with the EdgeVision checkpoint retained as a backup.

The system is a working development-stage prototype. Reliable helmet-status detection in difficult CCTV scenes, robust unique-rider counting and verified production deployment remain unfinished objectives.