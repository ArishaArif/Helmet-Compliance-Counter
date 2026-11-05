# Helmet Compliance Counter

## Project Work Log and Technical Decisions

**Document purpose:** Record the dataset preparation, decisions, model
training, evaluation status, and next steps completed for the Helmet
Compliance Counter project.

**Workspace:** `D:\HelmetComplianceCounter`\
**Current model:** YOLOv8n\
**Last updated:** 5 November 2026

------------------------------------------------------------------------

## 1. Project overview

Helmet Compliance Counter is a computer-vision project intended to
detect motorcycles/riders and classify helmet compliance in traffic
footage.

The current object-detection classes are:

  Class ID   Class name      Meaning
  ---------- --------------- ------------------------------------------
  0          BikeWithRider   Motorcycle with a rider
  1          NoHelmet        Rider/head region annotated as no helmet
  2          Helmet          Helmet region

The planned demonstration includes video inference, bounding boxes,
confidence scores, rider tracking IDs, helmet-status summaries, and a
compact dashboard.

The current detector does **not** by itself provide persistent rider
identities. Tracking and associating helmet-status detections with the
correct rider are separate implementation tasks.

## 2. Original dataset

**Dataset:** EdgeVision Dataset\
**Original location:** `data\EdgeVision Dataset\EdgeVision-Dataset`\
**Original images:** 2,392 JPEG images\
**Original annotations:** 8,275 objects\
**Annotation formats available:** YOLO, Pascal VOC (XML), and COCO\
**Selected format for YOLO training:** YOLO\
**Licence stated in the dataset README:** CC BY 4.0\
**Average image resolution stated in the README:** 1154 × 1411 (width ×
height)

### Original class distribution

  Class             Object annotations   Percentage
  --------------- -------------------- ------------
  BikeWithRider                  3,793        45.8%
  NoHelmet                       2,810        34.0%
  Helmet                         1,672        20.2%
  **Total**                  **8,275**     **100%**

These figures count annotated objects, not unique riders and not images.
A single rider can have both a motorcycle/rider box and a helmet-status
box.

### Initial structural check

The initial dataset check found: - 2,392 images. - 2,392 corresponding
YOLO label files. - No missing image/label pairs.

Further checks were then performed for invalid boxes, image orientation,
coordinate ranges, and crowded images.

## 3. Dataset quality investigation

### 3.1 Invalid YOLO annotations

A label-validation script identified 10 problematic annotations across
the original dataset.

-   `Image_00566.txt`, line 7: a `Helmet` box had a non-positive height
    (zero height).
-   The other nine flagged files were `Image_00958`, `Image_00960`,
    `Image_01035`, `Image_01037`, `Image_01045`, `Image_01059`,
    `Image_01060`, `Image_02080`, and `Image_02187`; these included
    coordinates outside the normalized 0--1 range.

The corresponding Pascal VOC XML annotations were also audited. The XML
audit checked 2,392 files and 8,275 objects and reported 79 structural
flags: - 78 boxes extended beyond image boundaries. - One zero-size box
was found in `Image_00566`.

The boundary-overflow boxes were visually reviewed. They often
represented objects partly outside the image frame, so they were not
automatically deleted. Removing them would have discarded visible object
information.

The zero-height `Helmet` annotation in `Image_00566` was invalid and was
removed during cleaning.

### 3.2 EXIF image orientation

An orientation audit identified 34 images with EXIF orientation values
3, 6, or 8, meaning the image display orientation depended on EXIF
rotation metadata.

The team investigated the image orientation and annotation alignment
rather than assuming that the raw pixel dimensions represented the
displayed image orientation. `Image_01035` was examined in detail: -
Original pixel dimensions: 4624 × 2604. - EXIF orientation: 6 (displayed
90 degrees clockwise). - The YOLO and Pascal VOC coordinates matched
when interpreted against the original image dimensions.

This showed that simply changing the image orientation without applying
the corresponding transformation to its bounding boxes could misalign
the labels.

**Decision:** Exclude all 34 affected images from the cleaned training
dataset rather than risk applying an incorrect image-only rotation or a
mismatched box transformation. The 34 images represented a small portion
of the original dataset (about 1.42%), so exclusion was considered safer
for this project stage.

This was a data-quality decision, not a claim that the images themselves
were unusable. The original dataset was preserved.

### 3.3 Two remaining coordinate overflows

After cleaning, a normalized-coordinate check found two annotations
whose `BikeWithRider` width was slightly greater than 1:

-   `Image_02080.txt`: width `1.002463`
-   `Image_02187.txt`: width `1.001825`

Both images were visually inspected. The boxes were considered
legitimate boundary-overflow cases and were retained in the cleaned
source dataset rather than deleting the objects.

These two annotations should remain documented as boundary-overflow
exceptions. They should not be described as normalized coordinates fully
within the 0--1 range.

### 3.4 Crowded-image review

A crowded-image analysis was run to identify images containing many
objects and potentially difficult scenes. The analysis used
`BikeWithRider` annotations as the closest available proxy for rider
count.

Examples: - `Image_00672`: 10 BikeWithRider annotations and 20 total
annotations. - `Image_00705`: 7 BikeWithRider annotations and 17 total
annotations. - `Image_00720`: 8 BikeWithRider annotations and 17 total
annotations.

Total annotation counts can include multiple boxes for the same rider,
so they must not be interpreted as unique rider counts.

## 4. Cleaning procedure and outcome

The cleaning script created a separate `cleaned_dataset` folder. The
original dataset was not modified.

The procedure: 1. Excluded images with EXIF orientation values 3, 6, or
8, together with their matching YOLO labels. 2. Removed the single
annotation with non-positive box dimensions in `Image_00566`. 3.
Retained other annotations, including boxes extending beyond image
boundaries. 4. Copied the retained images and labels into the cleaned
dataset. 5. Generated reports listing excluded images and removed
annotations.

### Cleaning results

  Item                                     Result
  -------------------------------------- --------
  Original images                           2,392
  Rotated-orientation images excluded          34
  Images retained                           2,358
  Invalid annotations removed                   1
  Annotations retained                      8,152
  Image/label pairs in cleaned dataset      2,358
  Missing image/label pairs                     0

The cleaned dataset retained approximately **98.58%** of the original
images.

### Cleaned class distribution

  Class             Object annotations   Percentage
  --------------- -------------------- ------------
  BikeWithRider                  3,735       45.82%
  NoHelmet                       2,785       34.16%
  Helmet                         1,632       20.02%
  **Total**                  **8,152**     **100%**

### Verification

The cleaned-dataset verification script reported: - 2,358 images and
2,358 label files. - No missing image/label pairs. - 8,152
annotations. - No annotations with invalid class IDs or non-positive box
dimensions. - Class counts matching the table above.

A separate coordinate-range check identified the two boundary-overflow
annotations described in Section 3.3. Thus, "zero invalid annotations"
refers to the structural checks performed by the verification script; it
does not mean every normalized coordinate was within 0--1.

### Cleaning reports

The cleaning script generated: -
`cleaned_dataset\exclusion_report.csv` -
`cleaned_dataset\removed_annotations.csv`

Other audit scripts and review outputs were retained in the project
workspace. Their evidence files should be organized after the main model
and demonstration workflow is complete.

## 5. Dataset split

A training/validation/test directory structure has been created at:

`D:\HelmetComplianceCounter\dataset_split`

It contains: - `images\train`, `images\val`, `images\test` -
`labels\train`, `labels\val`, `labels\test` - `data.yaml`

The YAML defines the dataset path and the three class names:

``` yaml
path: D:/HelmetComplianceCounter/dataset_split
train: images/train
val: images/val
test: images/test

names:
  0: BikeWithRider
  1: NoHelmet
  2: Helmet
```

**Important:** The exact image counts and class distributions in each
split have not yet been recorded in this document. They should be
measured from the existing split folders rather than assumed.

The test split must remain separate from model fitting and
model-selection decisions. It is intended for the final independent
evaluation.

## 6. Training environment and model

**Framework:** Ultralytics\
**Ultralytics version:** 8.4.202\
**Model family:** YOLOv8\
**Model size:** YOLOv8n (Nano)\
**Training device:** CPU\
**Computer memory:** approximately 7.5 GB usable physical RAM\
**NVIDIA GPU:** Not available

The pretrained `yolov8n.pt` weights were used as the starting point
rather than training the model from scratch.

Recorded training configuration:

  Setting                   Value
  ------------------------- ---------------------------
  Dataset config            `dataset_split\data.yaml`
  Epochs                    50
  Image size                416
  Batch size                4
  Device                    CPU
  Workers                   0
  Pretrained weights        Yes
  Early-stopping patience   10
  Plots                     Enabled
  Run name                  `helmet_yolov8n`

The training script is `code\train.py`. The run output is under:

`runs\helmet_yolov8n`

### Interrupted training and resumption

Training was interrupted when the laptop was shut down. The run
directory contained: - `runs\helmet_yolov8n\weights\last.pt` -
`runs\helmet_yolov8n\weights\best.pt`

Training was resumed from `last.pt` using Ultralytics' `resume=True`
option. The subsequent output showed epoch 22 of 50, confirming that the
run continued from a saved checkpoint rather than restarting at epoch 1.

## 7. Validation results after training

The completed training output reported validation on 471 images
containing 1,620 annotated objects.

### Overall metrics

  Metric               Result
  ----------- ---------------
  Precision     0.912 (91.2%)
  Recall        0.881 (88.1%)
  mAP50         0.922 (92.2%)
  mAP50-95      0.670 (67.0%)

### Per-class metrics

  Class             Images   Instances   Precision   Recall   mAP50   mAP50-95
  --------------- -------- ----------- ----------- -------- ------- ----------
  BikeWithRider        467         747       0.924    0.942   0.967      0.836
  NoHelmet             319         555       0.886    0.829   0.885      0.556
  Helmet               263         318       0.926    0.871   0.913      0.619

### Initial interpretation

-   `BikeWithRider` has the strongest results, including 96.7% mAP50 and
    83.6% mAP50-95.
-   `NoHelmet` has the lowest recall (82.9%), so missed no-helmet
    detections deserve attention.
-   `NoHelmet` and `Helmet` have lower mAP50-95 than `BikeWithRider`,
    suggesting that helmet-related localization and detection are areas
    to investigate.
-   These metrics are validation results from the training run. They are
    not a substitute for the final test-set evaluation.
-   Detection confidence scores shown during video inference are
    per-detection model confidence; they are not the same as
    dataset-level precision or accuracy.

Reported speed: 1.3 ms preprocessing, 47.8 ms inference, and 0.8 ms
postprocessing per image in the reported validation environment. Actual
video speed can differ with video resolution, tracking, drawing, and
dashboard processing.

## 8. Current project status

### Completed

-   Original dataset inspected and class schema established.
-   Image/label pairing checked.
-   YOLO and Pascal VOC annotations audited.
-   EXIF orientation issue investigated.
-   Invalid zero-height annotation removed in a separate cleaned copy.
-   Boundary-overflow annotations visually reviewed and retained where
    appropriate.
-   Cleaned dataset created and verified.
-   Train/validation/test folder structure and `data.yaml` created.
-   YOLOv8n trained, including resumption from a saved checkpoint.
-   Initial validation metrics obtained.

### Still to do

1.  Record exact train, validation, and test image/annotation counts.
2.  Evaluate `best.pt` on the held-out test split.
3.  Inspect validation/test plots, confusion matrix, and representative
    false positives and false negatives.
4.  Test the model on actual traffic video.
5.  Implement rider tracking and stable IDs.
6.  Associate helmet/no-helmet detections with the correct tracked
    rider.
7.  Implement unique counts and the compliance-rate calculation.
8.  Build the dashboard/frontend.
9.  Test the complete demonstration and document limitations.

## 9. Planned demonstration functionality

The intended dashboard is an original design inspired by common
traffic-monitoring interfaces, not a direct copy of the reference image.

Planned elements: - A thin top header titled **Helmet Compliance
Counter**. - Model/technology information (YOLOv8 and ByteTrack, once
tracking is implemented). - Live/processing status and elapsed video
time. - Main video area with detection boxes, rider IDs, helmet status,
and confidence. - A compact side panel showing crops from the existing
Helmet/NoHelmet detections, current tracked riders, and recent
results. - Summary counts for total unique bikes processed, currently
tracked riders, helmet-compliant riders, and no-helmet riders. -
Compliance percentage calculated from classified unique riders.

Color convention: - BikeWithRider: blue. - NoHelmet: red. - Helmet:
green. - In the final compliance display, green indicates
helmet-compliant and red indicates no-helmet/violation.

The crop from a `Helmet` or `NoHelmet` annotation represents the
annotated helmet/head region; it should not be described as a full-rider
crop unless a separate full-rider crop is implemented.

## 10. Important evaluation and reporting cautions

-   Do not call mAP "accuracy." Report precision, recall, mAP50, and
    mAP50-95 with their definitions.
-   Do not report validation metrics as test metrics.
-   Do not claim a rider count from raw frame detections. Tracking and
    counting logic must prevent repeated counting across frames.
-   A detection confidence score is not a guarantee that a detection is
    correct.
-   Keep the original dataset unchanged and preserve the cleaned dataset
    and reports.
-   When adding another dataset, check its licence, class definitions,
    annotation quality, image orientation, duplicates, and train/test
    leakage before combining it.
-   Compare model experiments using the same held-out test set only for
    final evaluation. Use validation results for tuning.

------------------------------------------------------------------------
## Independent test evaluation
- Test images: 237
- Test instances: 850
- Precision: 93.8%
- Recall: 86.5%
- mAP50: 93.0%
- mAP50-95: 66.3%
**Documentation note:** This work log records decisions and outputs
confirmed during development. Exact split counts, the final test
metrics, and the completed tracking/frontend implementation should be
added when those tasks are finished.
