# Helmet Compliance Counter

A computer vision-based system designed to detect motorcycle riders, identify helmet compliance, track individual riders, and generate automated traffic analytics.

## Project Overview

Helmet Compliance Counter aims to improve motorcycle helmet compliance monitoring through computer vision and object detection. The system uses a custom-trained YOLOv8 model to detect motorcycles with riders, helmet usage, and helmet violations in traffic footage.

The project also incorporates multi-object tracking to assign consistent IDs to detected motorcycle riders and support unique rider counting.

## Technologies

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- ByteTrack
- Flask
- HTML, CSS and JavaScript

## Current Progress

- Dataset inspection and annotation auditing
- Dataset cleaning and validation
- Training, validation and testing dataset preparation
- Custom YOLOv8n model training
- Model evaluation using precision, recall, mAP50 and mAP50-95
- Initial Flask-based video detection dashboard
- Initial ByteTrack integration

## Planned Development

- Incorporation of additional traffic video data
- Improved detection of motorcycles with riders
- Reduction of false positives involving cars, bicycles and distant objects
- More reliable rider tracking and unique counting
- Cumulative helmet compliance and violation analytics
- Multiple-image upload and image-by-image analysis
- Video and image export functionality
- CSV and Excel reporting
- Backend security and upload validation

## Project Structure

```text
HelmetComplianceCounter/
├── code/
├── cleaned_dataset/
├── dataset_split/
├── data/
├── documentation/
├── runs/
├── templates/
├── static/
├── app.py
└── yolov8n.pt
```

Large datasets, generated training artifacts, uploaded media and virtual environments are maintained locally rather than committed to the source repository.

## Project Status

Currently under active development. Detection, tracking, analytics, and frontend functionality are being improved through iterative testing on traffic footage.
