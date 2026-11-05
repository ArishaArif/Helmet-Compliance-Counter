import os
import cv2
import time
import threading
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = r"runs\helmet_yolov8n\weights\best.pt"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model = YOLO(MODEL_PATH)

video_path = None
processing = False
start_time = 0
stats_lock = threading.Lock()

stats = {
    "total": 0,
    "current": 0,
    "helmet": 0,
    "no_helmet": 0,
    "riders": []
}

CLASS_NAMES = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}

COLORS = {
    0: (255, 0, 0),    # Blue
    1: (0, 0, 255),    # Red
    2: (0, 255, 0)     # Green
}


def generate_frames():
    global processing, start_time, stats

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        processing = False
        return

    start_time = time.time()
    seen_ids = set()
    rider_status = {}

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            imgsz=416,
            device="cpu",
            verbose=False
        )

        result = results[0]
        bikes = []
        helmet_boxes = []
        nohelmet_boxes = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            ids = (
                result.boxes.id.int().cpu().tolist()
                if result.boxes.id is not None
                else [None] * len(boxes)
            )

            for box, cls, conf, track_id in zip(
                boxes, classes, confidences, ids
            ):
                x1, y1, x2, y2 = map(int, box)

                if cls == 0:
                    bikes.append({
                        "box": (x1, y1, x2, y2),
                        "id": track_id
                    })
                elif cls == 1:
                    nohelmet_boxes.append((x1, y1, x2, y2, conf))
                elif cls == 2:
                    helmet_boxes.append((x1, y1, x2, y2, conf))

        current_riders = []

        for bike in bikes:
            x1, y1, x2, y2 = bike["box"]
            rider_id = bike["id"]

            if rider_id is None:
                continue

            seen_ids.add(rider_id)

            # Match helmet detections whose centers fall inside the bike box.
            matched = []

            for hx1, hy1, hx2, hy2, conf in helmet_boxes:
                cx = (hx1 + hx2) // 2
                cy = (hy1 + hy2) // 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    matched.append(("Helmet", conf, (hx1, hy1, hx2, hy2)))

            for hx1, hy1, hx2, hy2, conf in nohelmet_boxes:
                cx = (hx1 + hx2) // 2
                cy = (hy1 + hy2) // 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    matched.append(("NoHelmet", conf, (hx1, hy1, hx2, hy2)))

            # Prefer NoHelmet if both classes are matched.
            no_helmet_match = next(
                (item for item in matched if item[0] == "NoHelmet"), None
            )
            helmet_match = next(
                (item for item in matched if item[0] == "Helmet"), None
            )

            detection = no_helmet_match or helmet_match

            if detection:
                status, confidence, head_box = detection
                rider_status[rider_id] = (status, confidence, head_box)
            else:
                status, confidence, head_box = rider_status.get(
                    rider_id, ("Unknown", 0, None)
                )

            if status == "NoHelmet":
                color = (0, 0, 255)
            elif status == "Helmet":
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            label = f"ID {rider_id} | {status}"
            if confidence:
                label += f" | {confidence:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

            if head_box:
                hx1, hy1, hx2, hy2 = head_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)

            current_riders.append({
                "id": rider_id,
                "status": status,
                "confidence": round(float(confidence) * 100, 1)
            })

        with stats_lock:
            stats["total"] = len(seen_ids)
            stats["current"] = len(bikes)
            stats["helmet"] = sum(
                1 for r in current_riders if r["status"] == "Helmet"
            )
            stats["no_helmet"] = sum(
                1 for r in current_riders if r["status"] == "NoHelmet"
            )
            stats["riders"] = current_riders
            stats["elapsed"] = int(time.time() - start_time)

        # Resize for a smoother dashboard on this CPU-based laptop.
        height, width = frame.shape[:2]
        if width > 960:
            new_height = int(height * 960 / width)
            frame = cv2.resize(frame, (960, new_height))

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() + b"\r\n"
        )

    cap.release()
    processing = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global video_path, processing, stats

    if "video" not in request.files:
        return "No video selected", 400

    file = request.files["video"]

    if not file.filename:
        return "No video selected", 400

    filename = secure_filename(file.filename)
    extension = os.path.splitext(filename)[1].lower()

    if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        return "Please upload an MP4, AVI, MOV or MKV video.", 400

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    processing = True

    with stats_lock:
        stats = {
            "total": 0,
            "current": 0,
            "helmet": 0,
            "no_helmet": 0,
            "riders": [],
            "elapsed": 0
        }

    return "Video uploaded"


@app.route("/video_feed")
def video_feed():
    if not video_path:
        return "Upload a video first", 400

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def get_stats():
    with stats_lock:
        return jsonify(stats)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)