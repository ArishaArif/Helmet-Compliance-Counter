import os, io, re, csv, sys, json, time, uuid, secrets, sqlite3, threading, getpass, hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import wraps
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from flask import (Flask, render_template, request, session, redirect, url_for,
                   jsonify, send_from_directory, Response, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))   # reads SECRET_KEY, model paths, etc.

# ---------------- configuration ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
MEDIA, UPLOADS = os.path.join(BASE, "media"), os.path.join(BASE, "uploads")
DB_PATH = os.path.join(BASE, "app.db")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", r"models\edgevision_best.pt")   # trained on EdgeVision
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", r"models\edgevision_best.pt")         # trained on OSF clips
LABELS = {1: "NoHelmet", 2: "Helmet"}   # class ids counted as people; change to match your models
CONF, IMAGE_IMGSZ, VIDEO_IMGSZ, MIN_FRAMES = 0.25, 640, 960, 3
MAX_IMG_BYTES, MAX_IMAGES, MAX_PIXELS = 10 * 1024 * 1024, 20, 40_000_000
IMG_EXT, VID_EXT = {"jpg", "jpeg", "png"}, {"mp4", "avi", "mov", "mkv"}
for d in (MEDIA, UPLOADS):
    os.makedirs(d, exist_ok=True)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

app = Flask(__name__)
app.config.update(SECRET_KEY=os.environ["SECRET_KEY"], MAX_CONTENT_LENGTH=600 * 1024 * 1024,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

# ---------------- database (parameterised queries only) ----------------
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))
if USE_PG:                                   # Neon (cloud Postgres)
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    POOL = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=True,
                          check=ConnectionPool.check_connection, kwargs={"row_factory": dict_row})
    UNIQUE_ERRORS = (psycopg.errors.UniqueViolation,)
else:                                        # local fallback when DATABASE_URL is empty
    UNIQUE_ERRORS = (sqlite3.IntegrityError,)

def _run(sql, args=()):
    if USE_PG:
        with POOL.connection() as c:         # commits on success, rolls back on error
            cur = c.execute(sql.replace("?", "%s"), args)
            return cur.fetchall() if cur.description else []
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row
    try:
        with c:
            cur = c.execute(sql, args)
            return cur.fetchall() if cur.description else []
    finally:
        c.close()

class _Result(list):
    def fetchone(self): return self[0] if self else None
    def fetchall(self): return list(self)

class _DB:   # db().execute(...) works the same on SQLite and Postgres, always with parameters
    def execute(self, sql, args=()): return _Result(_run(sql, args))
    def __enter__(self): return self
    def __exit__(self, *a): return False

def db(): return _DB()

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def init_db():
    pk = "BIGSERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY"
    for sql in (
        f"CREATE TABLE IF NOT EXISTS users(id {pk}, email TEXT UNIQUE NOT NULL, pw TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'analyst')",
        "CREATE TABLE IF NOT EXISTS model_runs(id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, kind TEXT, filename TEXT, created TEXT, "
        "status TEXT, progress INTEGER DEFAULT 0, error TEXT, sha256 TEXT, model_file TEXT, conf REAL, imgsz INTEGER, finished TEXT)",
        f"CREATE TABLE IF NOT EXISTS detections(id {pk}, job_id TEXT NOT NULL, source TEXT, track_id INTEGER, label TEXT, "
        "confidence REAL, time_sec REAL, crop TEXT)",
        "CREATE INDEX IF NOT EXISTS idx_det_job ON detections(job_id)"):
        db().execute(sql)

try:
    init_db()
    print("Database:", "Neon / Postgres" if USE_PG else "local SQLite (set DATABASE_URL in .env to use the cloud)")
except Exception as e:
    sys.exit(f"Database connection failed ({type(e).__name__}). Check DATABASE_URL in .env.")

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------------- validation ----------------
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")

def password_error(pw):
    if len(pw) < 8: return "Password must be at least 8 characters."
    if not re.search(r"[a-z]", pw) or not re.search(r"[A-Z]", pw):
        return "Use both upper-case and lower-case letters."
    if not re.search(r"\d", pw): return "Include at least one number."
    if not re.search(r"[^A-Za-z0-9]", pw): return "Include at least one special character."

def load_image(file):
    """Validate and re-decode an upload. Re-decoding drops metadata and hidden payloads."""
    raw = file.read(MAX_IMG_BYTES + 1)
    if len(raw) > MAX_IMG_BYTES: raise ValueError("larger than 10 MB")
    try:
        Image.open(io.BytesIO(raw)).verify()
        im = Image.open(io.BytesIO(raw))
        if im.format not in ("JPEG", "PNG"): raise ValueError
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR), hashlib.sha256(raw).hexdigest()
    except Exception:
        raise ValueError("not a valid JPEG or PNG image")

def video_ok(path):
    with open(path, "rb") as f: head = f.read(12)
    if not (b"ftyp" in head[4:12] or head[:4] == b"RIFF" or head[:4] == b"\x1a\x45\xdf\xa3"):
        return False
    cap = cv2.VideoCapture(path); ok = cap.isOpened(); cap.release()
    return ok

# ---------------- security: sessions, CSRF, login throttle ----------------
FAILS = {}

def locked(key):
    n, until = FAILS.get(key, (0, 0))
    if n and time.time() > until:
        FAILS.pop(key, None); return False
    return n >= 5

@app.before_request
def csrf_protect():
    session.setdefault("csrf", secrets.token_hex(16))
    if request.method == "POST":
        sent = request.form.get("csrf") or request.headers.get("X-CSRF") or ""
        if not secrets.compare_digest(sent, session["csrf"]): abort(400)

@app.context_processor
def inject(): return {"csrf": session.get("csrf", "")}

@app.after_request
def headers(r):
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    return r

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session:
            return (jsonify(error="Please sign in."), 401) if request.path.startswith("/api") else redirect(url_for("login"))
        return f(*a, **k)
    return w

def admin_required(f):
    @wraps(f)
    @login_required
    def w(*a, **k):
        if session.get("role") != "admin": abort(403)
        return f(*a, **k)
    return w

@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        key, email = request.remote_addr, request.form.get("email", "").strip().lower()
        if locked(key):
            err = "Too many attempts. Try again in a few minutes."
        else:
            row = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            if row and check_password_hash(row["pw"], request.form.get("password", "")):
                FAILS.pop(key, None); session.clear()
                session.update(uid=row["id"], role=row["role"], email=row["email"], csrf=secrets.token_hex(16))
                return redirect(url_for("index"))
            FAILS[key] = (FAILS.get(key, (0, 0))[0] + 1, time.time() + 300)
            err = "Invalid email or password."
    return render_template("login.html", error=err)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/api/users", methods=["POST"])
@admin_required
def create_user():
    d = request.get_json(silent=True) or {}
    email, pw, role = str(d.get("email", "")).strip().lower(), str(d.get("password", "")), d.get("role", "analyst")
    if len(email) > 254 or not EMAIL_RE.match(email): return jsonify(error="Enter a valid email address."), 400
    if password_error(pw): return jsonify(error=password_error(pw)), 400
    if role not in ("admin", "analyst"): return jsonify(error="Role must be admin or analyst."), 400
    try:
        with db() as c:
            c.execute("INSERT INTO users(email, pw, role) VALUES(?,?,?)", (email, generate_password_hash(pw), role))
    except UNIQUE_ERRORS:
        return jsonify(error="That email already has an account."), 409
    return jsonify(ok=True)

# ---------------- detection ----------------
_models = {}
def image_model():
    if "img" not in _models: _models["img"] = YOLO(IMAGE_MODEL)
    return _models["img"]

def head_dets(res):
    out = []
    if res.boxes is None: return out
    ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
    for b, c, p, i in zip(res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int),
                          res.boxes.conf.cpu().numpy(), ids):
        if c in LABELS: out.append((LABELS[c], float(p), tuple(int(v) for v in b), i))
    return out

def draw(img, label, conf, box, tid=None):
    x1, y1, x2, y2 = box
    col = (80, 80, 240) if label == "NoHelmet" else (110, 200, 80)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    cv2.putText(img, f"{'#%d ' % tid if tid is not None else ''}{label} {conf:.0%}", (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

def job_dir(job_id):
    d = os.path.join(MEDIA, job_id); os.makedirs(os.path.join(d, "crops"), exist_ok=True); return d

def save_det(job_id, source, tid, label, conf, t, crop):
    with db() as c:
        c.execute("INSERT INTO detections(job_id,source,track_id,label,confidence,time_sec,crop) VALUES(?,?,?,?,?,?,?)",
                  (job_id, source, tid, label, round(conf, 3), None if t is None else round(t, 2), crop))

def set_job(job_id, **kw):
    if kw.get("status") in ("done", "error"): kw["finished"] = now()
    with db() as c:
        c.execute(f"UPDATE model_runs SET {', '.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), job_id))

def run_images(job_id, items):
    folder, model = job_dir(job_id), image_model()
    for n, (name, img) in enumerate(items, 1):
        vis = img.copy()
        res = model.predict(img, conf=CONF, imgsz=IMAGE_IMGSZ, verbose=False)[0]
        for k, (label, conf, box, _) in enumerate(head_dets(res), 1):
            x1, y1, x2, y2 = box
            fn = f"img{n}_p{k}.jpg"
            cv2.imwrite(os.path.join(folder, "crops", fn), img[max(y1, 0):y2, max(x1, 0):x2])
            draw(vis, label, conf, box)
            save_det(job_id, name, None, label, conf, None, fn)
        cv2.imwrite(os.path.join(folder, f"annotated_{n}.jpg"), vis)
        set_job(job_id, progress=int(100 * n / len(items)))
    set_job(job_id, status="done")

LIVE, LATEST, RECENT, INFRAME, ACTIVE = {}, {}, {}, {}, set()   # live state for running video jobs
def run_video(job_id, path, name):
    folder, cap = job_dir(job_id), cv2.VideoCapture(path)
    try:
        model = YOLO(VIDEO_MODEL)                      # own instance, so tracker state is not shared
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        votes, best, first, i = defaultdict(Counter), {}, {}, 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            res = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF,
                              imgsz=VIDEO_IMGSZ, verbose=False)[0]
            vis, now_in = frame.copy(), 0
            for label, conf, box, tid in head_dets(res):
                if tid is None: continue
                now_in += 1
                votes[tid][label] += 1; first.setdefault(tid, i / fps)
                x1, y1, x2, y2 = box
                crop = frame[max(y1, 0):y2, max(x1, 0):x2].copy()
                if conf > best.get(tid, (0,))[0]: best[tid] = (conf, crop)
                if sum(votes[tid].values()) == MIN_FRAMES:   # person confirmed: show in "recent results" immediately
                    fn = f"track{tid}.jpg"; cv2.imwrite(os.path.join(folder, "crops", fn), crop)
                    RECENT.setdefault(job_id, []).append(dict(source=name, track_id=tid, label=label,
                        confidence=round(conf, 3), time_sec=round(first[tid], 2), crop=fn))
                draw(vis, label, conf, box, tid)
            i += 1; INFRAME[job_id] = now_in
            if i % 2 == 0:
                h, w = vis.shape[:2]
                LATEST[job_id] = cv2.imencode(".jpg", cv2.resize(vis, (960, int(h * 960 / w))) if w > 960 else vis)[1].tobytes()
            if i % 5 == 0:
                LIVE[job_id] = Counter(c.most_common(1)[0][0] for c in votes.values() if sum(c.values()) >= MIN_FRAMES)
                set_job(job_id, progress=min(99, int(100 * i / total)))
        for tid, c in votes.items():                  # one row per tracked person, label by majority vote
            if sum(c.values()) < MIN_FRAMES: continue
            conf, crop = best[tid]; fn = f"track{tid}.jpg"
            cv2.imwrite(os.path.join(folder, "crops", fn), crop)
            save_det(job_id, name, tid, c.most_common(1)[0][0], conf, first[tid], fn)
        if job_id in LATEST:
            with open(os.path.join(folder, "preview.jpg"), "wb") as f: f.write(LATEST[job_id])
        set_job(job_id, status="done", progress=100)
    except Exception as e:
        set_job(job_id, status="error", error=f"Processing failed: {type(e).__name__}")
    finally:
        cap.release(); ACTIVE.discard(job_id)
        for d in (LIVE, LATEST, RECENT, INFRAME): d.pop(job_id, None)

def counts(job_id):
    d = {r["label"]: r["n"] for r in db().execute(
        "SELECT label, COUNT(*) n FROM detections WHERE job_id=? GROUP BY label", (job_id,))}
    return summary(d.get("Helmet", 0), d.get("NoHelmet", 0))

def summary(h, n):
    return {"helmet": h, "no_helmet": n, "total": h + n, "compliance": round(100 * h / (h + n), 1) if h + n else None}

# ---------------- routes ----------------
def get_job(job_id):
    row = db().execute("SELECT * FROM model_runs WHERE id=?", (job_id,)).fetchone()
    if not row or (row["user_id"] != session["uid"] and session.get("role") != "admin"): abort(404)
    return row

@app.route("/")
@login_required
def index():
    return render_template("index.html", email=session.get("email"))

@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    files = [f for f in request.files.getlist("files") if f.filename]
    if not files: return jsonify(error="Choose at least one image or a video."), 400
    exts = {f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "" for f in files}
    job_id = uuid.uuid4().hex
    if exts <= IMG_EXT:
        if len(files) > MAX_IMAGES: return jsonify(error=f"Upload at most {MAX_IMAGES} images at once."), 400
        items, shas = [], []
        for f in files:
            try:
                img, sha = load_image(f); items.append((secure_filename(f.filename), img)); shas.append(sha)
            except ValueError as e: return jsonify(error=f"{secure_filename(f.filename)} was rejected: {e}."), 400
        kind, label = "image", f"{len(items)} image(s)"
        sha = shas[0] if len(shas) == 1 else hashlib.sha256("".join(shas).encode()).hexdigest()
    elif len(files) == 1 and exts <= VID_EXT:
        name = secure_filename(files[0].filename)
        path = os.path.join(UPLOADS, f"{job_id}.{name.rsplit('.', 1)[-1].lower()}")
        files[0].save(path)
        if not video_ok(path):
            os.remove(path); return jsonify(error=f"{name} was rejected: not a valid video file."), 400
        kind, label = "video", name
        sha = file_sha256(path)
    else:
        return jsonify(error="Upload JPG/PNG images, or a single MP4, AVI, MOV or MKV video."), 400
    with db() as c:
        c.execute("INSERT INTO model_runs(id,user_id,kind,filename,created,status,sha256,model_file,conf,imgsz) "
                  "VALUES(?,?,?,?,?,'running',?,?,?,?)",
                  (job_id, session["uid"], kind, label, now(), sha,
                   os.path.basename(IMAGE_MODEL if kind == "image" else VIDEO_MODEL), CONF,
                   IMAGE_IMGSZ if kind == "image" else VIDEO_IMGSZ))
    if kind == "image":
        try: run_images(job_id, items)
        except Exception as e: set_job(job_id, status="error", error=f"Processing failed: {type(e).__name__}")
    else:
        ACTIVE.add(job_id)
        threading.Thread(target=run_video, args=(job_id, path, name), daemon=True).start()
    return jsonify(job_id=job_id)

@app.route("/api/jobs")
@login_required
def jobs():
    rows = db().execute("SELECT id,kind,filename,created,status FROM model_runs WHERE user_id=? ORDER BY created DESC LIMIT 30",
                        (session["uid"],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/jobs/<job_id>")
@login_required
def job_status(job_id):
    job = get_job(job_id)
    dets = [dict(r) for r in db().execute(
        "SELECT source,track_id,label,confidence,time_sec,crop FROM detections WHERE job_id=? ORDER BY time_sec, id", (job_id,))]
    if job["status"] == "running" and job_id in RECENT: dets = list(RECENT[job_id])
    live = LIVE.get(job_id)
    d = os.path.join(MEDIA, job_id)
    return jsonify(job=dict(job), detections=dets, in_frame=INFRAME.get(job_id, 0),
                   counts=summary(live["Helmet"], live["NoHelmet"]) if live is not None else counts(job_id),
                   annotated=sorted(f for f in os.listdir(d) if f.startswith(("annotated_", "preview"))) if os.path.isdir(d) else [])

@app.route("/api/jobs/<job_id>/stream")
@login_required
def stream(job_id):
    get_job(job_id)
    def gen():                      # live annotated frames while a video is processing
        last = None
        while job_id in ACTIVE:
            frame = LATEST.get(job_id)
            if frame is not None and frame is not last:
                last = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/media/<job_id>/<path:name>")
@login_required
def media(job_id, name):
    get_job(job_id)
    return send_from_directory(os.path.join(MEDIA, job_id), name)

@app.route("/api/jobs/<job_id>/export")
@login_required
def export(job_id):
    job = get_job(job_id)
    fmt, cls = request.args.get("format", "csv"), request.args.get("cls", "all")
    if fmt not in ("csv", "txt", "xml", "json", "pdf") or cls not in ("all", "Helmet", "NoHelmet"): abort(400)
    q, args = "SELECT source,track_id,label,confidence,time_sec FROM detections WHERE job_id=?", [job_id]
    if cls != "all": q += " AND label=?"; args.append(cls)
    rows = [{"processed_at": job["created"], "input_file": r["source"], "track_id": r["track_id"] if r["track_id"] is not None else "",
             "class": r["label"], "confidence": r["confidence"], "time_sec": r["time_sec"] if r["time_sec"] is not None else ""}
            for r in db().execute(q + " ORDER BY time_sec, id", args)]
    cols, summ = list(rows[0].keys()) if rows else ["processed_at", "input_file", "track_id", "class", "confidence", "time_sec"], counts(job_id)
    fname = f"helmet_report_{job_id[:8]}_{cls}.{fmt}"
    head = {"Content-Disposition": f'attachment; filename="{fname}"'}
    if fmt == "csv":
        buf = io.StringIO(); w = csv.writer(buf); w.writerow(cols)
        for r in rows:  # leading quote blocks spreadsheet formula injection via file names
            w.writerow([("'" + str(v)) if str(v)[:1] in "=+-@" else v for v in r.values()])
        return Response(buf.getvalue(), mimetype="text/csv", headers=head)
    if fmt == "txt":
        lines = [f"Helmet report | input: {job['filename']} | processed: {job['created']} | sha256: {job['sha256']} | filter: {cls}",
                 f"Total {summ['total']} | Helmet {summ['helmet']} | NoHelmet {summ['no_helmet']}", "\t".join(cols)]
        lines += ["\t".join(str(v) for v in r.values()) for r in rows]
        return Response("\n".join(lines), mimetype="text/plain", headers=head)
    if fmt == "json":
        return Response(json.dumps({"input": job["filename"], "processed_at": job["created"], "input_sha256": job["sha256"], "filter": cls,
                                    "summary": summ, "detections": rows}, indent=2), mimetype="application/json", headers=head)
    if fmt == "xml":
        root = ET.Element("report", input=str(job["filename"]), processed_at=job["created"], filter=cls)
        ET.SubElement(root, "summary", {k: str(v) for k, v in summ.items()})
        for r in rows: ET.SubElement(root, "detection", {k: str(v) for k, v in r.items()})
        return Response(ET.tostring(root, encoding="utf-8", xml_declaration=True), mimetype="application/xml", headers=head)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
        from reportlab.lib import colors
    except ImportError:
        return jsonify(error="PDF export needs reportlab (pip install reportlab)."), 501
    buf, st = io.BytesIO(), getSampleStyleSheet()
    tbl = Table([cols] + [[str(v) for v in r.values()] for r in rows], repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2430")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                             ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    SimpleDocTemplate(buf, pagesize=A4).build([
        Paragraph("Helmet compliance report", st["Title"]),
        Paragraph(f"Input: {job['filename']}<br/>Processed: {job['created']}<br/>Filter: {cls}<br/>"
                  f"Total {summ['total']} | Helmet {summ['helmet']} | NoHelmet {summ['no_helmet']}", st["Normal"]),
        Spacer(1, 12), tbl])
    return Response(buf.getvalue(), mimetype="application/pdf", headers=head)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "create-admin":   # python app.py create-admin you@example.com
        email, pw = sys.argv[2].strip().lower(), getpass.getpass("Password: ")
        if not EMAIL_RE.match(email) or password_error(pw): sys.exit(password_error(pw) or "Invalid email.")
        with db() as c:
            c.execute("INSERT INTO users(email, pw, role) VALUES(?,?,'admin')", (email, generate_password_hash(pw)))
        sys.exit("Admin created.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)