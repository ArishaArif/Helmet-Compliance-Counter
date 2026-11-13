import os, io, re, csv, sys, json, time, uuid, secrets, sqlite3, threading, getpass, hashlib, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import wraps
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
try:
    import imageio.v2 as imageio          # writes the annotated MP4 (needs imageio-ffmpeg)
except ImportError:
    imageio = None
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
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "D:/HelmetComplianceCounter/runs/helmet_yolov8n/weights/best.pt")  # EdgeVision

def resolve_model(env_name):
    """Use the model named in .env; if it is unset or missing, fall back to the EdgeVision model and say so."""
    p = os.environ.get(env_name, "").strip()
    if p and not os.path.isabs(p): p = os.path.join(BASE, p)
    if p and os.path.exists(p): return p
    print(f"[models] {env_name} {'file not found' if p else 'not set'}: using fallback {FALLBACK_MODEL}")
    return FALLBACK_MODEL

IMAGE_MODEL, VIDEO_MODEL = resolve_model("IMAGE_MODEL"), resolve_model("VIDEO_MODEL")
LABELS = {1: "NoHelmet", 2: "Helmet"}   # class ids counted as people; change to match your models
CONF, IMAGE_IMGSZ, MIN_FRAMES = 0.25, 640, 3
VIDEO_IMGSZ = int(os.environ.get("VIDEO_IMGSZ", 640))        # 960 costs about 2.25x more CPU time than 640
VIDEO_STRIDE = max(0, int(os.environ.get("VIDEO_STRIDE", 1)))  # 1 = every frame, 2 = every 2nd, 0 = auto (picks 1 to 3 from measured speed)
TRACK_CONF, BIKE_CLASS, MAX_SEATS, HEAD_ZONE, SEAT_SHARE, UNV_FRAMES = 0.10, 0, 3, 0.65, 0.4, 10
STITCH_SECONDS = float(os.environ.get("STITCH_SECONDS", 1.0))   # max gap when joining a lost track to a new ID
STITCH_DIST = float(os.environ.get("STITCH_DIST", 0.75))        # max distance, in bike-box diagonals
STITCH_ENABLED = os.environ.get("STITCH", "0") == "1"           # OFF until you have measured how many ID switches really happen
TRACKER_CFG = os.path.join(BASE, "bytetrack_riders.yaml")
if not os.path.exists(TRACKER_CFG):
    print(f"[tracker] {TRACKER_CFG} not found: using the built-in bytetrack.yaml defaults")
    TRACKER_CFG = "bytetrack.yaml"
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
                          max_idle=30, kwargs={"row_factory": dict_row})
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
    for col in ("unverified INTEGER DEFAULT 0", "proc_fps REAL", "source_fps REAL"):
        try: db().execute(f"ALTER TABLE model_runs ADD COLUMN {col}")
        except Exception: pass                       # column already exists

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
EXPECTED = {0: "BikeWithRider", 1: "NoHelmet", 2: "Helmet"}
class ModelMismatch(Exception): pass

def check_classes(model, label):
    names = {int(k): v for k, v in model.names.items()}
    if names != EXPECTED:
        raise ModelMismatch(f"{label} model classes {names} do not match the expected {EXPECTED}.")

def job_error(e): return str(e) if isinstance(e, ModelMismatch) else f"Processing failed: {type(e).__name__}"

_models = {}
def image_model():
    if "img" not in _models:
        m = YOLO(IMAGE_MODEL); check_classes(m, "Image"); _models["img"] = m
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

def save_dets(job_id, rows):
    """Insert many detections with few database round trips (rows: source, track_id, label, conf, time_sec, crop)."""
    for i in range(0, len(rows), 50):
        part = rows[i:i + 50]
        db().execute("INSERT INTO detections(job_id,source,track_id,label,confidence,time_sec,crop) VALUES "
                     + ",".join(["(?,?,?,?,?,?,?)"] * len(part)),
                     tuple(v for r in part for v in (job_id, r[0], r[1], r[2], round(r[3], 3),
                                                     None if r[4] is None else round(r[4], 2), r[5])))

def set_job(job_id, **kw):
    if kw.get("status") in ("done", "error"): kw["finished"] = now()
    with db() as c:
        c.execute(f"UPDATE model_runs SET {', '.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), job_id))

def run_images(job_id, items):
    try:
        folder, model, rows = job_dir(job_id), image_model(), []
        for st in range(0, len(items), 8):                      # one batched model call per 8 pictures
            chunk = items[st:st + 8]
            results = model.predict([img for _, img in chunk], conf=CONF, imgsz=IMAGE_IMGSZ, verbose=False)
            for off, ((name, img), res) in enumerate(zip(chunk, results)):
                n, vis = st + off + 1, img.copy()
                for k, (label, conf, box, _) in enumerate(head_dets(res), 1):
                    x1, y1, x2, y2 = box
                    fn = f"img{n}_p{k}.jpg"
                    cv2.imwrite(os.path.join(folder, "crops", fn), img[max(y1, 0):y2, max(x1, 0):x2])
                    draw(vis, label, conf, box)
                    rows.append((name, None, label, conf, None, fn))
                cv2.imwrite(os.path.join(folder, f"annotated_{n}.jpg"), vis)
        save_dets(job_id, rows)                                 # one write to the cloud database
        set_job(job_id, status="done", progress=100)
    except Exception as e:
        set_job(job_id, status="error", error=job_error(e))

LIVE, LATEST, RECENT, INFRAME, PROG, ACTIVE = {}, {}, {}, {}, {}, set()   # live state for running video jobs
def _c(b): return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
def _area(b): return max(1, (b[2] - b[0]) * (b[3] - b[1]))

def rider_counts(st):
    """(helmet, no_helmet) for one rider. Only frames where heads were actually detected vote; a missed head is not evidence.
    Seats = the largest head total seen in at least SEAT_SHARE of those frames, so a passenger who is visible part of the time still counts."""
    hist = st["hist"]
    P = 1
    for k in range(max(h + n for h, n in hist), 0, -1):
        if sum(h + n >= k for h, n in hist) >= SEAT_SHARE * len(hist): P = k; break
    P = min(P, MAX_SEATS)
    pool = [h for h, n in hist if h + n == P] or [min(h, P) for h, n in hist if h + n > P]
    H = min(Counter(pool).most_common(1)[0][0], P)
    return H, P - H

def confirmed(st): return st["n"] >= MIN_FRAMES and len(st["hist"]) >= MIN_FRAMES
def unverified(st): return st["n"] >= UNV_FRAMES and len(st["hist"]) < MIN_FRAMES   # tracked a while, heads never reliably seen

def stitch(states, gap):
    """Join tracks that are probably one bike whose ID changed: the new track starts within `gap` frames of the old one's end,
    nearby and of similar size. Tracks that overlap in time are different bikes and are never joined."""
    if not STITCH_ENABLED: gap = 0                      # no merging: every ByteTrack ID is its own rider
    groups = []
    for bid, st in sorted(states.items(), key=lambda kv: kv[1]["f0"]):
        best, bd = None, None
        for g in groups:
            if not 0 < st["f0"] - g["f1"] <= gap: continue
            (ax, ay), (bx, by) = _c(g["b1"]), _c(st["b0"])
            d = math.hypot(ax - bx, ay - by) / max(math.hypot(g["b1"][2] - g["b1"][0], g["b1"][3] - g["b1"][1]), 1)
            if d <= STITCH_DIST and 0.5 <= _area(st["b0"]) / _area(g["b1"]) <= 2 and (bd is None or d < bd): best, bd = g, d
        if best is None:
            groups.append({**st, "hist": list(st["hist"]), "c": {k: list(v) for k, v in st["c"].items()}, "ids": [bid]})
        else:
            best.setdefault("merges", []).append((bid, st["f0"] - best["f1"], round(bd, 2)))   # logged so false merges can be checked
            best["n"] += st["n"]; best["hist"] += st["hist"]; best["ids"].append(bid)
            for k in best["c"]: best["c"][k][0] += st["c"][k][0]; best["c"][k][1] += st["c"][k][1]
            if st["bp"] > best["bp"]: best["bp"], best["crop"] = st["bp"], st["crop"]
            best["f1"], best["b1"] = st["f1"], st["b1"]
    return groups

def assign_heads(bikes, heads):
    """Give each head to the bike whose upper part contains it; if boxes overlap, prefer the bike whose centre line is closest."""
    out = defaultdict(list)
    for hd in heads:
        cx, cy = _c(hd[2]); owners = []
        for bid, b in bikes.items():
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[1] + HEAD_ZONE * (b[3] - b[1]):
                owners.append((abs(cx - (b[0] + b[2]) / 2) / max(b[2] - b[0], 1), bid))
        if owners: out[min(owners)[1]].append(hd)
    return out

def live_counts(states, gap):
    c = Counter()
    for st in stitch(states, gap):
        if confirmed(st): h, n = rider_counts(st); c["Helmet"] += h; c["NoHelmet"] += n
        elif unverified(st): c["Unverified"] += 1
    return c

def run_video(job_id, path, name):
    folder, cap, writer = job_dir(job_id), cv2.VideoCapture(path), None
    try:
        model = YOLO(VIDEO_MODEL)                      # own instance, so tracker state is not shared
        check_classes(model, "Video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        states, last, i, t0 = {}, ({}, []), 0, time.time()
        use_out, stride, gap = imageio is not None, (1 if VIDEO_STRIDE == 0 else VIDEO_STRIDE), int(STITCH_SECONDS * fps)
        while True:
            ok, frame = cap.read()
            if not ok: break
            if i % stride == 0:
                res = model.track(frame, persist=True, tracker=TRACKER_CFG, conf=TRACK_CONF,
                                  imgsz=VIDEO_IMGSZ, verbose=False)[0]
                bikes, heads = {}, []                  # ByteTrack follows the BikeWithRider boxes; heads are matched to them
                if res.boxes is not None and len(res.boxes):
                    ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
                    for bx, c, p, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                             res.boxes.conf.cpu().numpy(), ids):
                        bx = tuple(int(v) for v in bx)
                        if c == BIKE_CLASS and tid is not None: bikes[tid] = (bx, float(p))
                        elif c in LABELS and p >= CONF: heads.append((LABELS[c], float(p), bx))
                last = (bikes, heads)
                per = assign_heads({k: v[0] for k, v in bikes.items()}, heads)
                for bid, (bx, bp) in bikes.items():
                    st = states.setdefault(bid, {"n": 0, "hist": [], "c": {"Helmet": [0.0, 0], "NoHelmet": [0.0, 0]},
                                                 "t": i / fps, "bp": 0.0, "crop": None, "shown": False,
                                                 "f0": i, "f1": i, "b0": bx, "b1": bx})
                    st["n"] += 1; st["f1"], st["b1"] = i, bx
                    mine = per.get(bid, [])
                    if mine: st["hist"].append((sum(m[0] == "Helmet" for m in mine), sum(m[0] == "NoHelmet" for m in mine)))
                    for lab, cf, _ in mine: st["c"][lab][0] += cf; st["c"][lab][1] += 1
                    if bp > st["bp"]:
                        x1, y1, x2, y2 = bx; st["bp"], st["crop"] = bp, frame[max(y1, 0):y2, max(x1, 0):x2].copy()
                    if confirmed(st) and not st["shown"]:          # confirmed rider: show in "recent results" now
                        st["shown"] = True; _, n_ = rider_counts(st); fn = f"track{bid}.jpg"
                        cv2.imwrite(os.path.join(folder, "crops", fn), st["crop"])
                        RECENT.setdefault(job_id, []).append(dict(source=name, track_id=bid, label="NoHelmet" if n_ else "Helmet",
                            confidence=round(st["bp"], 3), time_sec=round(st["t"], 2), crop=fn))
                INFRAME[job_id] = len(bikes)
            bikes, heads = last
            vis = frame.copy()
            for bid, (bx, bp) in bikes.items():
                st = states.get(bid)
                if st and confirmed(st):
                    h, n = rider_counts(st); col, txt = ((80, 80, 240) if n else (110, 200, 80)), f"#{bid} H{h} N{n}"
                else:
                    col, txt = (60, 180, 240), f"#{bid} checking"
                cv2.rectangle(vis, bx[:2], bx[2:], col, 2)
                cv2.putText(vis, txt, (bx[0], max(bx[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            for lab, cf, hb in heads:
                cv2.rectangle(vis, hb[:2], hb[2:], (80, 80, 240) if lab == "NoHelmet" else (110, 200, 80), 1)
            h, w = vis.shape[:2]; sc = min(1.0, 960 / w)
            small = cv2.resize(vis, (int(w * sc) // 2 * 2, int(h * sc) // 2 * 2))
            if writer is None and use_out:     # every frame is written, so playback keeps the original speed
                try:
                    writer = imageio.get_writer(os.path.join(folder, "annotated.mp4"), fps=fps, codec="libx264", pixelformat="yuv420p",
                                                macro_block_size=1, output_params=["-preset", "ultrafast", "-crf", "26"])
                except Exception: use_out = False
            if writer is not None: writer.append_data(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            i += 1
            if VIDEO_STRIDE == 0 and i == 30:               # auto mode: skip frames only as much as the measured speed requires
                stride = min(3, max(1, math.ceil(fps / (i / max(time.time() - t0, 1e-6)))))
            if i % 2 == 0: LATEST[job_id] = cv2.imencode(".jpg", small)[1].tobytes()
            if i % 5 == 0:                                 # in-memory only: no cloud round trip inside the video loop
                LIVE[job_id] = live_counts(states, gap); PROG[job_id] = min(99, int(100 * i / total))
        groups, rows = stitch(states, gap), []
        for g in groups:
            if g.get("merges"): print(f"[stitch] rider {g['ids']}: joined (id, gap frames, distance) {g['merges']}")
        for st in groups:                                  # database rows: one per person on each confirmed rider
            if not confirmed(st): continue
            h, n = rider_counts(st); bid = st["ids"][0]; fn = f"track{bid}.jpg"
            cv2.imwrite(os.path.join(folder, "crops", fn), st["crop"])
            for lab, k in (("Helmet", h), ("NoHelmet", n)):
                sm, m = st["c"][lab]
                rows += [(name, bid, lab, sm / m if m else st["bp"], st["t"], fn)] * k
        save_dets(job_id, rows)
        if job_id in LATEST:
            with open(os.path.join(folder, "preview.jpg"), "wb") as f: f.write(LATEST[job_id])
        if writer is not None: writer.close(); writer = None
        el = max(time.time() - t0, 1e-6)
        print(f"[video {job_id[:8]}] {i} frames in {el:.1f}s = {i / el:.1f} fps processed (video is {fps:.1f} fps), imgsz={VIDEO_IMGSZ}, stride={stride}, tracks {len(states)} -> riders {len(groups)}")
        set_job(job_id, status="done", progress=100, unverified=sum(unverified(g) for g in groups),
                proc_fps=round(i / el, 1), source_fps=round(fps, 1))
    except Exception as e:
        set_job(job_id, status="error", error=job_error(e))
    finally:
        cap.release(); ACTIVE.discard(job_id)
        if writer is not None:
            try: writer.close()
            except Exception: pass
        for d in (LIVE, LATEST, RECENT, INFRAME, PROG): d.pop(job_id, None)

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
        threading.Thread(target=run_images, args=(job_id, items), daemon=True).start()
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
    return jsonify(job={**dict(job), "progress": PROG.get(job_id, job["progress"])}, detections=dets, in_frame=INFRAME.get(job_id, 0),
                   counts={**(summary(live["Helmet"], live["NoHelmet"]) if live is not None else summary(sum(d["label"] == "Helmet" for d in dets), sum(d["label"] == "NoHelmet" for d in dets))),
                           "unverified": live["Unverified"] if live is not None else (job["unverified"] or 0)},
                   annotated=sorted(f for f in os.listdir(d) if f.startswith(("annotated", "preview"))) if os.path.isdir(d) else [])

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
    threading.Thread(target=image_model, daemon=True).start()   # load the model now, not on the first upload
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
