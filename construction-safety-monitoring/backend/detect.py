"""
SentinelSite - Detection module
------------------------------------------------------------
Two separate paths, matching the two upload types on the dashboard:

  1. analyze_image(path)  -> PPE, restricted-zone, machinery-proximity checks
                              on a single uploaded photo.
  2. analyze_video(path)  -> fall detection across frames of a short
                              uploaded video clip.

Both reuse the SAME YOLOv8 person/PPE detector. Fall detection is a rule
(aspect ratio + centroid drop) applied to the same person boxes over time,
not a separate AI model.

Setup:
    pip install ultralytics opencv-python

IMPORTANT — model choice:
  * yolov8n.pt (stock) is trained on COCO. It detects `person`, `car`,
    `truck`, etc. but has NO PPE classes, so PPE violations (helmet/vest)
    will NOT fire and the violations table stays empty for those.
  * Once you fine-tune on the PPE dataset (Sprint 2), point MODEL_PATH at
    models/best.pt and the NO-Hardhat / NO-Safety Vest checks start working.
"""

import os
import base64

import cv2

# --- Model path resolved relative to THIS file, so it works no matter what
#     directory you launch `python app.py` from. ---------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

#MODEL_PATH = os.path.join(_HERE, "yolov8n.pt")        # stock model (until Sprint 2)
MODEL_PATH = os.path.join(_HERE, "models", "best.pt")  # <- fine-tuned PPE weights

# --- Load the model once, with a clear error if it can't be loaded. --------
#     If this fails, `import detect` fails and Flask won't start — which is
#     the usual reason uploads "suddenly" stop working. The message below
#     tells you exactly what to fix.
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is not installed in this environment. "
        "Run:  pip install ultralytics opencv-python"
    ) from e

try:
    model = YOLO(MODEL_PATH)
    print("=" * 60)
    print("SentinelSite PPE Model")
    print("Model path:", MODEL_PATH)

    params = sum(p.numel() for p in model.model.parameters())
    print("Model parameters:", params)

    model.info(verbose=True)
    print("=" * 60)
except Exception as e:
    raise RuntimeError(
        f"Could not load YOLO model at '{MODEL_PATH}'. "
        f"Check the file exists (or that you're online so it can download). "
        f"Original error: {e}"
    ) from e

# Separate POSE model for the video / fall path. This finds 17 body
# keypoints per person (shoulders, hips, knees, etc.) so falls can be
# judged from actual posture instead of box shape. yolov8n-pose.pt
# downloads automatically the first time (needs internet once).
POSE_MODEL_PATH = os.path.join(_HERE, "yolov8m-pose.pt")
try:
    pose_model = YOLO(POSE_MODEL_PATH)
    print("Fall detection model:", POSE_MODEL_PATH)
    pose_model.info(verbose=True)
except Exception as e:
    raise RuntimeError(
        f"Could not load YOLO pose model at '{POSE_MODEL_PATH}'. "
        f"Make sure you're online for the first download. Original error: {e}"
    ) from e


# Restricted zone rectangle in pixel coords (x1, y1, x2, y2).
# Defined as a FRACTION of the image size (not fixed pixels), so it scales
# correctly no matter what resolution photo gets uploaded. This currently
# marks the right-hand ~18% strip of the image as restricted — replace
# these fractions with values that match your actual camera framing.
RESTRICTED_ZONE_FRACTION = (0.82, 0.05, 0.98, 0.55)  # (x1, y1, x2, y2)

# How close a person can get to machinery before it's flagged, as a
# fraction of the image width — also scales with image size.
MACHINERY_PROXIMITY_FRACTION = 0.08

# Labels treated as "machinery". Includes COCO vehicle classes so proximity
# checks work with the stock model, plus your custom class names for best.pt.
MACHINERY_LABELS = {
    "machinery", "vehicle",                        # custom (best.pt)
    "car", "truck", "bus", "motorcycle", "train",  # COCO (yolov8n.pt)
}
PERSON_LABELS = {"Person", "person"}


def _run_yolo(frame, conf=0.25, imgsz=960):
    """Runs YOLOv8 on a single frame, returns a list of detections."""
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
        detections.append({"label": label, "confidence": round(conf, 2), "box": [x1, y1, x2, y2]})
    return detections


def _boxes_overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _box_distance(box_a, box_b):
    """Rough pixel distance between two box centers."""
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def _iou(box_a, box_b):
    """Intersection over Union — how much two boxes overlap, 0 to 1."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0

def find_violations(detections, frame_width, frame_height):
    """
    Turns raw YOLO detections into safety violations for a single photo.
    Restricted zone and machinery threshold are computed here, scaled to
    this specific image's actual size.
    """
    x1f, y1f, x2f, y2f = RESTRICTED_ZONE_FRACTION
    restricted_zone = (
        int(x1f * frame_width), int(y1f * frame_height),
        int(x2f * frame_width), int(y2f * frame_height),
    )
    machinery_threshold = MACHINERY_PROXIMITY_FRACTION * frame_width

    violations = []

    people = [d for d in detections if d["label"] in ("Person", "person")]
    hardhat = [d for d in detections if d["label"] == "Hardhat"]
    no_hardhat = [d for d in detections if d["label"] == "NO-Hardhat"]
    vest = [d for d in detections if d["label"] == "Safety Vest"]
    no_vest = [d for d in detections if d["label"] == "NO-Safety Vest"]
    machinery = [d for d in detections if d["label"] in ("machinery", "vehicle")]

    def overridden_by_positive(negative_det, positive_list, iou_threshold=0.3):
        for pos in positive_list:
            if _iou(negative_det["box"], pos["box"]) > iou_threshold and pos["confidence"] >= negative_det["confidence"]:
                return True
        return False

    for d in no_hardhat:
        if not overridden_by_positive(d, hardhat):
            violations.append({"type": "Helmet not detected", "severity": "high", "box": d["box"]})

    for d in no_vest:
        if not overridden_by_positive(d, vest):
            violations.append({"type": "Safety vest not detected", "severity": "medium", "box": d["box"]})

    for p in people:
        if _boxes_overlap(p["box"], restricted_zone):
            violations.append({"type": "Restricted zone entry", "severity": "medium", "box": p["box"]})
        for m in machinery:
            if _box_distance(p["box"], m["box"]) < machinery_threshold:
                violations.append({"type": "Unsafe proximity to machinery", "severity": "high", "box": p["box"]})

    return violations


def calculate_risk_score(violations):
    """Very simple weighted scoring — tune the weights as you see fit."""
    weights = {"high": 15, "medium": 8, "low": 3}
    penalty = sum(weights.get(v["severity"], 5) for v in violations)
    score = max(0, 100 - penalty)
    if score >= 80:
        level = "low"
    elif score >= 50:
        level = "medium"
    else:
        level = "high"
    return {"score": score, "level": level}


def _draw_annotations(frame, detections):
    annotated = frame.copy()
    used_positions = []

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = d["label"]
        text = f'{label} {d["confidence"]}'

        if label.startswith("NO-"):
            color = (60, 60, 255)
        elif label in ("Person", "person"):
            color = (100, 220, 60)
        else:
            color = (0, 190, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1 - 8, text_h + 4)

        collided = True
        attempts = 0
        while collided and attempts < 10:
            collided = False
            for (ux1, uy1, ux2, uy2) in used_positions:
                if abs(label_y - uy1) < text_h + 6 and not (x1 + text_w < ux1 or x1 > ux2):
                    label_y += text_h + 8
                    collided = True
                    break
            attempts += 1

        cv2.rectangle(annotated, (x1, label_y - text_h - 4), (x1 + text_w + 6, label_y + 4), color, -1)
        cv2.putText(annotated, text, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        used_positions.append((x1, label_y - text_h - 4, x1 + text_w + 6, label_y + 4))

    return annotated


def _encode_image_base64(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


def analyze_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    height, width = frame.shape[:2]
    detections = _run_yolo(frame)
    violations = find_violations(detections, width, height)
    risk = calculate_risk_score(violations)
    annotated = _draw_annotations(frame, detections)
    annotated_image_b64 = _encode_image_base64(annotated)
    return {
        "detections": detections,
        "violations": violations,
        "risk": risk,
        "annotated_image": annotated_image_b64,
    }


# ------------------------------------------------------------------
# FALL DETECTION (video path) — VERSION: FALL-FIX v4
# ------------------------------------------------------------------
# Every person is followed from frame to frame (camera movement is removed
# first), and three kinds of fall are recognised:
#
#   1. FALL TO THE GROUND - a person seen STANDING is then DOWN on the same
#      spot (body wider than tall / torso near horizontal, much lower than
#      when standing) and is no longer standing there.
#   2. FALL FROM HEIGHT - a person's whole body DROPS quickly: at least 1.2x
#      their own height within ~1 second, seen in 3+ frames in a row
#      (e.g. falling off a scaffold, ladder or roof edge).
#   4. THROWN / STRUCK - a person suddenly moves >= 1x their height (mostly
#      downward) within 2 s and ends up tilted / sprawled - e.g. hit by a
#      swinging load and knocked off a ledge.
#   3. WORKER SUSPENDED - a person hanging with the hands held up (holding a
#      rope / harness line) for most of at least 4 seconds after a fall.
#
# What is NOT a fall: walking, bending, crouching, sitting down, climbing,
# going down stairs, or the camera tilting/panning.
#
# Moments that look suspicious get a second look with the more accurate pose
# model, because a person lying on the ground is harder to see.
import math

KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12

POSE_FAST_MODEL = "yolov8s-pose.pt"       # every analysed frame (auto-downloads once if missing)
POSE_ACCURATE_MODEL = "yolov8m-pose.pt"   # second look at suspicious moments

POSE_IMGSZ = 640            # minimum model input size; tall/large frames use their long side (up to 1152)
POSE_IMGSZ_MAX = 1152
POSE_CONF = 0.15            # low on purpose: a person lying down / mid-fall scores lower
TARGET_SAMPLE_FPS = 5       # frames analysed per second of video
MAX_FRAME_LONG_SIDE = 1152   # big (HD/4K) frames are shrunk so their LONG side is at most this
MIN_FRAME_SHORT_SIDE = 540   # tiny frames are enlarged so small people are still found
REFINE_IMGSZ_SAME_MODEL = 960
REFINE_CONF = 0.10
REFINE_MAX_FRAMES = 24
MAX_FALL_IMAGES = 16

# 1. fall to the ground
UPRIGHT_MAX_ASPECT = 0.75
UPRIGHT_MAX_ANGLE = 35
DOWN_MIN_ASPECT = 1.3
DOWN_ANGLE = 60
DOWN_MAX_HEIGHT_RATIO = 0.6
DOWN_MIN_TOP_DROP = 0.35
DOWN_TORSO_ANGLE = 55        # torso at least this far from vertical to count as lying down
SAME_FALL_SEC = 10.0
STOOD_UP_SEC = 1.5
GROUND_MIN_PERSON_PX = 80     # a standing worker smaller than this (in the analysed frame) is too
                              # small to judge a fall to the ground reliably            # upright again on the same spot within this time = was only bending          # the same worker still down at the same spot = the same fall
DOWN_MIN_KEYPOINTS = 6        # a "person on the ground" must show a body (stops pipes, bags, mirrors)
DOWN_MIN_CONF = 0.15
FALL_LOOKBACK_SEC = 4.0
CONFIRM_WINDOW_SEC = 2.0
FALL_COOLDOWN_SEC = 4.0
TIP_WINDOW_SEC = 1.5
TIP_MIN_ANGLE = 45
TIP_MIN_TOP_DROP = 0.15
# 2. fall from height
DROP_MIN_BODY_HEIGHTS = 1.2   # whole body moves down by >= 1.2x its own height...
DROP_MAX_SEC = 1.2            # ...within this time
DROP_MIN_STEPS = 2            # ...seen across at least 3 frames, each lower than the last
DROP_MAX_GROWTH = 1.6         # (walking towards the camera makes a person BIGGER - not a fall)
DROP_MIN_SPEED = 2.0          # ...and at some moment faster than 2 body-heights per second
                              # (climbing down a ladder / stairs is much slower than that)
# 4. thrown / struck (knocked off balance, tumbling)
TUMBLE_MAX_SEC = 2.0          # moved within this time...
TUMBLE_MIN_MOVE = 1.0         # ...at least 1x its own height...
TUMBLE_MIN_DOWN = 0.5         # ...of which at least 0.5x downward...
TUMBLE_MIN_ANGLE = 45         # ...and ends up tilted / sprawled (walking & running stay upright)
TUMBLE_MIN_ASPECT = 1.1
GAP_DROP_MIN = 1.4            # ...OR the body drops >= 1.4x its height (even across missed frames)
GAP_DROP_MIN_SPEED = 0.9      # at >= 0.9 body-heights per second on average
GAP_DROP_SIZE_RANGE = (0.6, 1.5)   # and stays about the same size (walking towards a camera grows)
# 3. worker suspended (hanging)
HANG_MIN_SEC = 4.0
HANG_MIN_SEC_SHORT_CLIP = 2.5 # for clips shorter than ~5 s: 75% of the clip, at least 2.5 s
HANG_MIN_SHARE = 0.6          # hands-up in >= 60% of the frames of that stretch


def _load_pose(name):
    path = os.path.join(_HERE, name)
    if os.path.abspath(path) == os.path.abspath(POSE_MODEL_PATH):
        return pose_model, name           # already loaded at the top of this file
    try:
        return YOLO(path if os.path.exists(path) else name), name
    except Exception:
        return None, None


_fast_pose, _fast_name = _load_pose(POSE_FAST_MODEL)
if _fast_pose is None:                    # offline and no yolov8s-pose.pt: use the model loaded above
    _fast_pose, _fast_name = pose_model, os.path.basename(POSE_MODEL_PATH)
_accurate_pose = None


def _get_accurate_pose():
    global _accurate_pose
    if _accurate_pose is None:
        path = os.path.join(_HERE, POSE_ACCURATE_MODEL)
        if os.path.exists(path) and os.path.basename(path) != _fast_name:
            m, _ = _load_pose(POSE_ACCURATE_MODEL)
            _accurate_pose = (m, POSE_IMGSZ) if m is not None else (_fast_pose, REFINE_IMGSZ_SAME_MODEL)
        else:
            _accurate_pose = (_fast_pose, REFINE_IMGSZ_SAME_MODEL)
    return _accurate_pose


print("[SentinelSite] Fall detection: FALL-FIX v4 loaded (ground falls, falls from height, "
      f"thrown/struck workers, suspended workers) - pose model {_fast_name}"
      + (", second look yolov8m-pose.pt" if os.path.exists(os.path.join(_HERE, POSE_ACCURATE_MODEL)) else ""))


def _people_from(results, refined=False):
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return []          # the old code crashed here: empty keypoints come back as shape (1, 0, 51)
    kpts_all = results.keypoints.data if results.keypoints is not None else None
    return [{"box": [round(v) for v in boxes.xyxy[i].tolist()],
             "keypoints": kpts_all[i].tolist() if kpts_all is not None and i < len(kpts_all) else None,
             "conf": float(boxes.conf[i]), "refined": refined} for i in range(len(boxes))]


def _imgsz_for(frame):
    """Model input size = the frame's long side (rounded to 32), so tall phone videos
    are not shrunk before the model sees them."""
    long_side = max(frame.shape[:2])
    return int(min(POSE_IMGSZ_MAX, max(POSE_IMGSZ, round(long_side / 32) * 32)))


def _run_pose(frame, conf=POSE_CONF, imgsz=None):
    return _people_from(_fast_pose(frame, conf=conf, imgsz=imgsz or _imgsz_for(frame), verbose=False)[0])


def _run_pose_accurate(frame):
    m, _ = _get_accurate_pose()
    sz = max(_imgsz_for(frame), REFINE_IMGSZ_SAME_MODEL if m is _fast_pose else POSE_IMGSZ)
    return _people_from(m(frame, conf=REFINE_CONF, imgsz=sz, verbose=False)[0], refined=True)


def _kp(kpts, idx, conf_thresh=0.3):
    if not kpts or idx >= len(kpts):
        return None
    x, y, c = kpts[idx]
    return (x, y) if c >= conf_thresh else None


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _torso_angle_from_vertical(kpts):
    """0 deg = standing upright, 90 deg = lying flat. None if not visible."""
    shoulder = _midpoint(_kp(kpts, KP_L_SHOULDER), _kp(kpts, KP_R_SHOULDER))
    hip = _midpoint(_kp(kpts, KP_L_HIP), _kp(kpts, KP_R_HIP))
    if shoulder is None or hip is None:
        return None
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def _arms_up(kpts, h):
    """Hands raised as when hanging from / holding a rope: one hand well above the shoulders,
    the other not hanging down (or hidden)."""
    ls, rs = _kp(kpts, KP_L_SHOULDER), _kp(kpts, KP_R_SHOULDER)
    if ls is None or rs is None:
        return False
    shoulder_y = min(ls[1], rs[1])
    lifts = []
    for idx in (KP_L_WRIST, KP_R_WRIST):
        wpt = _kp(kpts, idx)
        lifts.append(None if wpt is None else (shoulder_y - wpt[1]) / h)   # > 0 means above the shoulders
    seen = [x for x in lifts if x is not None]
    if not seen or max(seen) < 0.12:
        return False
    return min(seen) > -0.08


def _posture(person, cam=(0.0, 0.0)):
    x1, y1, x2, y2 = person["box"]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    aspect = w / h
    kpts = person["keypoints"]
    angle = _torso_angle_from_vertical(kpts)
    nkp = sum(1 for k in (kpts or []) if k[2] >= 0.3)
    cx, cy = cam
    person.update({
        "w": w, "h": h, "angle": angle, "nkp": nkp,
        # positions with the camera's own movement removed
        "cx": (x1 + x2) / 2 - cx, "top": y1 - cy, "bottom": y2 - cy,
        "upright": aspect <= UPRIGHT_MAX_ASPECT and (angle is None or angle <= UPRIGHT_MAX_ANGLE)
                   and person["conf"] >= 0.3,
        # lying down: torso near horizontal when it can be seen (a kneeling / crouching worker's
        # torso stays fairly upright even though the box is wide); box shape only if torso unseen
        "down": ((angle is not None and angle >= DOWN_TORSO_ANGLE and aspect >= 0.9)
                 or (angle is None and aspect >= DOWN_MIN_ASPECT))
                and max(w, h) >= 24 and person["conf"] >= DOWN_MIN_CONF and nkp >= DOWN_MIN_KEYPOINTS,
        "arms_up": person["conf"] >= 0.4 and nkp >= 8 and _arms_up(kpts, h),
    })
    return person


def _matches_ref(down, ref):
    """Could this person on the ground be the worker who was standing at `ref`?"""
    H = ref["h"]
    return (abs(down["cx"] - ref["cx"]) <= 1.2 * H
            and abs(down["bottom"] - ref["bottom"]) <= 0.5 * H
            and down["h"] <= DOWN_MAX_HEIGHT_RATIO * H
            and (down["top"] - ref["top"]) >= DOWN_MIN_TOP_DROP * H)


def _still_standing(ref, people):
    rcx, rcy = ref["cx"], (ref["top"] + ref["bottom"]) / 2
    for p in people:
        if p["down"]:
            continue
        if p["upright"] and abs(p["cx"] - rcx) <= 0.3 * ref["h"] and abs(p["h"] - ref["h"]) <= 0.3 * ref["h"]:
            return True
        if (p["cx"] - p["w"] / 2 <= rcx <= p["cx"] + p["w"] / 2 and p["top"] <= rcy <= p["bottom"]
                and p["h"] >= 0.8 * ref["h"] and p["conf"] >= 0.3):
            return True
    return False


def _already_there(samples, down, ref, ref_t):
    """True if a person was ALREADY at the spot of `down` while `ref` was still standing elsewhere -
    then `down` is a different worker (e.g. one kneeling there all along), not `ref` after a fall."""
    tol = max(0.5 * down["w"], 0.3 * ref["h"])
    for s in samples:
        if not (ref_t - 1.0 <= s["t"] <= ref_t + 0.01):
            continue
        for q in s["people"]:
            if q is ref:
                continue
            if abs(q["cx"] - ref["cx"]) <= 0.2 * ref["h"] and abs(q["h"] - ref["h"]) <= 0.3 * ref["h"]:
                continue   # that is the standing worker himself
            if abs(q["cx"] - down["cx"]) <= tol and abs(q["bottom"] - down["bottom"]) <= 0.5 * ref["h"]:
                return True
    return False


def _iou_boxes(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def _camera_motion(grays):
    """Cumulative camera shift (x, y) for every sample, from frame-to-frame phase correlation."""
    out, cx, cy = [(0.0, 0.0)], 0.0, 0.0
    for a, b in zip(grays, grays[1:]):
        if a is None or b is None or a.shape != b.shape:
            out.append((cx, cy))
            continue
        (dx, dy), resp = cv2.phaseCorrelate(a, b)
        if resp > 0.05:
            cx += dx * _GRAY_SCALE
            cy += dy * _GRAY_SCALE
        out.append((cx, cy))
    return out


_GRAY_SCALE = 3.0   # grayscale copies for camera-motion estimation are 1/3 of the analysed size


def _assign_tracks(samples):
    """Links each person to the nearest similar person in earlier frames.
    Downward jumps are allowed to be larger, so a falling person keeps the same track."""
    next_id, live = 0, []
    for s in samples:
        used = set()
        for p in sorted(s["people"], key=lambda q: -q["conf"]):
            best, best_d = None, None
            pcy = (p["top"] + p["bottom"]) / 2
            for tr in live:
                if tr[0] in used or s["t"] - tr[2] > 1.6:
                    continue
                q = tr[1]
                scale = max(q["h"], q["w"], 1)
                dx = abs(p["cx"] - q["cx"]) / scale
                dy = (pcy - (q["top"] + q["bottom"]) / 2) / scale
                gap = s["t"] - tr[2]
                if dx > (0.9 if gap <= 0.4 else 1.2) or dy < -0.9 or dy > (1.8 if gap <= 0.4 else 2.6):
                    continue
                size = max(p["h"], p["w"]) / max(q["h"], q["w"], 1)
                if size > 2.2 or size < 0.4:
                    continue
                d = dx + abs(dy) * (0.5 if dy > 0 else 1.0)
                if best_d is None or d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = [next_id, p, s["t"]]
                next_id += 1
                live.append(best)
            p["track"] = best[0]
            used.add(best[0])
            best[1], best[2] = p, s["t"]


def _tracks(samples):
    by_track = {}
    for s in samples:
        for p in s["people"]:
            by_track.setdefault(p["track"], []).append((s, p))
    return by_track


def _tip_events(samples):
    tips = []
    for seq in _tracks(samples).values():
        for i, (su, u) in enumerate(seq):
            if not u["upright"]:
                continue
            for sv, v in seq[i + 1:]:
                if sv["t"] - su["t"] > TIP_WINDOW_SEC:
                    break
                tilted = (v["angle"] is not None and v["angle"] >= TIP_MIN_ANGLE) or v["w"] / v["h"] >= 1.0
                if tilted and v["top"] - u["top"] >= TIP_MIN_TOP_DROP * u["h"]:
                    tips.append({"t": sv["t"], "ref": u, "ref_t": su["t"]})
                    break
    return tips


def _vanish_events(samples, frame_w):
    """A person (seen in 2+ frames) who suddenly disappears, not at the side edges."""
    last_seen, count = {}, {}
    for s in samples:
        for p in s["people"]:
            if p["conf"] >= 0.3 or p.get("upright"):
                last_seen[p["track"]] = (s, p)
                count[p["track"]] = count.get(p["track"], 0) + 1
    out, end_t = [], samples[-1]["t"] if samples else 0
    for tr, (s, p) in last_seen.items():
        if count.get(tr, 0) < 2 or end_t - s["t"] < 0.5:
            continue
        if p["box"][0] < 0.03 * frame_w or p["box"][2] > 0.97 * frame_w:
            continue
        out.append({"t": s["t"], "ref": p, "ref_t": s["t"], "vanish": True})
    return out


def _sprawled(p):
    """Body clearly tilted or wider than tall, with a real body visible (not a pipe or bag)."""
    tilted = (p["angle"] is not None and p["angle"] >= TUMBLE_MIN_ANGLE) or p["w"] / p["h"] >= TUMBLE_MIN_ASPECT
    return tilted and p["nkp"] >= 8 and p["conf"] >= 0.25


def _left_the_spot(samples, sa, a, sb):
    """True if nobody is still standing where worker `a` was, at or after the moment `sb`
    (otherwise the person seen lower down is a DIFFERENT worker, not `a` after a fall)."""
    H = max(a["h"], a["w"])
    acy = (a["top"] + a["bottom"]) / 2
    for s in samples:
        if not (sb["t"] - 0.01 <= s["t"] <= sb["t"] + 0.6):
            continue
        for q in s["people"]:
            qcy = (q["top"] + q["bottom"]) / 2
            if (q["conf"] >= 0.3 and abs(q["cx"] - a["cx"]) <= 0.5 * H and abs(qcy - acy) <= 0.5 * H
                    and 0.6 <= max(q["h"], q["w"]) / H <= 1.6):
                return False
    return True


def _tumble_events(samples):
    """THROWN / STRUCK: within 2 s a tracked worker either
         (A) moves >= 1x their height (>= 0.5x downward) and ends up tilted / sprawled, or
         (B) drops >= 1.4x their height at >= 0.9 heights/s while staying about the same size
             (catches falls where the model missed the frames in the air),
       and nobody is left standing where the worker was (so it is the same worker)."""
    out = []
    for seq in _tracks(samples).values():
        found = False
        for i, (sa, a) in enumerate(seq):
            H = max(a["h"], a["w"])
            if H < 20 or a["conf"] < 0.3:
                continue
            for sb, b in seq[i + 1:]:
                dt = sb["t"] - sa["t"]
                if dt > TUMBLE_MAX_SEC:
                    break
                if b["conf"] < 0.2:
                    continue
                acy, bcy = (a["top"] + a["bottom"]) / 2, (b["top"] + b["bottom"]) / 2
                move = math.hypot(b["cx"] - a["cx"], bcy - acy) / H
                down = (bcy - acy) / H
                size = max(b["h"], b["w"]) / H
                rule_a = (move >= TUMBLE_MIN_MOVE and down >= TUMBLE_MIN_DOWN and size <= 2.0
                          and ((b["angle"] is not None and b["angle"] >= TUMBLE_MIN_ANGLE)
                               or b["w"] / b["h"] >= TUMBLE_MIN_ASPECT) and b["nkp"] >= 8)
                rule_b = (down >= GAP_DROP_MIN and down / max(dt, 1e-6) >= GAP_DROP_MIN_SPEED
                          and GAP_DROP_SIZE_RANGE[0] <= size <= GAP_DROP_SIZE_RANGE[1] and b["nkp"] >= 8)
                if (rule_a or rule_b) and _left_the_spot(samples, sa, a, sb):
                    out.append({"type": "tumble", "t": sb["t"], "start_t": sa["t"], "a": (sa, a), "b": (sb, b),
                                "track": a.get("track")})
                    found = True
                    break
            if found:
                break
    return out


def _drop_events(samples):
    """FALL FROM HEIGHT: a tracked person whose whole body drops >= 1.2x its height within ~1 s,
    each frame lower than the one before (camera movement already removed)."""
    out = []
    for seq in _tracks(samples).values():
        n = len(seq)
        for i in range(n):
            s0, p0 = seq[i]
            H = p0["h"]
            if H < 20 or p0["conf"] < 0.15:
                continue
            chain = [(s0, p0)]
            for j in range(i + 1, n):
                sj, pj = seq[j]
                if sj["t"] - s0["t"] > DROP_MAX_SEC:
                    break
                prev = chain[-1][1]
                if pj["top"] - prev["top"] < 0.1 * H:      # must keep going down
                    break
                chain.append((sj, pj))
            if len(chain) - 1 < DROP_MIN_STEPS:
                continue
            s1, p1 = chain[-1]
            drop_top = p1["top"] - p0["top"]
            drop_bottom = p1["bottom"] - p0["bottom"]
            fastest = max(((b[1]["top"] - a[1]["top"]) / H) / max(1e-6, b[0]["t"] - a[0]["t"])
                          for a, b in zip(chain, chain[1:]))
            if (min(drop_top, drop_bottom + 0.3 * H) >= DROP_MIN_BODY_HEIGHTS * H
                    and p1["h"] <= DROP_MAX_GROWTH * H and fastest >= DROP_MIN_SPEED):
                out.append({"type": "height", "t": s1["t"], "start_t": s0["t"], "chain": chain,
                            "drop": round(drop_top / H, 1)})
                break
    return out


def _hang_events(samples):
    """WORKER SUSPENDED: a tracked person with both arms up above the head for most of >= 4 s."""
    out = []
    clip_len = samples[-1]["t"] - samples[0]["t"] if samples else 0
    min_sec = min(HANG_MIN_SEC, max(HANG_MIN_SEC_SHORT_CLIP, 0.75 * clip_len))   # short clips: shorter stretch
    for seq in _tracks(samples).values():
        if len(seq) < 3:
            continue
        flags = [(s["t"], p["arms_up"], s, p) for s, p in seq]
        start = 0
        for end in range(len(flags)):
            while flags[end][0] - flags[start][0] > 12.0:
                start += 1
            span = flags[start:end + 1]
            dur = span[-1][0] - span[0][0]
            if dur >= min_sec and len(span) >= 8:
                share = sum(1 for f in span if f[1]) / len(span)
                if share >= HANG_MIN_SHARE and span[0][1]:
                    out.append({"type": "suspended", "t": span[0][0], "end_t": span[-1][0],
                                "frames": [(f[2], f[3]) for f in span if f[1]]})
                    break
        # extend to the full hanging stretch for the frame list
        if out and out[-1].get("frames") and out[-1]["frames"][0][1].get("track") == seq[0][1].get("track"):
            ev = out[-1]
            more = [(s, p) for s, p in seq if p["arms_up"] and s["t"] > ev["end_t"]]
            ev["frames"].extend(more)
            if more:
                ev["end_t"] = more[-1][0]["t"]
    return out


def _draw_pose_fall(frame, person, label, ref=None, cam=(0.0, 0.0)):
    """Annotate one frame: grey = where the worker stood, red = the worker now."""
    annotated = frame.copy()
    if ref is not None:
        rx1, ry1, rx2, ry2 = ref["box"]
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (200, 200, 200), 1)
        cv2.putText(annotated, "before", (rx1, max(14, ry1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    x1, y1, x2, y2 = person["box"]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 60, 255), 3)
    kpts = person.get("keypoints")
    for (x, y, c) in (kpts or []):
        if c >= 0.3:
            cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 255), -1)
    shoulder = _midpoint(_kp(kpts, KP_L_SHOULDER), _kp(kpts, KP_R_SHOULDER))
    hip = _midpoint(_kp(kpts, KP_L_HIP), _kp(kpts, KP_R_HIP))
    if shoulder and hip:
        cv2.line(annotated, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (60, 60, 255), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ly = max(th + 6, y1 - 8)
    lx = max(0, min(x1, annotated.shape[1] - tw - 10))
    cv2.rectangle(annotated, (lx, ly - th - 6), (lx + tw + 8, ly + 4), (60, 60, 255), -1)
    cv2.putText(annotated, label, (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated


def analyze_video(path, frame_skip=None):
    """
    Checks an uploaded video for worker FALLS (see the three rules above).
    Return shape is the same as before, so the dashboard and email alert work
    unchanged:
      fall_detected / fall_count       -> number of falls found
      fall_events + fall_frame_images  -> EVERY frame of each fall, one picture each
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video. Please upload an MP4 / MOV / AVI file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:
        fps = 30.0
    if frame_skip is None:
        frame_skip = max(1, round(fps / TARGET_SAMPLE_FPS))

    # ---- Pass 1: read the video once; fast pose model on every sampled frame
    samples, grays, frame_idx, frame_w = [], [], 0, MAX_FRAME_LONG_SIDE
    while True:
        if not cap.grab():
            break
        if frame_idx % frame_skip == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                if max(h, w) > MAX_FRAME_LONG_SIDE:
                    sc = MAX_FRAME_LONG_SIDE / max(h, w)
                    frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                elif min(h, w) < MIN_FRAME_SHORT_SIDE:
                    sc = min(MIN_FRAME_SHORT_SIDE / min(h, w), MAX_FRAME_LONG_SIDE / max(h, w))
                    frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)
                frame_w = frame.shape[1]
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (int(g.shape[1] / _GRAY_SCALE), int(g.shape[0] / _GRAY_SCALE))).astype("float32")
                grays.append(g)
                okj, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                samples.append({"t": frame_idx / fps, "frame_idx": frame_idx, "jpg": jpg if okj else None,
                                "people": _run_pose(frame)})
        frame_idx += 1
    cap.release()
    if not samples:
        raise ValueError("No frames could be read from this video. Try re-exporting it as an H.264 MP4.")

    cams = _camera_motion(grays)
    for s, cam in zip(samples, cams):
        s["cam"] = cam
        s["people"] = [_posture(p, cam) for p in s["people"]]

    def frame_of(s):
        return cv2.imdecode(s["jpg"], cv2.IMREAD_COLOR) if s["jpg"] is not None else None

    # ---- Pass 2: second look at suspicious moments (someone tipping over, lying, or vanishing)
    _assign_tracks(samples)
    tips = _tip_events(samples) + _vanish_events(samples, frame_w)
    windows = [((2 if tp.get("vanish") else 0), tp["t"], tp["t"] + 3.0) for tp in tips]
    for s in samples:
        if any(_sprawled(p) for p in s["people"]):
            windows.append((1, s["t"] - 1.0, s["t"] + 1.0))
    for i, s in enumerate(samples):
        for p in s["people"]:
            if p["down"] and any(_matches_ref(p, q) for s0 in samples[max(0, i - 25):i]
                                 for q in s0["people"] if q["upright"] and s["t"] - s0["t"] <= FALL_LOOKBACK_SEC):
                windows.append((1, s["t"] - 1.0, s["t"] + 2.0))
    windows.sort()
    merged = []
    for pr, a, b in windows:
        if merged and pr == merged[-1][0] and a <= merged[-1][2] + 0.01:
            merged[-1][2] = max(merged[-1][2], b)
        else:
            merged.append([pr, a, b])
    queues = [[s for s in samples if a <= s["t"] <= b] for _, a, b in merged]
    to_check, seen = [], set()
    while any(queues) and len(to_check) < REFINE_MAX_FRAMES:
        for q in queues:
            while q and id(q[0]) in seen:
                q.pop(0)
            if q and len(to_check) < REFINE_MAX_FRAMES:
                s0 = q.pop(0)
                seen.add(id(s0))
                to_check.append(s0)
    for s in to_check:
        img = frame_of(s)
        if img is None:
            continue
        extra = [_posture(p, s["cam"]) for p in _run_pose_accurate(img)]
        for p in extra:
            s["people"] = [q for q in s["people"] if _iou_boxes(q["box"], p["box"]) < 0.5]
        s["people"].extend(extra)
    if to_check:
        _assign_tracks(samples)
        tips = _tip_events(samples) + _vanish_events(samples, frame_w)

    # ---- Pass 3a: FALL TO THE GROUND
    ground, history, pending = [], [], []

    def tipped_before(c):
        return any(abs(tp["ref"]["cx"] - c["ref"]["cx"]) <= 0.5 * c["ref"]["h"]
                   and c["ref_t"] - 0.01 <= tp["t"] <= c["t"] + 0.01 for tp in tips)

    def confirm(c):
        for f in ground:
            if abs(c["t"] - f["t"]) < FALL_COOLDOWN_SEC:
                return
            if abs(c["t"] - f["t"]) < SAME_FALL_SEC and abs(c["down"]["cx"] - f["down"]["cx"]) <= 1.5 * f["ref"]["h"]:
                return   # same worker, same spot - not a new fall
        ground.append(c)

    for s in samples:
        t, people = s["t"], s["people"]
        for c in pending:
            if c["t"] < t <= c["t"] + CONFIRM_WINDOW_SEC and any(
                    p["down"] and _matches_ref(p, c["ref"]) for p in people):
                c["hits"] += 1
        for c in [c for c in pending if c["hits"] >= 2 or tipped_before(c)]:
            confirm(c)
        pending = [c for c in pending if c["hits"] < 2 and not tipped_before(c)
                   and t - c["t"] <= CONFIRM_WINDOW_SEC]
        history = [(ht, hp) for ht, hp in history if t - ht <= FALL_LOOKBACK_SEC]
        for p in people:
            if not p["down"]:
                continue
            # which standing worker could this be? not anyone who is still standing right now,
            # and of the rest the nearest one (then the most recent)
            refs = [(ht, hp) for ht, hp in history if hp["h"] >= GROUND_MIN_PERSON_PX
                    and _matches_ref(p, hp) and not _still_standing(hp, people)]
            if not refs:
                continue
            ref_t, ref = min(refs, key=lambda r: (round(abs(r[1]["cx"] - p["cx"]) / max(1.0, 0.25 * r[1]["h"])), -r[0]))
            if any(abs(t - f["t"]) < FALL_COOLDOWN_SEC for f in ground):
                continue
            if any(abs(c["down"]["cx"] - p["cx"]) < 0.5 * ref["h"] for c in pending):
                continue
            if _already_there(samples, p, ref, ref_t):
                continue   # someone else was already at that spot while this worker was standing
            c = {"type": "ground", "t": t, "down": p, "ref": ref, "ref_t": ref_t, "hits": 1, "sample": s}
            if tipped_before(c):
                confirm(c)
            else:
                pending.append(c)
        history.extend((t, p) for p in people if p["upright"])
    for c in pending:
        if samples[-1]["t"] - c["t"] <= 1.5 / TARGET_SAMPLE_FPS:
            confirm(c)

    def stood_straight_up(c):
        """A worker who is upright again on that spot within 1.5 s was bending over, not falling.
        (Ignores anything that was already standing there at the moment of the fall - another
        person or an object the model mistakes for a person.)"""
        ref = c["ref"]
        at_fall = [q for q in c["sample"]["people"] if q["upright"]]
        for s in samples:
            if c["t"] < s["t"] <= c["t"] + STOOD_UP_SEC:
                for q in s["people"]:
                    if not (q["upright"] and q["conf"] >= 0.3
                            and abs(q["cx"] - ref["cx"]) <= 0.3 * ref["h"]
                            and abs(q["h"] - ref["h"]) <= 0.3 * ref["h"]):
                        continue
                    if any(abs(q["cx"] - o["cx"]) <= 0.3 * o["h"] and abs(q["h"] - o["h"]) <= 0.3 * o["h"]
                           for o in at_fall):
                        continue
                    return True
        return False

    ground = [c for c in ground if not stood_straight_up(c)]

    # ---- Pass 3b: FALL FROM HEIGHT and 3c: WORKER SUSPENDED
    drops = _drop_events(samples)
    hangs = _hang_events(samples)
    tumbles = _tumble_events(samples)

    # one event per real fall: merge events that are the same incident (close in time)
    events = sorted(ground + drops + hangs + tumbles, key=lambda e: e["t"])
    falls = []
    for e in events:
        if falls and abs(e["t"] - falls[-1][-1]["t"]) < FALL_COOLDOWN_SEC:
            falls[-1].append(e)
        else:
            falls.append([e])

    # ---- every frame belonging to each fall, with a picture and a label
    flagged = []
    for n, group in enumerate(falls, start=1):
        frames = {}
        for e in group:
            if e["type"] == "ground":
                ref = e["ref"]
                for s in samples:
                    if not (e["ref_t"] < s["t"] <= e["t"] + 5.0):
                        continue
                    for p in s["people"]:
                        if p["down"] and _matches_ref(p, ref):
                            frames.setdefault(s["frame_idx"], (s, p, "worker on the ground", ref))
                            break
                        if s["t"] < e["t"] and abs(p["cx"] - ref["cx"]) <= 0.6 * ref["h"] and not p["upright"] \
                                and p["top"] - ref["top"] >= TIP_MIN_TOP_DROP * ref["h"]:
                            frames.setdefault(s["frame_idx"], (s, p, "worker falling", ref))
                            break
            elif e["type"] == "height":
                for k, (s, p) in enumerate(e["chain"]):
                    lab = "worker before the fall" if k == 0 else "worker falling from height"
                    frames.setdefault(s["frame_idx"], (s, p, lab, None))
                track = e["chain"][-1][1].get("track")
                for s in samples:   # what happens to that worker in the next 3 seconds
                    if e["t"] < s["t"] <= e["t"] + 3.0:
                        for p in s["people"]:
                            if p.get("track") == track:
                                frames.setdefault(s["frame_idx"], (s, p, "worker after the fall", None))
            elif e["type"] == "tumble":
                track = e["track"]
                for s in samples:   # the worker before, during and up to 5 s after being thrown
                    if e["start_t"] <= s["t"] <= e["t"] + 5.0:
                        for p in s["people"]:
                            if p.get("track") == track:
                                lab = ("worker before the incident" if s["t"] < e["t"] and not _sprawled(p)
                                       else "worker thrown / falling" if s["t"] <= e["t"] or _sprawled(p)
                                       else "worker after the fall")
                                frames.setdefault(s["frame_idx"], (s, p, lab, None))
                                break
            else:
                for s, p in e["frames"]:
                    frames.setdefault(s["frame_idx"], (s, p, "worker suspended / hanging", None))
        for fidx in sorted(frames):
            s, p, what, ref = frames[fidx]
            ang = round(p["angle"]) if p.get("angle") is not None else 0
            flagged.append((n, s, p, f"FALL {n}: {what}" + (f" {ang} deg" if p.get("angle") is not None else ""), ref, ang))

    if len(flagged) > MAX_FALL_IMAGES:   # many frames: keep pictures spread evenly
        step = len(flagged) / MAX_FALL_IMAGES
        flagged = [flagged[int(i * step)] for i in range(MAX_FALL_IMAGES)]

    fall_events, fall_frame_images = [], []
    for n, s, p, label, ref, ang in flagged:
        img = frame_of(s)
        fall_events.append({
            "frame": s["frame_idx"],
            "timestamp_sec": round(s["t"], 1),
            "torso_angle": ang,
            "fall_number": n,
            "label": label,
        })
        fall_frame_images.append(_encode_image_base64(_draw_pose_fall(img, p, label, ref)) if img is not None else None)

    kind_names = {"ground": "fall to the ground", "height": "fall from height", "suspended": "worker suspended / hanging",
                  "tumble": "worker thrown / struck"}
    summary = [{"fall_number": n, "timestamp_sec": round(g[0]["t"], 1),
                "kind": ", ".join(sorted({kind_names[e["type"]] for e in g}))} for n, g in enumerate(falls, start=1)]

    return {
        "frames_analyzed": len(samples),
        "frame_skip": frame_skip,
        "fps": round(fps, 1),
        "fall_detected": len(falls) > 0,
        "fall_count": len(falls),                 # number of falls (one per incident)
        "fall_events": fall_events,               # one entry per flagged frame (matches the images)
        "fall_frame_images": fall_frame_images,
        "falls": summary,
        # kept for backwards-compatibility with older frontend code:
        "fall_event": fall_events[0] if fall_events else None,
        "fall_frame_image": fall_frame_images[0] if fall_frame_images else None,
    }



if __name__ == "__main__":
    import sys
    print(f"Model loaded OK from: {MODEL_PATH}")
    print(f"Known class names: {list(model.names.values())[:15]} ...")
    if len(sys.argv) > 1:
        out = analyze_image(sys.argv[1])
        out.pop("annotated_image", None)  # don't dump base64 to the terminal
        print("Detections:", len(out["detections"]))
        print("Violations:", out["violations"])
        print("Risk:", out["risk"])
    else:
        print("Pass an image path to test detection, e.g.:  python detect.py uploads/photo.jpg")