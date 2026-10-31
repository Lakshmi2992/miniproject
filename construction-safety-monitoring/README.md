# SentinelSite — AI-Based Construction Site Safety Monitoring System

This is the full project, consolidated in one place, with every update we
agreed on:
- Photo upload for PPE, restricted-zone, and machinery-proximity checks.
- Short video upload for fall detection (not live CCTV).
- Renamed to **SentinelSite**.
- Architecture: a web application with a Flask REST API backend.

---

## 1. What this project actually is

A supervisor uploads either a **photo** (checked instantly for PPE
violations, restricted-zone entry, and unsafe machinery distance) or a
**short video clip, 10–30 seconds** (checked frame-by-frame for a fall).
The system then shows the results on a dashboard: detected violations, a
site risk score, and a running log for reporting.

There is **no live camera streaming** anywhere in this project. Everything
is upload-then-process, which is what makes this realistically buildable
and demoable in 3 months by one person.

## 2. Why photo AND video, not just one

| Feature | Needs motion? | Input used |
|---|---|---|
| Helmet / vest detection | No — a single frame shows PPE state | Photo |
| Restricted zone entry | No — a single frame shows position | Photo |
| Machinery proximity | No — a single frame shows distance | Photo |
| Fall detection | **Yes** — a fall is a change over time | Short video clip |

A fall cannot be seen in one photo, so fall detection is the one feature
that needs multiple consecutive frames. A short uploaded video clip gives
you that, without any of the real-time infrastructure that live CCTV would
require (persistent connections, frame buffering, low-latency streaming to
a browser). Uploading a video is technically no different from uploading a
photo — just processed frame-by-frame with OpenCV after the upload
finishes, not while it's happening.

**How fall detection works here (see `detect.py`):**
1. Extract frames from the uploaded clip (every 3rd frame, to keep it fast).
2. Run the same YOLOv8 person detector you're already using for PPE checks
   — no second model needed.
3. Track the person's bounding box shape and vertical position across
   frames: a standing person's box is tall-and-narrow; a fallen person's
   box is wide-and-flat, and the box's center drops sharply.
4. If both change within a short window of frames → flag a fall event.

This is a rule applied on top of your existing detector, not a separate AI
model to train — it keeps fall detection achievable without extra
datasets or extra training time.

## 3. Web, API, or both?

**Both, and that's the normal architecture, not a decision you need to make
between them.** SentinelSite is:
- A **web application** — the dashboard the supervisor actually sees and
  clicks (`dashboard.html`).
- With a **REST API backend** — the Flask endpoints (`/api/analyze`,
  `/api/analyze-video`, `/api/violations`) that the dashboard calls
  internally to talk to your Python/YOLOv8 detection logic.

For your synopsis or viva: describe it as *"a web-based application with a
Flask REST API backend."* You're not choosing one over the other — the API
is simply how your own frontend talks to your own backend.

## 4. Where to start (platform setup)

1. **Local dev**: Python 3.10+, VS Code.
2. **Version control**: create a GitHub repo (`sentinelsite`), commit after
   every sprint.
3. **Model training**: Google Colab (free GPU) — don't train locally on a
   laptop CPU, it'll be painfully slow.
4. **Dataset**:
   - Kaggle (direct download): https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
   - Same dataset on Roboflow Universe (lets you export/augment further):
     https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
   - ~2,800 images, already in YOLOv8 format, with exactly the classes you
     need: `Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, Mask,
     NO-Mask, machinery, vehicle, Safety Cone`.
5. **Database**: start with SQLite for development, move to MySQL later —
   don't fight DB config in week 1.

## 5. Project structure

```
sentinelsite/
├── frontend/
│   ├── dashboard.html          <- the UI (Sprint 1 deliverable)
│   └── login.html               <- login page
├── backend/
│   ├── app.py                  <- Flask server + API endpoints + login
│   ├── detect.py                <- YOLOv8 detection + fall-detection logic
│   ├── db.py                     <- SQLite database (users, violations, uploads)
│   ├── requirements.txt
│   ├── sentinelsite.db            <- created automatically on first run
│   ├── models/                  <- your fine-tuned best.pt goes here (Sprint 2)
│   └── uploads/                 <- uploaded photos/videos land here
├── datasets/                     <- downloaded Kaggle/Roboflow dataset
├── notebooks/                     <- your Colab training notebook
└── README.md                      <- this file
```

**Default login (created automatically the first time you run `app.py`):**
No default login credentials are shipped. Create an administrator using the instructions below.
(Change this in `db.py` before showing it to anyone outside your own testing.)

## 6. How to run it

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** — this serves the dashboard through your own
Flask app. Try uploading a photo or a short video clip; if your model isn't
fine-tuned yet, the pipeline still runs end-to-end (it just won't recognize
helmet/vest classes until Sprint 2's training step is done — that's
expected at this stage, not a bug).

## 6b. Sprint 1 checklist — what's actually done now

| Item | Status |
|---|---|
| Flask + frontend setup | ✅ Done |
| Login page | Requires an administrator account created with `make_admin.py` |
| Photo upload + preview | ✅ Done — shows the real uploaded photo with real bounding boxes, not a mock |
| SQLite database | ✅ Done — `sentinelsite.db`, created automatically, survives restarts |
| MySQL | ⏳ Intentionally deferred — SQLite now, swap later (see `db.py` comment) |

## 7. Full sprint planner (6 sprints, 2 weeks each = 3 months)

**System modules**

| Module | Features |
|---|---|
| 1. Upload Module | Photo Upload, Video Upload, Upload History |
| 2. Detection Module | PPE Detection, Restricted Zone Check, Machinery Proximity Check |
| 3. Fall Detection Module | Frame Extraction, Posture Tracking, Fall Event Flagging |
| 4. Risk Assessment Module | Violation Classification, Risk Score, Compliance Score |
| 5. Reporting Module | Automated Safety Report, Report History, Export Report |
| 6. Dashboard Module | Supervisor Dashboard, Violation Feed, Compliance Charts |
| 7. Admin Module | Zone/Threshold Config, User Management, Analytics |

**User stories**

| ID | User Story |
|---|---|
| US01 | As a supervisor, I want to log in so that I can securely access the dashboard. |
| US02 | As a supervisor, I want to upload a site photo so that PPE and zone violations are detected. |
| US03 | As the system, I want to detect helmets/vests in a photo so that PPE violations are flagged. |
| US04 | As the system, I want to check restricted-zone entry and machinery distance so that unsafe positioning is flagged. |
| US05 | As a supervisor, I want to upload a short video so that a fall can be detected across frames. |
| US06 | As a supervisor, I want a calculated risk score so that I understand overall site safety. |
| US07 | As a supervisor, I want an automatically generated safety report so that I can document compliance. |
| US08 | As a supervisor, I want to view past uploads and violations so that I can track trends over time. |
| US09 | As an admin, I want to configure zones/thresholds so that detection stays accurate per site. |
| US10 | As an admin, I want to view analytics across all uploads so that I can evaluate platform usage. |

**Sprint planner**

| Sprint | Sprint Goal | User Stories | Deliverables |
|---|---|---|---|
| Sprint 1 | Project Setup & Upload Module | US01–US02 | Flask + dashboard UI, DB setup, photo upload + preview, project renamed to SentinelSite |
| Sprint 2 | PPE Detection | US03 | Fine-tuned YOLOv8 model on the Kaggle/Roboflow dataset, helmet/vest detection working on uploaded photos |
| Sprint 3 | Zone, Machinery & Fall Detection | US04–US05 | Restricted-zone check, machinery-proximity check, video upload + frame-based fall detection |
| Sprint 4 | Risk Scoring & Reporting | US06–US07 | Risk score algorithm, compliance score, automated report generation |
| Sprint 5 | Dashboard & History | US08 | Dashboard wired to real API data (not mock), upload history, violation feed |
| Sprint 6 | Admin Module & Final Integration | US09–US10 | Admin panel, analytics, testing, bug fixes, documentation |

## 8. What to show in your Sprint 1 demo

1. Open the dashboard — walk through each panel and explain what feeds it
   (photo upload → PPE/zone/machinery checks; video upload → fall check;
   risk gauge ← risk-scoring model in Sprint 4).
2. Run `app.py` and show the dashboard being served through your own
   backend, not just opened as a static file.
3. Upload a sample photo and a short test video to show both pipelines
   responding (even with the stock model, this proves upload → detect →
   respond works end-to-end).
4. Mention next sprint's goal: fine-tune YOLOv8 on the PPE dataset so
   helmet/vest classes are actually recognized.

## Administrator setup

Set `SENTINEL_ADMIN_PASSWORD` to a unique password of at least 12 characters, then run `python make_admin.py` from the backend directory to create the administrator account. Do not commit secrets or local database files.

For deployment, set a long random `FLASK_SECRET_KEY` environment variable. The app uses a temporary random key if this is not configured.
