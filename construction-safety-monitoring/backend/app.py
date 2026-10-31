import os
import csv
import io
from functools import wraps
from flask import Flask, jsonify, request, session, redirect, send_from_directory, Response
from werkzeug.utils import secure_filename

import detect
import db

app = Flask(__name__, static_folder="../frontend")
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

db.init_db()


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        # If an admin disabled/deleted this account, kick them out immediately.
        user = db.get_user_by_name(session["user"])
        if not user or user.get("active") == 0:
            session.clear()
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    """Only users whose role is 'admin' (checked against the DB every time,
    so demoting someone takes effect immediately)."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        is_api = request.path.startswith("/api/")
        if "user" not in session:
            return (jsonify({"error": "login required"}), 401) if is_api else redirect("/login")
        user = db.get_user_by_name(session["user"])
        if not user or user.get("active") == 0 or user.get("role") != "admin":
            return (jsonify({"error": "admin access required"}), 403) if is_api else redirect("/")
        return f(*args, **kwargs)
    return wrapper


def current_username():
    return session.get("user", "")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if db.check_user(username, password):
            user = db.get_user_by_name(username)
            session["user"] = user["username"]
            session["role"] = user.get("role") or "supervisor"
            db.touch_login(user["username"])
            db.log_audit(user["username"], "login", "")
            # admins land on the admin console, everyone else on the dashboard
            return redirect("/admin" if session["role"] == "admin" else "/")
        return redirect("/login?error=1")
    return app.send_static_file("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if db.username_exists(username):
            return redirect("/register?error=taken")
        db.create_user(username, password)  # role defaults to 'supervisor'
        session["user"] = username
        session["role"] = "supervisor"
        return redirect("/")
    return app.send_static_file("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def dashboard():
    return app.send_static_file("dashboard.html")


@app.route("/db-view")
@login_required
def db_view_page():
    return app.send_static_file("db_view.html")


@app.route("/violations")
@login_required
def violations_page():
    return app.send_static_file("violations.html")


@app.route("/reports")
@login_required
def reports_page():
    return app.send_static_file("reports.html")


@app.route("/api/reports-summary")
@login_required
def reports_summary():
    violations = db.get_recent_violations(limit=1000)
    uploads = db.get_all_uploads(limit=1000)

    severity_counts = {"high": 0, "medium": 0, "low": 0}
    type_counts = {}
    for v in violations:
        severity_counts[v["severity"]] = severity_counts.get(v["severity"], 0) + 1
        type_counts[v["type"]] = type_counts.get(v["type"], 0) + 1

    photo_uploads = len([u for u in uploads if u["upload_type"] == "photo"])
    video_uploads = len([u for u in uploads if u["upload_type"] == "video"])

    return jsonify({
        "total_uploads": len(uploads),
        "total_violations": len(violations),
        "photo_uploads": photo_uploads,
        "video_uploads": video_uploads,
        "severity_counts": severity_counts,
        "type_counts": type_counts,
    })


@app.route("/api/db-view")
@login_required
def db_view_api():
    return jsonify({
        "users": db.get_all_users(),
        "violations": db.get_recent_violations(limit=50),
        "uploads": db.get_all_uploads(),
    })


@app.route("/api/current-user")
@login_required
def current_user():
    user = db.get_user_by_name(session.get("user", "")) or {}
    return jsonify({"username": session.get("user", ""), "role": user.get("role", "supervisor")})


@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze_photo():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    result = detect.analyze_image(path)

    total_people = len([d for d in result["detections"] if d["label"] in ("Person", "person")])
    ppe_violation_types = {"Helmet not detected", "Safety vest not detected"}
    ppe_violation_count = len([v for v in result["violations"] if v["type"] in ppe_violation_types])

    db.log_upload(filename, "photo", total_people=total_people, ppe_violations=ppe_violation_count)
    for v in result["violations"]:
        db.log_violation(v["type"], v["severity"], "photo")

    return jsonify(result)


@app.route("/api/analyze-video", methods=["POST"])
@login_required
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "no video file provided"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        result = detect.analyze_video(path)
    except Exception as e:
        app.logger.exception("Video analysis failed for %s", filename)
        return jsonify({
            "error": "Video analysis failed. Check that the pose model is installed and the uploaded video is valid.",
            "detail": str(e),
        }), 500

    db.log_upload(filename, "video")
    if result["fall_detected"]:
        db.log_violation("Fall alert", "high", "video")

    return jsonify(result)


@app.route("/api/violations")
@login_required
def get_violations():
    limit = request.args.get("limit", 20, type=int)
    within_hours = request.args.get("within_hours", type=int)
    if within_hours is not None:
        return jsonify(db.get_recent_violations_within_hours(hours=within_hours, limit=limit))
    return jsonify(db.get_recent_violations(limit=limit))

@app.route("/api/fall-alerts-today")
@login_required
def fall_alerts_today():
    alerts = db.get_today_fall_alerts()
    latest = alerts[0] if alerts else None
    return jsonify({"count": len(alerts), "latest": latest})

@app.route("/api/site-risk")
@login_required
def site_risk():
    violations = db.get_violations_last_24h()
    weights = {"high": 8, "medium": 4, "low": 2}
    penalty = sum(weights.get(v["severity"], 3) for v in violations)
    score = max(0, 100 - penalty)

    if score >= 80:
        level = "low"
    elif score >= 50:
        level = "medium"
    else:
        level = "high"

    return jsonify({"score": score, "level": level, "active_violations": len(violations)})

@app.route("/api/ppe-compliance")
@login_required
def ppe_compliance():
    return jsonify(db.get_ppe_compliance_last_24h())

@app.route("/api/weekly-compliance")
@login_required
def weekly_compliance():
    return jsonify(db.get_weekly_ppe_compliance())

# ======================================================================
# ADMIN CONSOLE
# ======================================================================
UPLOAD_DIR_ABS = os.path.abspath(UPLOAD_DIR)


@app.route("/admin")
@admin_required
def admin_page():
    return app.send_static_file("admin.html")


@app.route("/admin/files/<path:filename>")
@admin_required
def admin_file(filename):
    # send_from_directory blocks path traversal (../) for us
    return send_from_directory(UPLOAD_DIR_ABS, filename)


@app.route("/api/admin/overview")
@admin_required
def admin_overview():
    return jsonify(db.get_admin_overview(UPLOAD_DIR_ABS))


# ---------- users ----------
@app.route("/api/admin/users")
@admin_required
def admin_users():
    return jsonify({"users": db.list_users_detailed(), "me": current_username()})


@app.route("/api/admin/users", methods=["POST"])
@admin_required
def admin_create_user():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    role = data.get("role") or "supervisor"

    if not username or len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if role not in ("admin", "supervisor"):
        return jsonify({"error": "Invalid role"}), 400
    if db.username_exists(username):
        return jsonify({"error": "That username is already taken"}), 409

    db.create_user(username, password, role)
    db.log_audit(current_username(), "create_user", f"{username} ({role})")
    return jsonify({"ok": True})


def _target_user_or_404(user_id):
    user = db.get_user_by_id(user_id)
    if not user:
        return None, (jsonify({"error": "User not found"}), 404)
    return user, None


@app.route("/api/admin/users/<int:user_id>/role", methods=["POST"])
@admin_required
def admin_set_role(user_id):
    user, err = _target_user_or_404(user_id)
    if err: return err
    role = (request.get_json(silent=True) or {}).get("role")
    if role not in ("admin", "supervisor"):
        return jsonify({"error": "Invalid role"}), 400
    if user["username"] == current_username() and role != "admin":
        return jsonify({"error": "You can't remove your own admin role"}), 400
    if user.get("role") == "admin" and role != "admin" and db.count_active_admins() <= 1:
        return jsonify({"error": "Can't demote the last admin"}), 400
    db.set_user_role(user_id, role)
    db.log_audit(current_username(), "set_role", f"{user['username']} -> {role}")
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:user_id>/active", methods=["POST"])
@admin_required
def admin_set_active(user_id):
    user, err = _target_user_or_404(user_id)
    if err: return err
    active = bool((request.get_json(silent=True) or {}).get("active"))
    if not active:
        if user["username"] == current_username():
            return jsonify({"error": "You can't disable your own account"}), 400
        if user.get("role") == "admin" and db.count_active_admins() <= 1:
            return jsonify({"error": "Can't disable the last active admin"}), 400
    db.set_user_active(user_id, active)
    db.log_audit(current_username(), "enable_user" if active else "disable_user", user["username"])
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:user_id>/reset-password", methods=["POST"])
@admin_required
def admin_reset_password(user_id):
    user, err = _target_user_or_404(user_id)
    if err: return err
    new_password = (request.get_json(silent=True) or {}).get("password") or ""
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    db.set_user_password(user_id, new_password)
    db.log_audit(current_username(), "reset_password", user["username"])
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def admin_delete_user(user_id):
    user, err = _target_user_or_404(user_id)
    if err: return err
    if user["username"] == current_username():
        return jsonify({"error": "You can't delete your own account"}), 400
    if user.get("role") == "admin" and db.count_active_admins() <= 1:
        return jsonify({"error": "Can't delete the last admin"}), 400
    db.delete_user(user_id)
    db.log_audit(current_username(), "delete_user", user["username"])
    return jsonify({"ok": True})


# ---------- violations ----------
@app.route("/api/admin/violations")
@admin_required
def admin_violations():
    severity = request.args.get("severity") or None
    limit = request.args.get("limit", 200, type=int)
    return jsonify(db.get_violations_admin(limit=limit, severity=severity))


@app.route("/api/admin/violations/<int:violation_id>", methods=["DELETE"])
@admin_required
def admin_delete_violation(violation_id):
    if not db.delete_violation(violation_id):
        return jsonify({"error": "Violation not found"}), 404
    db.log_audit(current_username(), "delete_violation", f"id {violation_id}")
    return jsonify({"ok": True})


@app.route("/api/admin/violations", methods=["DELETE"])
@admin_required
def admin_clear_violations():
    n = db.clear_violations()
    db.log_audit(current_username(), "clear_violations", f"{n} rows removed")
    return jsonify({"ok": True, "removed": n})


@app.route("/api/admin/export/violations.csv")
@admin_required
def admin_export_violations():
    rows = db.get_violations_admin(limit=100000)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "type", "severity", "source", "timestamp"])
    for r in rows:
        writer.writerow([r["id"], r["type"], r["severity"], r["source"], r["timestamp"]])
    db.log_audit(current_username(), "export_violations", f"{len(rows)} rows")
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )


# ---------- uploads ----------
@app.route("/api/admin/uploads")
@admin_required
def admin_uploads():
    return jsonify(db.get_all_uploads(limit=500))


def _remove_upload(upload):
    """Delete the DB row, and the file on disk if no other row still uses it
    (re-uploading the same filename overwrites the file, so rows can share one)."""
    db.delete_upload_record(upload["id"])
    if db.count_uploads_with_filename(upload["filename"]) == 0:
        path = os.path.join(UPLOAD_DIR_ABS, upload["filename"])
        if os.path.isfile(path):
            os.remove(path)
            return True
    return False


@app.route("/api/admin/uploads/<int:upload_id>", methods=["DELETE"])
@admin_required
def admin_delete_upload(upload_id):
    upload = db.get_upload(upload_id)
    if not upload:
        return jsonify({"error": "Upload not found"}), 404
    file_removed = _remove_upload(upload)
    db.log_audit(current_username(), "delete_upload", upload["filename"])
    return jsonify({"ok": True, "file_removed": file_removed})


@app.route("/api/admin/uploads/cleanup", methods=["POST"])
@admin_required
def admin_cleanup_uploads():
    days = (request.get_json(silent=True) or {}).get("days", 30)
    try:
        days = max(0, int(days))
    except (TypeError, ValueError):
        return jsonify({"error": "days must be a number"}), 400
    old = db.get_uploads_older_than(days)
    files_removed = sum(1 for u in old if _remove_upload(u))
    db.log_audit(current_username(), "cleanup_uploads", f"older than {days}d: {len(old)} records, {files_removed} files")
    return jsonify({"ok": True, "records_removed": len(old), "files_removed": files_removed})



# ---------- audit log ----------
@app.route("/api/admin/audit")
@admin_required
def admin_audit():
    return jsonify(db.get_audit(limit=200))


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)  