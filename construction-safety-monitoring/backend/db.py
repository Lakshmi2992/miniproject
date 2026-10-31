import sqlite3
import os
import shutil
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "sentinelsite.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        severity TEXT NOT NULL,
        source TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_type TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )""")
    
    # Migration: add PPE-tracking columns if they don't exist yet. SQLite
    # has no "ADD COLUMN IF NOT EXISTS", so we just ignore the error if
    # the column is already there — safe to run every startup.
    for col_def in ["total_people INTEGER DEFAULT 0", "ppe_violations INTEGER DEFAULT 0"]:
        try:
            conn.execute(f"ALTER TABLE uploads ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Migration: role / active / created_at / last_login on users (admin feature).
    # Old users automatically become role='supervisor', active=1.
    for col_def in ["role TEXT DEFAULT 'supervisor'", "active INTEGER DEFAULT 1",
                    "created_at TEXT", "last_login TEXT"]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        detail TEXT,
        timestamp TEXT NOT NULL
    )""")

    conn.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""")
    # No default accounts are created. Create an admin explicitly with make_admin.py.
    conn.commit()
    conn.close()


def check_user(username, password):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return False
    return row["active"] != 0  # disabled accounts cannot log in


def username_exists(username):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row is not None


def create_user(username, password, role="supervisor"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO users (username, password_hash, role, active, created_at) VALUES (?, ?, ?, 1, ?)",
        (username, generate_password_hash(password), role, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_all_users():
    conn = get_conn()
    rows = conn.execute("SELECT id, username FROM users ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_uploads(limit=50):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM uploads ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_violation(v_type, severity, source):
    conn = get_conn()
    conn.execute(
        "INSERT INTO violations (type, severity, source, timestamp) VALUES (?, ?, ?, ?)",
        (v_type, severity, source, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_violations(limit=20):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows][::-1]


def get_recent_violations_within_hours(hours=2, limit=20):
    """Only violations logged in the last `hours` hours (newest-first in DB,
    reversed to oldest-first to match get_recent_violations)."""
    conn = get_conn()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT * FROM violations WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
        (cutoff, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows][::-1]


def log_upload(filename, upload_type, total_people=0, ppe_violations=0):
    conn = get_conn()
    conn.execute(
        "INSERT INTO uploads (filename, upload_type, timestamp, total_people, ppe_violations) VALUES (?, ?, ?, ?, ?)",
        (filename, upload_type, datetime.now().isoformat(), total_people, ppe_violations),
    )
    conn.commit()
    conn.close()
    
def get_ppe_compliance_last_24h():
    conn = get_conn()
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

    rows = conn.execute(
        """
        SELECT total_people, ppe_violations
        FROM uploads
        WHERE upload_type = ?
          AND timestamp >= ?
        """,
        ("photo", cutoff),
    ).fetchall()

    conn.close()

    total_people = sum(r["total_people"] or 0 for r in rows)
    total_violations = sum(r["ppe_violations"] or 0 for r in rows)

    if total_people == 0:
        return {
            "compliance": None,
            "total_people": 0
        }

    total_ppe_checks = total_people * 2
    passed_ppe_checks = max(0, total_ppe_checks - total_violations)

    compliance = round(
        (passed_ppe_checks / total_ppe_checks) * 100
    )

    return {
        "compliance": compliance,
        "total_people": total_people
    }

def get_weekly_ppe_compliance():
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())
    sunday_next = monday + timedelta(days=7)

    conn = get_conn()

    # Number of uploads for each day
    upload_rows = conn.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day,
               COUNT(*) AS upload_count
        FROM uploads
        WHERE timestamp >= ?
          AND timestamp < ?
        GROUP BY substr(timestamp, 1, 10)
        """,
        (monday.isoformat(), sunday_next.isoformat()),
    ).fetchall()

    # Violations for each day
    violation_rows = conn.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day,
               severity,
               COUNT(*) AS count
        FROM violations
        WHERE timestamp >= ?
          AND timestamp < ?
        GROUP BY substr(timestamp, 1, 10), severity
        """,
        (monday.isoformat(), sunday_next.isoformat()),
    ).fetchall()

    conn.close()

    uploads_by_day = {
        row["day"]: row["upload_count"]
        for row in upload_rows
    }

    violations_by_day = {}

    for row in violation_rows:
        day = row["day"]

        if day not in violations_by_day:
            violations_by_day[day] = {
                "high": 0,
                "medium": 0,
                "low": 0
            }

        violations_by_day[day][row["severity"]] = row["count"]

    weights = {
        "high": 30,
        "medium": 15,
        "low": 5
    }

    result = []

    for i in range(7):
        day = monday + timedelta(days=i)
        key = day.isoformat()

        upload_count = uploads_by_day.get(key, 0)
        day_violations = violations_by_day.get(
            key,
            {"high": 0, "medium": 0, "low": 0}
        )

        if upload_count == 0:
            compliance = None

        else:
            total_penalty = (
                day_violations["high"] * weights["high"] +
                day_violations["medium"] * weights["medium"] +
                day_violations["low"] * weights["low"]
            )

            average_penalty = total_penalty / upload_count

            compliance = round(
                max(0, min(100, 100 - average_penalty))
            )

        result.append({
            "day": day.strftime("%a"),
            "date": key,
            "compliance": compliance,
            "uploads": upload_count,
            "high": day_violations["high"],
            "medium": day_violations["medium"],
            "low": day_violations["low"]
        })

    return result
    
def get_today_fall_alerts():
    conn = get_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT * FROM violations WHERE type = ? AND timestamp LIKE ? ORDER BY id DESC",
        ("Fall detected", today + "%"),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_violations_last_24h():
    conn = get_conn()
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    rows = conn.execute(
        "SELECT * FROM violations WHERE timestamp >= ? ORDER BY id DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ======================================================================
# ADMIN HELPERS
# ======================================================================
def get_user_by_name(username):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def touch_login(username):
    conn = get_conn()
    conn.execute("UPDATE users SET last_login = ? WHERE username = ?",
                 (datetime.now().isoformat(), username))
    conn.commit()
    conn.close()


def list_users_detailed():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, COALESCE(role,'supervisor') AS role, "
        "COALESCE(active,1) AS active, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_active_admins():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND COALESCE(active,1) = 1").fetchone()[0]
    conn.close()
    return n


def set_user_role(user_id, role):
    conn = get_conn()
    conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    conn.close()


def set_user_active(user_id, active):
    conn = get_conn()
    conn.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
    conn.commit()
    conn.close()


def set_user_password(user_id, new_password):
    conn = get_conn()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                 (generate_password_hash(new_password), user_id))
    conn.commit()
    conn.close()


def delete_user(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_violations_admin(limit=200, severity=None):
    conn = get_conn()
    if severity:
        rows = conn.execute(
            "SELECT * FROM violations WHERE severity = ? ORDER BY id DESC LIMIT ?",
            (severity, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_violation(violation_id):
    conn = get_conn()
    cur = conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
    conn.commit()
    conn.close()
    return cur.rowcount


def clear_violations():
    conn = get_conn()
    cur = conn.execute("DELETE FROM violations")
    conn.commit()
    conn.close()
    return cur.rowcount


def get_upload(upload_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def count_uploads_with_filename(filename):
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM uploads WHERE filename = ?", (filename,)).fetchone()[0]
    conn.close()
    return n


def delete_upload_record(upload_id):
    conn = get_conn()
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()


def get_uploads_older_than(days):
    conn = get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute("SELECT * FROM uploads WHERE timestamp < ?", (cutoff,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_audit(actor, action, detail=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_log (actor, action, detail, timestamp) VALUES (?, ?, ?, ?)",
        (actor, action, detail, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_audit(limit=200):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_admin_overview(upload_dir):
    conn = get_conn()
    q = lambda sql, args=(): conn.execute(sql, args).fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d")
    stats = {
        "users_total": q("SELECT COUNT(*) FROM users"),
        "users_active": q("SELECT COUNT(*) FROM users WHERE COALESCE(active,1) = 1"),
        "admins": q("SELECT COUNT(*) FROM users WHERE role = 'admin'"),
        "uploads_total": q("SELECT COUNT(*) FROM uploads"),
        "photo_uploads": q("SELECT COUNT(*) FROM uploads WHERE upload_type = 'photo'"),
        "video_uploads": q("SELECT COUNT(*) FROM uploads WHERE upload_type = 'video'"),
        "violations_total": q("SELECT COUNT(*) FROM violations"),
        "violations_today": q("SELECT COUNT(*) FROM violations WHERE timestamp LIKE ?", (today + "%",)),
        "falls_total": q("SELECT COUNT(*) FROM violations WHERE type = 'Fall detected'"),
    }

    severity = {"high": 0, "medium": 0, "low": 0}
    for r in conn.execute("SELECT severity, COUNT(*) AS c FROM violations GROUP BY severity"):
        severity[r["severity"]] = r["c"]

    top_types = [dict(r) for r in conn.execute(
        "SELECT type, COUNT(*) AS count FROM violations GROUP BY type ORDER BY count DESC LIMIT 5")]

    trend = []
    for i in range(6, -1, -1):
        day = datetime.now().date() - timedelta(days=i)
        n = q("SELECT COUNT(*) FROM violations WHERE timestamp LIKE ?", (day.isoformat() + "%",))
        trend.append({"label": day.strftime("%a"), "date": day.isoformat(), "count": n})
    conn.close()

    uploads_bytes, uploads_files = 0, 0
    if os.path.isdir(upload_dir):
        for f in os.listdir(upload_dir):
            fp = os.path.join(upload_dir, f)
            if os.path.isfile(fp):
                uploads_files += 1
                uploads_bytes += os.path.getsize(fp)

    system = {
        "db_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
        "uploads_files": uploads_files,
        "uploads_bytes": uploads_bytes,
        "disk_free_bytes": shutil.disk_usage(os.path.dirname(DB_PATH)).free,
        "server_time": datetime.now().isoformat(timespec="seconds"),
    }
    return {"stats": stats, "severity": severity, "top_types": top_types, "trend": trend, "system": system}


# ---------- key/value settings ----------
def get_setting(key, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()

