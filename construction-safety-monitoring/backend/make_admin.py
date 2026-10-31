"""Create or reset the admin account.

Set SENTINEL_ADMIN_PASSWORD in the environment before running:
    python make_admin.py
"""
import os
import db
from werkzeug.security import generate_password_hash
from datetime import datetime

USERNAME = os.environ.get("SENTINEL_ADMIN_USERNAME", "admin")
PASSWORD = os.environ.get("SENTINEL_ADMIN_PASSWORD", "")
if len(PASSWORD) < 12:
    raise SystemExit("Set SENTINEL_ADMIN_PASSWORD to a password of at least 12 characters.")

db.init_db()
conn = db.get_conn()
row = conn.execute("SELECT id FROM users WHERE username = ?", (USERNAME,)).fetchone()
if row:
    conn.execute("UPDATE users SET password_hash = ?, role = 'admin', active = 1 WHERE username = ?",
                 (generate_password_hash(PASSWORD), USERNAME))
else:
    conn.execute("INSERT INTO users (username, password_hash, role, active, created_at) VALUES (?, ?, 'admin', 1, ?)",
                 (USERNAME, generate_password_hash(PASSWORD), datetime.now().isoformat()))
conn.commit()
conn.close()
print(f'Admin account {USERNAME!r} created/updated.')
