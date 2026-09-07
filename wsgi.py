"""
wsgi.py — saxenashvam0321@gmail.com
--------
Entry point for production WSGI servers.

The application object is created at import time and the database is prepared
(tables created, default admin seeded) once, before the first request is served
— a fresh deployment therefore comes up ready to log into without any manual
setup step.

    waitress-serve --listen=0.0.0.0:8000 wsgi:application      (Windows or Linux)
    gunicorn --bind 0.0.0.0:8000 wsgi:application              (Linux/macOS)

`app` is aliased to `application` because some platforms look for one name and
some the other.
"""

import os

from app import create_app, bootstrap_database

application = create_app()

# Idempotent: creates any missing tables and the admin account, then does
# nothing on subsequent boots. Set SKIP_DB_BOOTSTRAP=1 if you manage schema
# migrations yourself and don't want the app touching the database on start.
if os.environ.get("SKIP_DB_BOOTSTRAP", "").strip().lower() not in {"1", "true", "yes"}:
    bootstrap_database(application)

# Alias for `gunicorn wsgi:app` / `waitress-serve wsgi:app`.
app = application


if __name__ == "__main__":
    # `python wsgi.py` serves the app with waitress rather than Flask's
    # development server, so it can be used as a simple production launcher.
    from waitress import serve

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving LiverAI with waitress on http://{host}:{port}")
    serve(application, host=host, port=port, threads=8)
