"""WSGI entry point for gunicorn: `gunicorn wsgi:app`."""
from webapp import create_app

app = create_app()
