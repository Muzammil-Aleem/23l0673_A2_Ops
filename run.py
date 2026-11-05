"""Local entry point (Task 4.4): `python run.py`.
Debug mode is OFF: the Werkzeug debugger lets anyone who can reach the page execute
Python on the server, the reloader would load the TensorFlow model twice, and full
tracebacks would leak internals to users. In Docker the app runs under gunicorn
(wsgi.py) instead of the Flask development server."""
import os

from dotenv import load_dotenv

load_dotenv()

from webapp import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
