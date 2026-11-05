"""Task 8 — push several images through the LOCAL Flask app (which, in remote mode,
forwards them to the cloud model API) and print what came back + the history rows.

  python scripts/send_samples.py --n 6                     # random test-split images
  python scripts/send_samples.py img1.jpg img2.png ...      # your own files
"""
import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visionguard.config import DATA_RAW, SPLITS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    a = ap.parse_args()

    files = [Path(f) for f in a.files]
    truth = {}
    if not files:
        df = pd.read_csv(SPLITS_DIR / "test.csv").sample(a.n, random_state=random.randint(0, 9999))
        files = [DATA_RAW / p for p in df.path]
        truth = {DATA_RAW / p: ("AI-generated" if lab else "Real") for p, lab in zip(df.path, df.label)}

    for f in files:
        with open(f, "rb") as fh:
            r = requests.post(f"{a.url}/predict", files={"image": (f.name, fh)}, timeout=60)
        verdict = re.search(r'class="verdict[^"]*">([^<]+)<', r.text)
        conf = re.search(r"Confidence: <strong>([^<]+)<", r.text)
        print(f"{r.status_code}  {f.name[:40]:<40} -> {verdict.group(1) if verdict else '?':<13} "
              f"{conf.group(1) if conf else '':>7}   truth={truth.get(f, '?')}")

    print("\nLast rows of the database trail (/api/history):")
    rows = requests.get(f"{a.url}/api/history", params={"limit": len(files)}, timeout=10).json()
    print(pd.DataFrame(rows)[["id", "created_at", "filename", "label", "confidence", "model_name",
                              "model_version", "inference_mode", "latency_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
