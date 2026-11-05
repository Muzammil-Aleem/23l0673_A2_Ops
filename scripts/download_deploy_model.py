"""Download a published champion and its metadata during a hosting build."""
import json
import os
import urllib.request
from pathlib import Path


def main():
    destination = Path(__file__).resolve().parent.parent / "models"
    destination.mkdir(exist_ok=True)
    for variable, filename in (
        ("CHAMPION_MODEL_URL", "champion.keras"),
        ("CHAMPION_META_URL", "champion.json"),
    ):
        url = os.environ.get(variable, "")
        if not url.startswith("https://"):
            raise ValueError(f"Set {variable} to a public HTTPS download URL")
        target = destination / filename
        temporary = target.with_suffix(target.suffix + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "VisionGuard-deploy"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if not temporary.stat().st_size:
            raise ValueError(f"Downloaded {filename} is empty")
        if filename.endswith(".json"):
            json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(target)
        print(f"Downloaded {filename}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
