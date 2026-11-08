"""Prepare the API-only Vercel project. Rerun after changing serving code."""
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    target = root / "deploy" / "vercel"
    target.mkdir(parents=True, exist_ok=True)
    for package in ("visionguard", "model_api"):
        output = target / package
        output.mkdir(exist_ok=True)
        for source in (root / package).glob("*.py"):
            shutil.copy2(source, output / source.name)
    scripts = target / "scripts"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(root / "scripts" / "download_deploy_model.py", scripts / "download_deploy_model.py")
    shutil.copy2(root / "model_api" / "requirements.txt", target / "requirements.txt")
    (target / ".python-version").write_text("3.12\n", encoding="utf-8")
    # Vercel's uv resolver otherwise considers Windows too, where the CPU
    # TensorFlow package can reference an unavailable tensorflow-intel release.
    (target / "pyproject.toml").write_text(
        '[project]\n'
        'name = "visionguard-model-api"\n'
        'version = "1.0.0"\n'
        'requires-python = ">=3.12,<3.13"\n\n'
        'dependencies = [\n'
        '    "flask==3.1.1",\n'
        '    "gunicorn==23.0.0",\n'
        '    "pillow==11.2.1",\n'
        '    "numpy==2.1.3",\n'
        '    "tensorflow-cpu==2.19.0",\n'
        ']\n\n'
        '[tool.uv]\n'
        'environments = ["sys_platform == \\\"linux\\\""]\n', encoding="utf-8")
    (target / ".gitignore").write_text("models/\n.env\n.vercel/\n__pycache__/\n", encoding="utf-8")
    (target / "app.py").write_text(
        '"""Vercel entrypoint: model files are downloaded during the build."""\n'
        'import os\n'
        'from pathlib import Path\n\n'
        'root = Path(__file__).resolve().parent\n'
        'os.environ["MODEL_PATH"] = str(root / "models" / "champion.keras")\n'
        'os.environ["MODEL_META"] = str(root / "models" / "champion.json")\n'
        'if not os.environ.get("MODEL_API_KEY"):\n'
        '    raise RuntimeError("Set MODEL_API_KEY in Vercel environment variables")\n\n'
        'from model_api.app import create_app  # noqa: E402\n\n'
        'app = create_app()\n', encoding="utf-8")
    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "buildCommand": "python scripts/download_deploy_model.py",
        "functions": {"app.py": {"maxDuration": 300, "includeFiles": "models/**"}},
    }
    (target / "vercel.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {target}")
    print("Commit deploy/vercel; use deploy/vercel as the Vercel Root Directory.")
    print("Model URLs and MODEL_API_KEY must be set in Vercel before deploying.")


if __name__ == "__main__":
    main()
