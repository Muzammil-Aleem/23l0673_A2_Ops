"""Task 1.1 — print exact library versions and whether TensorFlow sees a GPU.
Run:  python labs/env_check.py   (screenshot the terminal, prompt visible)"""
import datetime
import getpass
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

print(f"user={getpass.getuser()}  time={datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python      {sys.version.split()[0]}  ({sys.executable})")
print(f"platform    {platform.platform()}")
for pkg in ["tensorflow", "keras", "keras-tuner", "mlflow", "dagshub", "dvc", "numpy",
            "pandas", "pillow", "scikit-learn", "matplotlib", "flask"]:
    try:
        print(f"{pkg:<12}{version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:<12}NOT INSTALLED")

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print(f"TF built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPUs visible to TensorFlow: {len(gpus)} {gpus}")
