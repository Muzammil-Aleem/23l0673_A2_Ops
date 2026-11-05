"""Task 2.1 — tf.data input pipeline.

Stages (in order):
  1. from_tensor_slices(paths, labels)  - read the split list (CSV) into a dataset
  2. map(load)                          - read bytes + shared preprocess (decode, EXIF,
                                          RGB, centre-crop square, resize) via numpy_function
  3. cache()                            - keep decoded 224x224 images in RAM: decoding is
                                          the slow part and the dataset is small (~1k images)
  4. shuffle (train only)               - new order every epoch, seeded
  5. batch(BATCH_SIZE)
  6. map(augment) (train only)          - random flip / rotation / zoom / contrast, applied
                                          AFTER cache so every epoch sees new variants
  7. prefetch(AUTOTUNE)                 - prepare the next batch while the model trains

Run `python -m training.pipeline` to save a figure of one augmented batch.
"""
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from visionguard.config import BATCH_SIZE, DATA_RAW, FIGURES, IMG_SIZE, SEED, SPLITS_DIR
from visionguard.preprocess import preprocess_path

AUTOTUNE = tf.data.AUTOTUNE


def read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    df["abs_path"] = [str(DATA_RAW / p) for p in df.path]
    return df


def make_augmenter(strength: str = "medium") -> keras.Sequential:
    factors = {"light": (0.03, 0.05, 0.1), "medium": (0.06, 0.12, 0.2), "strong": (0.12, 0.2, 0.3)}
    rot, zoom, contrast = factors[strength]
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(rot, fill_mode="reflect", seed=SEED),
        keras.layers.RandomZoom(zoom, fill_mode="reflect", seed=SEED),
        keras.layers.RandomContrast(contrast, seed=SEED),
    ], name="augmentation")


def _load_fn(img_size):
    def load(path, label):
        img = tf.numpy_function(lambda p: preprocess_path(p.decode(), img_size), [path], tf.float32)
        img.set_shape((img_size, img_size, 3))
        return img, tf.cast(label, tf.float32)
    return load


def make_dataset(split: str, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE,
                 augment: str | None = None, shuffle: bool | None = None) -> tf.data.Dataset:
    df = read_split(split)
    shuffle = (split == "train") if shuffle is None else shuffle
    ds = tf.data.Dataset.from_tensor_slices((df.abs_path.values, df.label.values))
    ds = ds.map(_load_fn(img_size), num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment and split == "train":
        aug = make_augmenter(augment)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def labels_of(split: str) -> np.ndarray:
    return read_split(split).label.values


def save_batch_figure(path=FIGURES / "augmented_batch.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    x, y = next(iter(make_dataset("train", augment="medium")))
    fig, axes = plt.subplots(4, 8, figsize=(16, 8.5))
    for i, ax in enumerate(axes.flat):
        if i < len(x):
            ax.imshow(x[i].numpy().astype("uint8"))
            ax.set_title("AI" if y[i] == 1 else "real", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch ({IMG_SIZE}x{IMG_SIZE}, centre-crop + resize)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    save_batch_figure()
