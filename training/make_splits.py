"""Create stratified train / val / test split lists (70 / 15 / 15) from the audit.

Corrupt files and extra copies of duplicates are dropped BEFORE splitting, so the
same picture can never end up in both train and test (that would leak). Images
whose hash appears in both classes are dropped entirely because their label is
ambiguous.

Outputs data/splits/{train,val,test}.csv with columns path,label.
Run:  python -m training.make_splits
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from visionguard.config import REPORTS, SEED, SPLITS_DIR


def main():
    df = pd.read_csv(REPORTS / "audit.csv")
    before = len(df)
    df = df[df.readable & ~df.is_duplicate & ~df.cross_class_duplicate]
    df = df[(df.width >= 32) & (df.height >= 32)]  # icons/thumbnails carry no usable signal
    print(f"kept {len(df)} of {before} files after removing corrupt/duplicate/tiny")

    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["path", "label"]].to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        print(f"{name:<5} {len(part):>5}  class balance: {part.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
