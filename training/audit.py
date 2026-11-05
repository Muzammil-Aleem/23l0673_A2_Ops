"""Task 1.2 — dataset audit.

Walks data/raw, and for every file builds a row with: class, file size, width,
height, colour mode, format, SHA-256, whether it is readable. Then finds exact
duplicates (same SHA-256), including duplicates that sit in BOTH classes.

Outputs: reports/audit.csv, reports/audit_summary.md (tables for the report).
Run:  python -m training.audit
"""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from visionguard.config import DATA_RAW, IMAGE_EXTENSIONS, REPORTS, CLASS_NAMES, folder_to_label


def label_for(path: Path):
    for part in path.relative_to(DATA_RAW).parts[:-1]:
        lab = folder_to_label(part)
        if lab is not None:
            return lab
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    row = {"path": path.relative_to(DATA_RAW).as_posix(), "file_size": path.stat().st_size,
           "width": None, "height": None, "mode": None, "format": None,
           "readable": True, "error": "", "sha256": sha256_of(path)}
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.load()  # verify the pixel data actually decodes, not just the header
    except Exception as exc:  # noqa: BLE001 - any failure means "unreadable"
        row.update(readable=False, error=f"{type(exc).__name__}: {exc}"[:200])
    return row


def main():
    files = [p for p in DATA_RAW.rglob("*") if p.is_file()]
    rows, skipped = [], []
    for p in sorted(files):
        lab = label_for(p)
        if lab is None:
            skipped.append(p)
            continue
        row = inspect(p)
        row["label"] = lab
        row["class"] = CLASS_NAMES[lab]
        row["ext_is_image"] = p.suffix.lower() in IMAGE_EXTENSIONS
        rows.append(row)

    df = pd.DataFrame(rows)
    # exact duplicates: every file whose hash appeared earlier is a duplicate
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    classes_per_hash = df.groupby("sha256")["label"].nunique()
    df["cross_class_duplicate"] = df["sha256"].map(classes_per_hash) > 1

    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "audit.csv", index=False)

    ok = df[df.readable]
    counts = df.groupby("class").agg(files=("path", "count"), readable=("readable", "sum"),
                                     corrupt=("readable", lambda s: (~s).sum()),
                                     duplicates=("is_duplicate", "sum"))
    stats = ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"])
    stats_all = ok[["width", "height", "file_size"]].agg(["min", "median", "max"])
    modes = pd.crosstab(ok["class"], ok["mode"])
    formats = pd.crosstab(ok["class"], ok["format"])

    lines = ["# Dataset audit summary\n",
             f"Total files: {len(df)} (ignored files outside class folders: {len(skipped)})\n",
             "## Counts per class\n", counts.to_markdown(), "\n",
             "## Width / height / file size (bytes) per class\n", stats.to_markdown(), "\n",
             "## Overall\n", stats_all.to_markdown(), "\n",
             "## Colour modes\n", modes.to_markdown(), "\n",
             "## Formats\n", formats.to_markdown(), "\n",
             f"## Problems\n\n- corrupt / unreadable files: {int((~df.readable).sum())}",
             f"- exact duplicate files (extra copies): {int(df.is_duplicate.sum())}",
             f"- files whose hash appears in BOTH classes: {int(df.cross_class_duplicate.sum())}\n"]
    bad = df[~df.readable][["path", "error"]]
    if len(bad):
        lines += ["### Unreadable files\n", bad.to_markdown(index=False), "\n"]
    (REPORTS / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Wrote", REPORTS / "audit.csv")


if __name__ == "__main__":
    main()
