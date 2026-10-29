"""
One-off migration: reorganizes already-generated static datasets from the old
flat layout into SAM2's native PNGRawDataset layout (one subfolder per sample,
same convention as video datasets).

Before:  {root}/static/{dataset}/images/{name}.jpg
                                  /masks/{name}.png
After:   {root}/static/{dataset}/images/{name}/00000.jpg
                                  /masks/{name}/00000.png

Idempotent: a sample already migrated (destination file present) is skipped.
Moves files in place -- doesn't touch anything else. Run this on the cluster
against the real US-43d/ directory.

Usage:
  python migrate_static_layout.py [root]
  (root defaults to the data_root used in the training config)
"""

import sys
from pathlib import Path

DEFAULT_ROOT = "/data/home/jlpmarques/storage/Dissertation/US-43d"


def migrate_kind(kind_dir: Path, ext: str):
    moved = 0
    already = 0
    for f in sorted(kind_dir.glob(f"*{ext}")):
        if not f.is_file():
            continue
        dest_dir = kind_dir / f.stem
        dest = dest_dir / f"00000{ext}"
        if dest.exists():
            already += 1
            continue
        dest_dir.mkdir(exist_ok=True)
        f.rename(dest)
        moved += 1
    return moved, already


def migrate_dataset(dataset_dir: Path):
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        return None
    return migrate_kind(images_dir, ".jpg") + migrate_kind(masks_dir, ".png")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    static_root = root / "static"
    if not static_root.exists():
        print(f"Static root not found: {static_root}")
        return

    for dataset_dir in sorted(p for p in static_root.iterdir() if p.is_dir()):
        result = migrate_dataset(dataset_dir)
        if result is None:
            print(f"[{dataset_dir.name}] skipped (no images/ or masks/ subfolder)")
            continue
        img_moved, img_already, msk_moved, msk_already = result
        print(
            f"[{dataset_dir.name}] images: {img_moved} moved, {img_already} already done | "
            f"masks: {msk_moved} moved, {msk_already} already done"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
