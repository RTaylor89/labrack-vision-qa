# Data

Staged, non-patient images only. No PHI, no real sample IDs, no production barcodes.

## Layout

```text
data/
├── raw/            # unlabeled staged photos, straight from the camera
├── images/         # images used for training/validation (git-ignored)
├── labels/         # YOLO-format .txt label files
└── dataset.yaml    # class names + train/val/test paths
```

Bulk image files are git-ignored (see `.gitignore`). Commit `dataset.yaml`, label
files, and a few small sample images only.

## Classes

```yaml
names:
  0: rack
  1: tube
  2: cap
  3: empty_slot
```

`tilted_tube` / abnormal-tube-position is added only if it can be labeled consistently.

## Safety rule

Use only empty tubes, clean racks, fake labels, and synthetic identifiers. If anything
resembling real patient data appears in an image, remove it before committing.
