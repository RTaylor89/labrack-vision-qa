# Data

Staged, non-patient images only. No PHI, no real sample IDs, no production barcodes.

Staged photography, image annotation, and testing methodology were contributed
by Kate Leemann.

## Layout

```text
data/
├── raw/            # unlabeled staged photos, straight from the camera
├── images/         # images used for training/validation/testing
├── labels/         # YOLO-format .txt label files
├── source/         # original export metadata and labels for provenance
├── split_manifest.csv
└── dataset.yaml    # class names + train/val/test paths
```

Dataset images and labels are intentionally eligible for Git so project peers can
validate the staged dataset. Before committing, verify every image uses staged,
non-patient materials and contains no PHI, real sample IDs, or production barcodes.

## Export from Roboflow

1. Confirm the Roboflow project is an **Object Detection** project and that its
   classes use this exact order: `rack`, `tube`, `cap`, `empty_slot`.
2. Review every annotation, then create a dataset version. Keep a dedicated test
   split that is hand-labeled and was never auto-labeled or used for training.
3. Open that version, select **Download Dataset**, choose a YOLOv8/YOLO11
   object-detection export, and download the ZIP. These formats use the same
   normalized YOLO text-label structure expected by Ultralytics YOLO11.
4. Extract the ZIP outside this repository first. Do not replace this project's
   `data/dataset.yaml` blindly; Roboflow may generate different path names.

Roboflow commonly exports folders named `train`, `valid`, and `test`, each with
`images/` and `labels/` beneath it. Copy their contents into this repository as:

```text
data/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/       # copy Roboflow's valid/images files here
│   └── test/
└── labels/
    ├── train/
    ├── val/       # copy Roboflow's valid/labels files here
    └── test/
```

For every image, preserve the matching image and label stem—for example,
`rack_001.jpg` must pair with `rack_001.txt`. An empty `.txt` is valid for a
reviewed image containing none of the four classes.

After staging, confirm `data/dataset.yaml` still contains:

```yaml
path: data
train: images/train
val: images/val
test: images/test

names:
  0: rack
  1: tube
  2: cap
  3: empty_slot
```

## Classes

```yaml
names:
  0: rack
  1: tube
  2: cap
  3: empty_slot
```

`tilted_tube` / abnormal-tube-position is added only if it can be labeled consistently.

## Current partial import

The first Roboflow export was staged on 2026-07-29. It contains 140 images and
is incomplete; do not treat it as the final training or evaluation dataset.

The export placed every image in one training folder and used three source
classes: `cap` (0), `empty` (1), and `rack` (2). During staging:

- source `rack` ID 2 became project `rack` ID 0;
- source `cap` ID 0 became project `cap` ID 2;
- source `empty` ID 1 became project `empty_slot` ID 3; and
- project `tube` ID 1 received no annotations because it was absent from this
  partial export.

Seven Rack_B images intentionally have no `rack` annotation. A rack may be
present, but it is not clearly visible enough in those images to support a
reliable annotation. These are not treated as missing labels.

The `tube` class remains reserved in `dataset.yaml` as a roadmap goal. Tubes are
not sufficiently visible in the current staged images, so the project will not
claim tube detection or train that class until additional staged images provide
clear, consistently annotatable tube views.

The source contained a mixture of polygon and box annotations. For the current
YOLO11 object-detection pipeline, polygons were converted to tight axis-aligned
boxes. The untouched source labels and Roboflow metadata are retained under
`source/roboflow_partial_2026-07-29/`.

To prevent adjacent frames of the same physical rack from appearing in both
training and evaluation, the split is grouped by rack identity:

| Split | Source groups | Images |
|---|---|---:|
| train | Rack_A, Rack_B | 125 |
| val | Rack_C | 5 |
| test | Rack_D | 10 |

`split_manifest.csv` records every assignment. Apply these same rack-based
assignments when the completed annotation export is added; do not randomly
redistribute the existing images.

## Safety rule

Use only empty tubes, clean racks, fake labels, and synthetic identifiers. If anything
resembling real patient data appears in an image, remove it before committing.
