# Data

Staged, non-patient images only. No PHI, no real sample IDs, no production barcodes.

I handled the dataset integration, validation, conversion, splitting, and
supporting code. Kate Leemann collected the staged photos, completed the image
annotations, and assisted me in designing the testing methodology.

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

I intentionally keep dataset images and labels eligible for Git so my project
peers can validate the staged dataset. Before I commit an image, I verify that
it uses staged, non-patient materials and contains no PHI, real sample IDs, or
production barcodes.

## Export from Roboflow

1. I confirm that the Roboflow project is an **Object Detection** project and that its
   classes use this exact order: `rack`, `tube`, `cap`, `empty_slot`.
2. I review every annotation, then create a dataset version. I keep a dedicated
   test split that is hand-labeled and was never auto-labeled or used for training.
3. I open that version, select **Download Dataset**, choose a YOLOv8/YOLO11
   object-detection export, and download the ZIP. These formats use the same
   normalized YOLO text-label structure expected by Ultralytics YOLO11.
4. I extract the ZIP outside this repository first. I do not replace
   `data/dataset.yaml` blindly because Roboflow may generate different paths.

I expect Roboflow folders named `train`, `valid`, and `test`, each with
`images/` and `labels/` beneath it. I copy their contents into this repository as:

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

For every image, I preserve the matching image and label stem—for example,
`rack_001.jpg` must pair with `rack_001.txt`. An empty `.txt` is valid for a
reviewed image containing none of the four classes.

After staging, I confirm that `data/dataset.yaml` still contains:

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

I staged the first Roboflow export on 2026-07-29. It contains 140 images and is
incomplete, so I do not treat it as the final training or evaluation dataset.

The export placed every image in one training folder and used three source
classes: `cap` (0), `empty` (1), and `rack` (2). During staging, I applied these
class mappings:

- source `rack` ID 2 became project `rack` ID 0;
- source `cap` ID 0 became project `cap` ID 2;
- source `empty` ID 1 became project `empty_slot` ID 3; and
- project `tube` ID 1 received no annotations because it was absent from this
  partial export.

Kate intentionally left seven Rack_B images without a `rack` annotation. A rack
may be present, but it is not clearly visible enough in those images to support
a reliable annotation. I do not treat those cases as missing labels.

I keep the `tube` class reserved in `dataset.yaml` as a roadmap goal. Tubes are
not sufficiently visible in the current staged images, so I do not claim tube
detection or train that class until additional staged images provide clear,
consistently annotatable tube views.

The source contained a mixture of polygon and box annotations. For my current
YOLO11 object-detection pipeline, I converted polygons to tight axis-aligned
boxes. I retained the untouched source labels and Roboflow metadata under
`source/roboflow_partial_2026-07-29/`.

To prevent adjacent frames of the same physical rack from appearing in both
training and evaluation, I grouped the split by rack identity:

| Split | Source groups | Images |
|---|---|---:|
| train | Rack_A, Rack_B | 125 |
| val | Rack_C | 5 |
| test | Rack_D | 10 |

I record every assignment in `split_manifest.csv`. When I add the completed
annotation export, I will preserve these rack-based assignments instead of
randomly redistributing the existing images.

## Safety rule

I use only empty tubes, clean racks, fake labels, and synthetic identifiers. If
I find anything resembling real patient data in an image, I remove it before
committing.
