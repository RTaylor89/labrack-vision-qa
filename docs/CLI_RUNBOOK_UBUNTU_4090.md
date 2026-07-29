# LabRack Vision QA — Complete CLI Runbook

This is the sequence I can follow from a clean terminal while recording the
project. All images must be staged and non-patient. Every reported finding is a
possible issue requiring human review.

## 1. Open the repository

```bash
cd /path/to/labrack-vision-qa
pwd
git status --short
```

I run all remaining commands from the repository root.

## 2. Create and verify the environment

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import cv2, torch, ultralytics; print('torch', torch.__version__); print('ultralytics', ultralytics.__version__); print('opencv', cv2.__version__); print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
yolo checks
```

Python must be 3.10 or newer.

On my Ubuntu RTX 4090 system I also run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA NOT AVAILABLE')"
```

I stop and fix the PyTorch installation if CUDA is unavailable. I use the
current Linux/Pip/CUDA command from the official PyTorch installation selector
instead of guessing a wheel.

## 3. Select a dataset version

```bash
export LABRACK_DATASET_YAML=data/dataset.yaml
```

The active dataset has 125 training images, 5 Rack_C validation images, and 10
Rack_D diagnostic images. Its schema is:

```yaml
names:
  0: rack
  1: tube
  2: cap
  3: empty_slot
```

The current partial export has no usable tube instances. Tube detection remains
a roadmap goal until the staged dataset contains clearly visible tubes.

For a new reviewed Roboflow export, I preserve the untouched files under
`data/source/<version>/`, build a separate version under
`data/versions/<version>/`, and select it without changing source code:

```bash
export LABRACK_DATASET_YAML=data/versions/2026-08-15/dataset.yaml
```

I split by physical rack or collection session, not by random neighboring
frames. Rack_D has already informed model selection, so I use a new rack group
for a future unbiased final test.

## 4. Validate code and annotations

```bash
python -m pytest -q
python -m compileall -q src auto_label.py smoke_train.py
```

I also open the full notebook and run its dataset-validation cells:

```bash
jupyter lab notebooks/02_full_pipeline_demo.ipynb
```

The checks cover image/label pairing, class IDs 0–3, normalized five-column
YOLO rows, and duplicate labels. I still review the staged images visually
because structural checks cannot prove that a box is on the correct object.

## 5. Reproduce the M3 training progression

On Apple silicon I use `device=mps`, `workers=0`, and the exact sequence below.

First I prove that YOLO11s fits at 640:

```bash
yolo detect train \
  model=yolo11s.pt \
  data="$LABRACK_DATASET_YAML" \
  epochs=1 \
  imgsz=640 \
  batch=4 \
  device=mps \
  workers=0 \
  project=runs \
  name=smoke_yolo11s \
  exist_ok=True
```

Then I prove that 960 pixels fits:

```bash
yolo detect train \
  model=yolo11s.pt \
  data="$LABRACK_DATASET_YAML" \
  epochs=1 \
  imgsz=960 \
  batch=2 \
  device=mps \
  workers=0 \
  project=runs \
  name=smoke_yolo11s_960 \
  exist_ok=True
```

Both smoke runs completed on the M3 Pro. They test execution and memory only;
their one-epoch metrics are not model-quality evidence.

## 6. Run the controlled candidates

I change one capacity/resolution choice at a time and select on validation.

### YOLO11n at 960

```bash
yolo detect train \
  model=yolo11n.pt data="$LABRACK_DATASET_YAML" \
  epochs=100 patience=20 imgsz=960 batch=2 \
  device=mps workers=0 seed=0 \
  project=runs name=candidate_yolo11n_960 exist_ok=True plots=True
```

### YOLO11s at 640

```bash
yolo detect train \
  model=yolo11s.pt data="$LABRACK_DATASET_YAML" \
  epochs=100 patience=20 imgsz=640 batch=4 \
  device=mps workers=0 seed=0 \
  project=runs name=candidate_yolo11s_640 exist_ok=True plots=True
```

### YOLO11s at 960

```bash
yolo detect train \
  model=yolo11s.pt data="$LABRACK_DATASET_YAML" \
  epochs=100 patience=20 imgsz=960 batch=2 \
  device=mps workers=0 seed=0 \
  project=runs name=candidate_yolo11s_960 exist_ok=True plots=True
```

### Gated YOLO11m at 640

YOLO11s/960 clearly improved validation empty-slot recall, so I ran the
medium-model gate:

```bash
yolo detect train \
  model=yolo11m.pt data="$LABRACK_DATASET_YAML" \
  epochs=100 patience=20 imgsz=640 batch=2 \
  device=mps workers=0 seed=0 \
  project=runs name=candidate_yolo11m_640 exist_ok=True plots=True
```

The medium run stopped at epoch 52 and did not beat YOLO11s/960.

## 7. Read the actual selection evidence

```bash
column -s, -t results/metrics/candidate_comparison.csv
column -s, -t results/metrics/selected_validation_metrics.csv
```

The selected YOLO11s/960 checkpoint reached validation mAP50 0.825,
mAP50–95 0.594, and empty-slot recall 0.858. The validation split has only five
images, so these measurements remain provisional.

## 8. Run the same progression on the RTX 4090

On Ubuntu I keep the model, data, resolution, patience, and seed the same. I
change hardware settings to `device=0`, `workers=8`, and `amp=True`. I begin
with the same conservative batches, then increase batch only after confirming
VRAM use with `watch -n 1 nvidia-smi`.

Example selected-candidate reproduction:

```bash
yolo detect train \
  model=yolo11s.pt \
  data="$LABRACK_DATASET_YAML" \
  epochs=100 \
  patience=20 \
  imgsz=960 \
  batch=2 \
  device=0 \
  workers=8 \
  amp=True \
  seed=0 \
  project=runs \
  name=cuda_yolo11s_960 \
  exist_ok=True \
  plots=True
```

The RTX 4090 should shorten experiment time and allow larger batches. It does
not guarantee better accuracy: data, labels, configuration, and model selection
determine accuracy, and CUDA/MPS numerical differences can change exact scores.

## 9. Promote the fixed checkpoint

For the measured M3 result:

```bash
mkdir -p weights
cp runs/candidate_yolo11s_960/weights/best.pt \
  weights/labrack_yolo11s_960.pt
export LABRACK_MODEL_PATH=weights/labrack_yolo11s_960.pt
```

For an independently completed CUDA reproduction, I change only the source run
directory after checking its validation evidence.

## 10. Deliberate diagnostic evaluation

This command reproduces the current Rack_D diagnostic result:

```bash
yolo detect val \
  model="$LABRACK_MODEL_PATH" \
  data="$LABRACK_DATASET_YAML" \
  split=test \
  imgsz=960 \
  batch=2 \
  device=0 \
  workers=8 \
  project=runs \
  name=selected_yolo11s_960_test \
  exist_ok=True \
  plots=True
```

On Apple silicon I substitute `device=mps workers=0`. I do not repeatedly tune
against this result. Because Rack_D informed this iteration, I describe it as
diagnostic held-out evidence, not a pristine final test.

## 11. Run the complete application

```bash
python -m src.run \
  --image data/images/test/Rack_D_image00304_jpg.rf.DzfIrdoLsLIpJhLkkOfw.jpg \
  --output-dir output/demo \
  --model "$LABRACK_MODEL_PATH" \
  --imgsz 960 \
  --conf 0.25
```

I inspect all three real outputs:

```bash
find output/demo -maxdepth 1 -type f -print
cat output/demo/*_summary.txt
python -m json.tool output/demo/*_results.json
xdg-open output/demo/*_annotated.jpg
```

On the measured M3 run, the detection step was 0.732 seconds and full CLI wall
time was 2.07 seconds. The difficult demo image contains 81 annotated empty
slots and no caps; the selected model reported seven possible empty positions
and one cap. I present that mismatch openly as a possible issue requiring human
review.

## 12. Run the notebook

```bash
export LABRACK_DATASET_YAML=data/dataset.yaml
export LABRACK_MODEL_PATH=weights/labrack_yolo11s_960.pt
jupyter lab notebooks/02_full_pipeline_demo.ipynb
```

I use **Restart Kernel and Run All** with optional training/evaluation switches
left `False` during the short demo. The notebook still contains the full
training and evaluation code, dataset-version selection, annotation validation,
application call, JSON inspection, and metric comparison.

## 13. Record the 3–5 minute video

1. Show the environment and staged dataset configuration.
2. Run the automated tests.
3. Show the candidate comparison and explain why YOLO11s/960 won.
4. Run the application command.
5. Open the annotated image, JSON, and plain-text summary.
6. Show the diagnostic metrics and the difficult-image mismatch.
7. State that every finding requires human review and tube detection is not
   measured yet.
8. Show how `LABRACK_DATASET_YAML` and `LABRACK_MODEL_PATH` select later
   reviewed versions.

I test the uploaded video link in a private browser window and replace the
pending README link before submission.
