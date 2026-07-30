# LabRack Vision QA — Fresh Ubuntu RTX 4090 Runbook

This is the controlled sequence I will follow for the next reviewed dataset
version on my Ubuntu RTX 4090 workstation. I will train all six candidates from
isolated official Ultralytics YOLO11 base checkpoints. I will not copy, load,
resume, or fine-tune the current LabRack checkpoint created on the Apple M3
system.

I use the official COCO-pretrained YOLO11 checkpoints rather than random
initializations. By “fresh,” I mean that I start each LabRack candidate from
the official base for its model size and never from an earlier LabRack training
result.

I use staged, non-patient images only. I describe every reported finding as a
possible issue requiring human review.

## 1. Open the repository

```bash
cd /path/to/labrack-vision-qa
pwd
git status --short
```

I run all remaining commands from the repository root.

## 2. Create and verify the environment

If the basic Ubuntu tools are missing, I install them first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git unzip
```

I confirm that the repository and external output location have at least 20 GB
of free space for six raw training runs, validation plots, predictions,
checkpoints, and the handoff package:

```bash
df -h .
```

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

At the official PyTorch selector I choose **Stable**, **Linux**, **Pip**,
**Python**, and the CUDA version recommended for my installed NVIDIA driver. I
copy and run the generated command inside this virtual environment. I then
install the project requirements:

```bash
python -m pip install -r requirements.txt
python -c "import cv2, torch, ultralytics; print('torch', torch.__version__); print('ultralytics', ultralytics.__version__); print('opencv', cv2.__version__); print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
yolo checks
```

Python must be 3.10 or newer.

On my Ubuntu RTX 4090 system I run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA NOT AVAILABLE')"
```

I stop and fix the PyTorch installation if CUDA is unavailable or the reported
GPU is not my RTX 4090. I use the current Linux/Pip/CUDA command from the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
instead of guessing a CUDA wheel. I use the official Ultralytics YOLO11
checkpoint names documented in the
[YOLO11 model guide](https://docs.ultralytics.com/models/yolo11/).

## 3. Select a dataset version

```bash
export LABRACK_DATASET_YAML=data/dataset.yaml
```

I use an active dataset with 125 training images, 5 Rack_C validation images,
and 10 Rack_D diagnostic images. I keep this schema:

```yaml
names:
  0: rack
  1: tube
  2: cap
  3: empty_slot
```

I have no usable tube instances in the current partial export. I keep tube
detection as a roadmap goal until the staged dataset contains clearly visible
tubes.

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

I use these checks for image/label pairing, class IDs 0–3, normalized
five-column YOLO rows, and duplicate labels. I still review the staged images
visually because structural checks cannot prove that a box is on the correct
object.

## 5. Keep the 4090 experiment separate from all earlier runs

I extract the supplied portable repository ZIP into a new directory. I do not
copy the Mac `.venv`, `runs/`, `output/`, root-level `yolo11*.pt` files, or any
previous LabRack checkpoint into that directory.

I create an external output folder so the raw runs and return package do not
change my repository while training:

```bash
export LABRACK_4090_OUTPUT="$HOME/labrack-4090-output"
mkdir -p "$LABRACK_4090_OUTPUT"
```

I confirm that no prior LabRack checkpoint can be selected accidentally:

```bash
find . -path './.git' -prune -o -path './.venv' -prune -o \
  -name '*.pt' -print
```

I exclude every `.pt` file, including the earlier selected checkpoint under
`weights/`, from the supplied RTX 4090 rebuild ZIP. I retain the current M3
tables and plots as prior-version documentation, but no Mac-trained model binary
is present.
I create a new isolated `official_base_models/` directory beneath each unique
output run ID.

## 6. Inspect the immutable six-candidate plan

```bash
python scripts/run_rtx4090_matrix.py \
  --data "$LABRACK_DATASET_YAML" \
  --output-root "$LABRACK_4090_OUTPUT" \
  --run-id final_rtx4090 \
  --dry-run
```

I confirm that the printed matrix contains exactly:

| Candidate | Official start | Image size | Batch |
|---|---|---:|---:|
| YOLO11n / 640 | `yolo11n.pt` | 640 | 16 |
| YOLO11n / 960 | `yolo11n.pt` | 960 | 8 |
| YOLO11s / 640 | `yolo11s.pt` | 640 | 16 |
| YOLO11s / 960 | `yolo11s.pt` | 960 | 8 |
| YOLO11m / 640 | `yolo11m.pt` | 640 | 16 |
| YOLO11m / 960 | `yolo11m.pt` | 960 | 8 |

I fix `epochs=100`, `patience=20`, `device=0`, `workers=8`, `amp=True`,
`seed=0`, `deterministic=True`, and `resume=False` unless I pass an explicit
documented override. I also keep `cache=False` so each candidate reads the same
source dataset files. I do not change batch size midway through this controlled
final matrix.

## 7. Run a separate one-epoch six-candidate smoke matrix

I use a unique smoke run ID. This checks that all six model/resolution
combinations fit and that the complete validation, selection, POC, and packaging
path works:

```bash
python scripts/run_rtx4090_matrix.py \
  --data "$LABRACK_DATASET_YAML" \
  --output-root "$LABRACK_4090_OUTPUT" \
  --run-id smoke_rtx4090 \
  --epochs 1 \
  --patience 0
```

I never use smoke metrics or smoke checkpoints in the final deliverables. The
full run uses a new run ID and therefore downloads a separate isolated copy of
each official base checkpoint.

If any candidate runs out of memory, I stop. I change the fixed matrix batch
for that resolution in one documented code revision, delete neither the failed
run nor its evidence, and start the complete smoke matrix again with a new run
ID. I do not silently continue with different settings.

## 8. Run the full RTX 4090 matrix for the next dataset version

I choose a unique final run ID and record it:

```bash
export LABRACK_4090_RUN_ID="final_rtx4090_$(date -u +%Y%m%d)"

python scripts/run_rtx4090_matrix.py \
  --data "$LABRACK_DATASET_YAML" \
  --output-root "$LABRACK_4090_OUTPUT" \
  --run-id "$LABRACK_4090_RUN_ID"
```

In a second terminal I can monitor temperature, utilization, VRAM, and power:

```bash
watch -n 2 nvidia-smi
```

I refuse to overwrite an existing run ID. I instantiate a new YOLO object from
the isolated official base for every candidate. I do not resume or use a
checkpoint from another candidate.

I perform these stages:

1. I record Git, Python, PyTorch, Ultralytics, OpenCV, CUDA, cuDNN, RTX 4090,
   `nvidia-smi`, and `pip freeze` evidence.
2. I hash the dataset YAML, split manifest, images, and labels.
3. I download and hash isolated official `yolo11n.pt`, `yolo11s.pt`, and
   `yolo11m.pt` checkpoints.
4. I train `n`, `s`, and `m` at both 640 and 960 pixels.
5. I evaluate every candidate on validation and save per-class metrics and
   plots. Ultralytics supports explicit `val`/`test` split selection and saved
   validation plots as documented in its
   [validation guide](https://docs.ultralytics.com/modes/val/).
6. I select the highest validation `empty_slot` recall. I use overall
   mAP50–95, then overall mAP50, only as tie-breakers.
7. I evaluate only the selected checkpoint on the test split.
8. I generate selected-model predictions, annotated POC output, JSON, plain
   text, and measured timing.
9. I create a handoff directory, internal file checksums, a ZIP, and an
   adjacent ZIP checksum file.

I do not assume YOLO11s/960 will remain the winner. I report the model selected
by the new validation evidence.

## 9. Verify the completed handoff on Ubuntu

```bash
export LABRACK_HANDOFF_ZIP="$LABRACK_4090_OUTPUT/${LABRACK_4090_RUN_ID}_handoff.zip"

test -f "$LABRACK_4090_OUTPUT/$LABRACK_4090_RUN_ID/COMPLETE.txt"
cat "$LABRACK_4090_OUTPUT/$LABRACK_4090_RUN_ID/COMPLETE.txt"
unzip -t "$LABRACK_HANDOFF_ZIP"
sha256sum -c "$LABRACK_HANDOFF_ZIP.sha256"
```

I extract a verification copy and check every packaged file:

```bash
mkdir -p "$LABRACK_4090_OUTPUT/verify"
unzip -q "$LABRACK_HANDOFF_ZIP" \
  -d "$LABRACK_4090_OUTPUT/verify"

cd "$LABRACK_4090_OUTPUT/verify/${LABRACK_4090_RUN_ID}_handoff"
sha256sum -c CHECKSUMS.sha256
column -s, -t candidate_comparison.csv
cat selection.json
cd -
```

I confirm that `candidate_comparison.csv` has six rows and that the handoff
contains:

- environment and CUDA evidence;
- a top-level `run_manifest.json` with status `complete`, the exact six
  candidates, hyperparameters, starting-weight policy, and selection rule;
- the complete dataset hash manifest;
- every candidate's `best.pt`, training history, validation metrics, and plots;
- the declared selection rule and selected checkpoint hash;
- selected-model diagnostic test metrics and plots;
- selected predictions;
- the real POC annotated image, JSON, summary, and timing.

## 10. Bring the evidence back for the final repository update

I bring back these two files without renaming them:

```text
<run-id>_handoff.zip
<run-id>_handoff.zip.sha256
```

I place them in the parent `GH/` folder beside `labrack-vision-qa/`. I do not
manually merge individual training files into `results/`, because that could
mix M3 and RTX 4090 evidence.

After I attach the handoff ZIP in Codex, I ask for one final evidence import. I
expect that final update to:

1. verify the ZIP and every internal checksum;
2. preserve the current M3 results as the completed prior version;
3. place the six-candidate CUDA comparison and selected validation evidence
   under a clearly named RTX 4090 results directory;
4. replace the default selected checkpoint and update its SHA-256;
5. update `src/config.py` if the winning model size or image size changes;
6. regenerate POC predictions, JSON, summaries, and timing from the selected
   CUDA checkpoint;
7. update the README, results documentation, AI usage log, notebook commentary,
   notebook metric tables, demo script, checklist, presentation, and PDF;
8. execute the notebook with the final selected checkpoint;
9. rerun tests, render the slides and PDF, and package the final grading ZIP;
10. report only the real RTX 4090 measurements and retain the human-review
    boundary.

## 11. Run the final notebook after the evidence import

After the 4090 evidence has been imported and the selected checkpoint has been
promoted, I run:

```bash
export LABRACK_DATASET_YAML=data/dataset.yaml
export LABRACK_MODEL_PATH=weights/<selected-rtx4090-checkpoint>.pt
jupyter lab notebooks/02_full_pipeline_demo.ipynb
```

I use **Restart Kernel and Run All**. I verify that every displayed metric,
checkpoint name, image size, timing, and output comes from the imported RTX
4090 evidence. I do not type a result into the notebook unless it exists in the
handoff package.

## 12. Record the 3–5 minute video from the final state

1. I show the environment and staged dataset configuration.
2. I run the automated tests.
3. I show all six CUDA candidates and the declared selection rule.
4. I explain why the validation evidence selected the winner.
5. I run the application command with the selected RTX 4090 checkpoint.
6. I open the annotated image, JSON, and plain-text summary.
7. I show the diagnostic metrics and an honest failure case.
8. I state that every finding requires human review and tube detection remains
   unmeasured until the dataset contains usable tube annotations.
9. I show how `LABRACK_DATASET_YAML` supports a later reviewed dataset version.

I test the uploaded video link in a private browser window and replace the
pending README link before submission.
