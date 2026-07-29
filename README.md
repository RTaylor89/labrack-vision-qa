# LabRack Vision QA

> Educational prototype only. Not validated for clinical, diagnostic, or production laboratory use. Human review is required.

## Team Members

- Roderick Taylor
- Kate Leemann

## Team Contributions

- Kate Leemann contributed the staged laboratory photography, image
  annotations, and testing methodology.

## Project Tier

**Tier 1: Core Project**

We are choosing Tier 1 because this project is focused, realistic, and demo-friendly. It uses pretrained computer vision models with a small staged dataset instead of trying to build a production lab system from scratch.

## Problem Statement

Labs rely on consistent sample handling, and visual checks are part of keeping that workflow clean. Staff may need to confirm that a rack has the expected tubes, that tubes are capped, and that obvious placement issues are caught before the samples move further through the process.

The problem matters because small visual issues can create rework, delays, or extra QA review. This project is not a diagnostic system and will not use real patient data. It is a staged visual QA assistant for lab-style rack images.

## Solution Overview

LabRack Vision QA takes a photo of a staged lab sample rack and returns an annotated image showing detected objects such as racks, tubes, caps, and empty slots. It also generates a short QA-style summary that reports counts and flags simple visible issues.

The first version focuses on still images. The final capstone extension could add video, SAM-style promptable segmentation, a small dashboard, or an exportable QA report.

## Current Build Status

Version 1 runs end to end on staged, non-patient still images:

1. validate an image path;
2. run a fine-tuned Ultralytics YOLO11s object detector at 960 pixels;
3. count detected rack, cap, and empty-slot objects;
4. describe possible issues that require human review; and
5. write an annotated image, structured JSON, and a plain-text summary.

The current partial dataset does not contain usable tube annotations. Tube
detection is therefore a roadmap goal and is not included in the reported
model performance. See [`data/README.md`](data/README.md) for provenance,
class remapping, and split details.

## Technical Approach

- **CV technique:** Object detection
- **Model:** Fine-tuned YOLO11s at 960-pixel input
- **Framework:** Python, Ultralytics, OpenCV, VSCode/Jupyter

YOLO11 fits the project because the core task is finding and labeling objects
in a photo. Segmentation remains a possible later experiment if bounding boxes
are not precise enough around crowded or overlapping objects.

## Dataset

- **Source:** Self-collected staged photos using empty tubes, racks, fake labels, and no PHI.
- **Current partial import:** 140 images.
- **Grouped split:** 125 train, 5 validation, and 10 held-out test images.
- **Split rule:** rack groups A/B train, C validation, and D test, preventing
  adjacent images of one physical rack from crossing splits.

**Current labels:**

- rack
- cap
- empty_slot
- tube (reserved class with zero current instances; roadmap goal)

**Data handling rule:** No real patient labels, no real sample IDs, and no medical decision-making.

## Success Metrics

- **Primary target:** At least 0.75 mAP50 on the holdout test set for the
  currently trained classes.
- **Secondary metric:** Process one image and generate an annotated output plus a QA summary in under 3 seconds per image.
- **Demo success:** The project produces an annotated rack image and a plain-English summary that a lab worker could quickly review.

## Measured Results

On 2026-07-29 I ran the planned model and image-size progression on an Apple M3
Pro with PyTorch MPS. Model selection used the five-image Rack_C validation
split. YOLO11s at 960 pixels was the clear winner, especially on the
`empty_slot` class.

| Candidate | Best epoch | Validation mAP50 | mAP50–95 | Empty-slot recall |
|---|---:|---:|---:|---:|
| YOLO11n / 640 baseline | 25 | 0.729 | 0.503 | 0.428 |
| YOLO11n / 960 | 15 | 0.809 | 0.539 | 0.470 |
| YOLO11s / 640 | 43 | 0.761 | 0.552 | 0.404 |
| **YOLO11s / 960** | **82** | **0.825** | **0.594** | **0.858** |
| YOLO11m / 640 | 32 | 0.730 | 0.520 | 0.381 |

The selected validation result was:

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| all trained classes | 447 | 0.799 | 0.898 | 0.825 | 0.594 |
| rack | 5 | 0.872 | 1.000 | 0.995 | 0.876 |
| cap | 276 | 0.716 | 0.837 | 0.663 | 0.503 |
| empty_slot | 166 | 0.807 | 0.858 | 0.817 | 0.405 |
| tube | 0 | — | — | — | — |

After fixing the configuration, I ran one diagnostic evaluation on the
10-image Rack_D group. It produced aggregate mAP50 0.937, mAP50–95 0.601, and
empty-slot recall 0.761. Rack_D had already informed this improvement cycle, so
these numbers are useful diagnostic evidence but are not presented as a
pristine final test of generalization.

A real end-to-end example recorded 0.732 seconds for detection and 2.07 seconds
of full CLI wall time, including startup, model loading, annotation, and report
writing. On that difficult image, the annotation contains 81 empty slots and no
caps; the selected model reported seven possible empty positions and one cap.
That remains a possible issue requiring human review and demonstrates why this
prototype cannot approve a rack autonomously.

See [`results/README.md`](results/README.md), the metric plots in
[`results/metrics/`](results/metrics/), and the real annotated samples in
[`results/predictions/`](results/predictions/).

## What Changed From the Blueprint

- The Blueprint proposed segmentation; Version 1 ships object detection because
  bounding boxes are sufficient for the current counting and review workflow.
- The planned 200–250 images became a 140-image partial staged export.
- The planned `tube` class is reserved but not measured because the current
  images do not support consistent tube annotation.
- Controlled candidate runs selected YOLO11s at 960 pixels instead of choosing
  a model size in advance.

The scope changed to fit the available staged data, while the end-to-end
promise remained intact: one image produces an annotated image, JSON, and a
plain-English summary for human review.

## Challenges and Fixes

- **Small empty slots:** 960-pixel input preserved more detail than the
  640-pixel candidates.
- **Adjacent-frame leakage:** rack-level grouping kept physical rack groups
  separated across train, validation, and test.
- **Apple MPS memory and portability:** one-epoch smoke tests established safe
  batch sizes and `workers=0` before the full runs.

## Lessons and Next Steps

The pipeline works, but the dataset must grow. Grouped splits protect the
evaluation, higher resolution helps small-slot detection, and a larger model
alone did not improve the current validation result. The next bounded dataset
milestone is a newly staged rack group with completed annotations, especially
clearer tube examples. Any resulting QA finding remains a possible issue
requiring human review.

## Milestone Plan

| Phase               | Goal                                                   | Milestone                            |
| ------------------- | ------------------------------------------------------ | ------------------------------------ |
| Blueprint           | Plan approved                                          | Midterm submitted                    |
| First Working Demo  | Run pretrained YOLO11/YOLO11-seg on staged rack images | End-to-end pipeline works            |
| Making It Ours      | Add our staged dataset, labels, and QA logic            | System works on the lab rack problem |
| Improve and Measure | Tune thresholds, test results, record metrics          | Metrics documented                   |
| Package and Present | Final README, demo notebook, slides, and video         | Final submitted                      |

## Risks and Plan B

**Risk 1: Dataset is too small or labels are inconsistent.**
Plan B: Reduce the class list to rack, tube, and cap, then use simple counting logic instead of trying to classify every issue.

**Risk 2: Segmentation masks are messy because tubes are transparent or reflective.**
Plan B: Use object detection boxes for the first version and keep segmentation as an enhancement.

**Risk 3: Lab privacy concerns limit usable images.**
Plan B: Use only staged images with empty tubes, fake labels, and no patient data.

## Resources Needed

- **Compute used for this run:** Apple M3 Pro with PyTorch MPS.
- **Tools:** Python, Ultralytics, OpenCV, Roboflow or Label Studio, GitHub
- **Cost:** Negligible, running on personally owned hardware.

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Analyze one staged held-out image with the fine-tuned checkpoint
python -m src.run \
  --image data/images/test/Rack_D_image00304_jpg.rf.DzfIrdoLsLIpJhLkkOfw.jpg
```

The fine-tuned checkpoint must exist at `weights/labrack_yolo11s_960.pt`. Model
weights are generated artifacts and are intentionally ignored by Git. The
selected checkpoint is published with the `v1.0.0` GitHub release as
[`labrack_yolo11s_960.pt`](https://github.com/RTaylor89/labrack-vision-qa/releases/download/v1.0.0/labrack_yolo11s_960.pt);
place that file in `weights/`. Its SHA-256 is
`3fa53efd7a2417b5a20812accd79e6d1eda54e71cbf0ed95f4e8825faf7a4246`.
To reproduce it from the staged dataset instead:

```bash
yolo detect train model=yolo11s.pt data=data/dataset.yaml \
  epochs=100 patience=20 imgsz=960 batch=2 device=mps workers=0 seed=0 \
  project=runs name=candidate_yolo11s_960 exist_ok=True plots=True

mkdir -p weights
cp runs/candidate_yolo11s_960/weights/best.pt \
  weights/labrack_yolo11s_960.pt
```

Use `device=0` on a CUDA system or `device=cpu` when MPS is unavailable.
Options for the POC command include `--output-dir`, `--model`, `--conf`, and
`--imgsz`.
Run all unit and dataset-validation tests with `pytest`.

## Demo Video

Demo video URL: **pending recording and upload**. The repository includes a
timed 3–5 minute narration plan in
[`docs/demo_script.md`](docs/demo_script.md). Replace this line with the final
shareable video link before submission.

For a clean Ubuntu workstation with an RTX 4090, follow the complete
[`Ubuntu + RTX 4090 CLI runbook`](docs/CLI_RUNBOOK_UBUNTU_4090.md). The
version-selectable, fully commented demonstration is in
[`notebooks/02_full_pipeline_demo.ipynb`](notebooks/02_full_pipeline_demo.ipynb).

## Repository Structure

```text
labrack-vision-qa/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py       # model, thresholds, class names, colors — one place
│   ├── detector.py     # YOLO11 inference only
│   ├── qa_rules.py     # counts, cap-vs-tube, empty slots, low-confidence flags
│   ├── annotate.py     # draw boxes/labels, save annotated image
│   ├── report.py       # JSON results + plain-text QA summary
│   └── run.py          # CLI: one image -> annotated + JSON + summary
├── tests/
│   ├── test_qa_rules.py
│   └── test_io.py
├── data/
│   ├── README.md
│   └── dataset.yaml    # YOLO classes + train/val/test paths
├── docs/
│   ├── CLI_RUNBOOK_UBUNTU_4090.md
│   ├── demo_script.md
│   ├── FP_LabRackVisionQA_Taylor_ITAI1378.pptx
│   ├── presentation.pdf
│   ├── proposal.pdf
│   └── AI_usage_log.md
├── results/
│   ├── metrics/
│   ├── poc/
│   └── predictions/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_full_pipeline_demo.ipynb
├── weights/            # generated checkpoint; ignored by Git
└── output/             # generated results
```

## AI Usage Log

[`docs/AI_usage_log.md`](docs/AI_usage_log.md) documents how AI tools supported
planning, debugging, implementation, verification, and presentation work. Each
entry records what I learned and what I kept or changed after checking the
result against the repository or a real run.
