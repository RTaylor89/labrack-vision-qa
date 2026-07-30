# LabRack Vision QA

> Educational prototype only. Not validated for clinical, diagnostic, or production laboratory use. Human review is required.

## Contributors

- Roderick Taylor
- Kate Leemann

## Contribution Roles

- I, Roderick Taylor, handled the programming, debugging, model training,
  evaluation, and technical deliverables.
- Kate Leemann collected the staged laboratory photos, completed the image
  annotations, and assisted me in designing the testing methodology.

## Project Tier

**Tier 1: Core Project**

I chose Tier 1 because I wanted a focused, realistic, and demo-friendly
project. I used pretrained computer-vision models with a small staged dataset
instead of trying to build a production laboratory system.

## Problem Statement

I focused on visual rack checks because laboratories rely on consistent sample
handling. Staff may need to confirm that a rack has the expected tubes, that
tubes are capped, and that obvious placement issues are caught before samples
move further through the process.

I treated this as a useful computer-vision problem because small visual issues
can create rework, delays, or extra QA review. I did not build a diagnostic
system and did not use real patient data. I built a staged visual QA assistant
for laboratory-style rack images.

## Solution Overview

I built LabRack Vision QA to take a photo of a staged sample rack and return an
annotated image showing detected objects such as racks, caps, and empty slots. I
also generate a short QA-style summary that reports counts and flags possible
visible issues for human review.

I limited Version 1 to still images. I kept video, SAM-style promptable
segmentation, a small dashboard, and an exportable QA report outside the
current scope.

## Current Build Status

My Version 1 pipeline runs end to end on staged, non-patient still images:

1. I validate an image path.
2. I run a fine-tuned Ultralytics YOLO11s object detector at 960 pixels.
3. I count detected rack, cap, and empty-slot objects.
4. I describe possible issues that require human review.
5. I write an annotated image, structured JSON, and plain-text summary.

I did not train or report tube performance because the current partial dataset
does not contain usable tube annotations. I document provenance, class
remapping, and split details in [`data/README.md`](data/README.md).

## Future RTX 4090 Dataset Version

I use the measured Apple M3 run below as my current completed submission. After
Kate and I finish the next reviewed dataset version, I plan to train a new
controlled matrix containing YOLO11n, YOLO11s, and YOLO11m at both 640 and 960
pixels on my Ubuntu RTX 4090 workstation.

For that future version, every candidate will start from an isolated official
Ultralytics YOLO11 base checkpoint with `resume=False`; none will start from the
current LabRack checkpoint. I will select the future model on validation using
a rule declared before training: maximum `empty_slot` recall, then overall
mAP50–95 and overall mAP50 as tie-breakers. I will update the repository only
after I verify the resulting evidence package.

## Technical Approach

- **CV technique:** Object detection
- **Model:** Fine-tuned YOLO11s at 960-pixel input
- **Framework:** Python, Ultralytics, OpenCV, VSCode/Jupyter

I chose YOLO11 because my core task is finding and labeling objects in a photo.
I kept segmentation as a possible later experiment if bounding boxes prove
insufficient around crowded or overlapping objects.

## Dataset

- **Source:** Kate collected staged photos using empty tubes, racks, fake labels,
  and no PHI.
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

- **Primary target:** I targeted at least 0.75 mAP50 on the holdout test set for the
  currently trained classes.
- **Secondary metric:** I targeted under 3 seconds to process one image and
  generate an annotated output plus a QA summary.
- **Demo success:** I required my project to produce an annotated rack image and
  a plain-English summary that a laboratory worker could quickly review.

## Measured M3 Results

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

I selected the following validation result:

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
I describe that mismatch as a possible issue requiring human review. It
demonstrates why I do not use this prototype to approve a rack autonomously.

See [`results/README.md`](results/README.md), the metric plots in
[`results/metrics/`](results/metrics/), and the real annotated samples in
[`results/predictions/`](results/predictions/).

## What Changed From the Blueprint

- I changed the Blueprint's proposed segmentation to object detection because
  bounding boxes are sufficient for the current counting and review workflow.
- I completed this iteration with a 140-image partial staged export instead of
  the planned 200–250 images.
- I reserved the planned `tube` class but did not measure it because the current
  images do not support consistent tube annotation.
- I selected YOLO11s at 960 pixels through controlled candidate runs instead of
  choosing a model size in advance.

I changed the scope to fit the available staged data while preserving my
end-to-end promise: one image produces an annotated image, JSON, and a
plain-English summary for human review.

## Challenges and Fixes

- **Small empty slots:** I used 960-pixel input because it preserved more detail than the
  640-pixel candidates.
- **Adjacent-frame leakage:** I used rack-level grouping to keep physical rack groups
  separated across train, validation, and test.
- **Apple MPS memory and portability:** I used one-epoch smoke tests to establish
  safe batch sizes and `workers=0` before the full runs.

## Lessons and Next Steps

I learned that my pipeline works, but my dataset must grow. I use grouped splits
to protect the evaluation, I use higher resolution to help small-slot
detection, and I found that a larger model alone did not improve my validation
result. My next bounded dataset milestone is a newly staged rack group with
completed annotations, especially clearer tube examples. I will continue to
describe every QA finding as a possible issue requiring human review.

## Milestone Plan

| Phase               | Goal                                                   | Milestone                            |
| ------------------- | ------------------------------------------------------ | ------------------------------------ |
| Blueprint           | Plan approved                                          | Midterm submitted                    |
| First Working Demo  | Run pretrained YOLO11/YOLO11-seg on staged rack images | End-to-end pipeline works            |
| Making It Mine      | Add the staged dataset, labels, and my QA logic          | My system works on the lab rack problem |
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

- **Current measured run:** Apple M3 Pro with PyTorch MPS.
- **Future dataset version:** My Ubuntu workstation with an NVIDIA RTX 4090.
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

I keep the current fine-tuned checkpoint at
`weights/labrack_yolo11s_960.pt`. Model
weights are generated artifacts and are intentionally ignored by Git. The
selected checkpoint is published with the `v1.0.0` GitHub release as
[`labrack_yolo11s_960.pt`](https://github.com/RTaylor89/labrack-vision-qa/releases/download/v1.0.0/labrack_yolo11s_960.pt);
place that file in `weights/`. Its SHA-256 is
`3fa53efd7a2417b5a20812accd79e6d1eda54e71cbf0ed95f4e8825faf7a4246`.
After the next reviewed dataset version is available, I will inspect and run
the clean six-candidate RTX 4090 matrix:

```bash
export LABRACK_4090_OUTPUT="$HOME/labrack-4090-output"

python scripts/run_rtx4090_matrix.py \
  --data data/dataset.yaml \
  --output-root "$LABRACK_4090_OUTPUT" \
  --run-id final_rtx4090 \
  --dry-run

python scripts/run_rtx4090_matrix.py \
  --data data/dataset.yaml \
  --output-root "$LABRACK_4090_OUTPUT" \
  --run-id "final_rtx4090_$(date -u +%Y%m%d)"
```

I use a script that refuses CPU/MPS for that future run, requires my RTX 4090 by
default, trains all six candidates, packages the evidence, and never resumes an
earlier run. I follow
[`docs/CLI_RUNBOOK_UBUNTU_4090.md`](docs/CLI_RUNBOOK_UBUNTU_4090.md) for the
complete setup, smoke, verification, and return procedure. I can pass
`--output-dir`, `--model`, `--conf`, and `--imgsz` to the POC command. I run all
unit and dataset-validation tests with `pytest`.

## Demo Video

Demo video URL: **pending my recording and upload**. I included a timed 3–5
minute narration plan in [`docs/demo_script.md`](docs/demo_script.md). I will
replace this line with my final shareable video link before submission.

For my clean Ubuntu workstation with an RTX 4090, I follow the complete
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
│   ├── test_io.py
│   └── test_rtx4090_pipeline.py
├── scripts/
│   └── run_rtx4090_matrix.py
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
