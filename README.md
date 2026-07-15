# LabRack Vision QA

> Educational prototype only. Not validated for clinical, diagnostic, or production laboratory use. Human review is required.

## Team Members

- Roderick Taylor
- Kate Leemann

## Project Tier

**Tier 1: Core Project**

We are choosing Tier 1 because this project is focused, realistic, and demo-friendly. It uses pretrained computer vision models with a small staged dataset instead of trying to build a production lab system from scratch.

## Problem Statement

Labs rely on consistent sample handling, and visual checks are part of keeping that workflow clean. Staff may need to confirm that a rack has the expected tubes, that tubes are capped, and that obvious placement issues are caught before the samples move further through the process.

The problem matters because small visual issues can create rework, delays, or extra QA review. This project is not a diagnostic system and will not use real patient data. It is a staged visual QA assistant for lab-style rack images.

## Solution Overview

LabRack Vision QA takes a photo of a staged lab sample rack and returns an annotated image showing detected objects such as racks, tubes, caps, and empty slots. It also generates a short QA-style summary that reports counts and flags simple visible issues.

The first version focuses on still images. The final capstone extension could add video, SAM-style promptable segmentation, a small dashboard, or an exportable QA report.

## Technical Approach

- **CV technique:** Object detection and instance segmentation
- **Model:** YOLO11 and YOLO11-seg
- **Framework:** Python, Ultralytics, OpenCV, VSCode/Jupyter

YOLO11 fits the project because the core task is finding and labeling objects in a photo. YOLO11-seg adds masks so bounding boxes can be compared against more precise object outlines when the image is crowded or when rack/tube boundaries overlap.

## Data Plan

- **Source:** Self-collected staged photos using empty tubes, racks, fake labels, and no PHI.
- **Initial size:** 200–250 images for the first working demo.
- **Target size:** 100–150 labeled staged images for training and validation.
- **Holdout test set:** 50–100 images.

**Labels:**

- rack
- tube
- cap
- empty slot
- tilted tube or abnormal tube position, if labeling is consistent enough

**Data handling rule:** No real patient labels, no real sample IDs, and no medical decision-making.

## Success Metrics

- **Primary metric:** At least 0.75 mAP50 on the holdout test set for the core classes: rack, tube, cap, and empty slot.
- **Secondary metric:** Process one image and generate an annotated output plus a QA summary in under 3 seconds per image.
- **Demo success:** The project produces an annotated rack image and a plain-English summary that a lab worker could quickly review.

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

- **Compute:** Local GPU (RTX 4090) on VSCode/Jupyter
- **Tools:** Python, Ultralytics, OpenCV, Roboflow or Label Studio, GitHub
- **Cost:** Negligible, running on personally owned hardware.

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Analyze one staged rack image (pretrained yolo11n.pt downloads on first run)
python -m src.run --image input/rack_001.jpg
```

For `input/rack_001.jpg` this writes `output/rack_001_annotated.jpg`,
`output/rack_001_results.json`, and `output/rack_001_summary.txt`.

Options: `--output-dir`, `--model` (swap in fine-tuned weights), `--conf`
(detection threshold). Run the QA-rule and validation tests with `pytest`.

> Note: the demo uses pretrained `yolo11n.pt`, whose COCO classes are not lab
> objects — the pipeline runs end to end, but real rack/tube/cap counts require
> a fine-tuned model. Point `MODEL_PATH` in `src/config.py` at your weights.

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
│   ├── proposal.pdf
│   └── AI_usage_log.md
├── notebooks/
│   └── 01_exploration.ipynb
├── input/              # place staged images here
└── output/             # generated results
```

## AI Usage Log

I will maintain `docs/AI_usage_log.md` throughout the project. It documents how I used AI tools for planning, code debugging, explanation, and slide drafting. I verify outputs manually and make sure the final project reflects my own understanding and implementation.
