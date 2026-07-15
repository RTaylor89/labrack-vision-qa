# LabRack Vision QA

> Educational prototype only. Not validated for clinical, diagnostic, or production laboratory use. Human review is required.

## Team Members

- Roderick Taylor
- Kate Leemann

## Project Tier

**Tier 1: Core Project**

I am choosing Tier 1 because this project is focused, realistic, and demo-friendly. It uses pretrained computer vision models with a small staged dataset instead of trying to build a production lab system from scratch.

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
| Make It Yours       | Add my staged dataset, labels, and QA logic            | System works on the lab rack problem |
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

- **Compute:** Google Colab free GPU or local VSCode/Jupyter
- **Tools:** Python, Ultralytics, OpenCV, Roboflow or Label Studio, GitHub
- **Cost:** $0 expected
- **Optional:** Small labeling/API cost only if needed later

## Repository Structure

```text
labrack-vision-qa/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   ├── proposal.pdf
│   └── AI_usage_log.md
└── notebooks/
    └── 01_exploration.ipynb
```

## AI Usage Log

I will maintain `docs/AI_usage_log.md` throughout the project. It documents how I used AI tools for planning, code debugging, explanation, and slide drafting. I verify outputs manually and make sure the final project reflects my own understanding and implementation.
