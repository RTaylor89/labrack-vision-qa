# Results

I produced these artifacts with real Ultralytics YOLO11 and project-pipeline
runs on 2026-07-29. I used staged, non-patient images only.

## Selection run

- dataset: 125 train, 5 Rack_C validation, and 10 Rack_D diagnostic images
- trained classes: `rack`, `cap`, and `empty_slot`
- reserved class with no current instances: `tube`
- hardware: Apple M3 Pro using PyTorch MPS
- selected model: YOLO11s initialized from `yolo11s.pt`
- training: 100 epochs, patience 20, image size 960, batch 2, seed 0
- selected checkpoint: epoch 82 by validation mAP50–95
- training time: 4028.4 seconds

I completed both one-epoch YOLO11s fit checks before the candidate runs:
640 pixels with batch 4, then 960 pixels with batch 2. PyTorch warned that two
MPS operations do not have deterministic implementations. The seed was fixed,
but an identical rerun is not guaranteed to be bit-for-bit identical.

## Candidate comparison on Rack_C validation

| Candidate | mAP50 | mAP50–95 | Empty-slot precision | Empty-slot recall |
|---|---:|---:|---:|---:|
| YOLO11n / 640 baseline | 0.729 | 0.503 | 0.579 | 0.428 |
| YOLO11n / 960 | 0.809 | 0.539 | 0.775 | 0.470 |
| YOLO11s / 640 | 0.761 | 0.552 | 0.761 | 0.404 |
| **YOLO11s / 960** | **0.825** | **0.594** | **0.807** | **0.858** |
| YOLO11m / 640 | 0.730 | 0.520 | 0.742 | 0.381 |

The larger medium model did not improve validation recall, so I did not promote
it. I keep the complete machine-readable comparison in
`metrics/candidate_comparison.csv`.

## Selected validation result

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| all | 447 | 0.799 | 0.898 | 0.825 | 0.594 |
| rack | 5 | 0.872 | 1.000 | 0.995 | 0.876 |
| cap | 276 | 0.716 | 0.837 | 0.663 | 0.503 |
| empty_slot | 166 | 0.807 | 0.858 | 0.817 | 0.405 |

I do not report a tube metric because the tube class had no instances.

## Rack_D diagnostic result

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| all | 822 | 0.899 | 0.918 | 0.937 | 0.601 |
| rack | 10 | 0.923 | 1.000 | 0.995 | 0.765 |
| cap | 381 | 0.934 | 0.992 | 0.977 | 0.692 |
| empty_slot | 431 | 0.842 | 0.761 | 0.839 | 0.346 |

I originally grouped Rack_D away from training, but I used its baseline failures
to inform this retraining progression. I therefore describe it as diagnostic
held-out evidence, not a pristine final test. I need a new physical rack group
for an unbiased future final evaluation.

## Difficult POC image

`Rack_D_image00304...` has one rack, zero caps, and 81 empty slots in its
annotation. At confidence 0.25 the selected model reported one rack, one cap,
and seven possible empty positions. Eight detections were below the separate
0.50 review threshold, so the application recommended manual review.

I describe this as a possible issue requiring human review. I do not use the
prototype to approve a rack, make a clinical decision, or replace human
inspection.

## Performance

- application-recorded detection step: 0.732 seconds
- measured full CLI wall time: 2.07 seconds
- Ultralytics test inference: 37.1 ms per image, excluding application startup

## Artifact map

- `metrics/candidate_comparison.csv`: validation comparison used for selection
- `metrics/selected_validation_metrics.csv`: selected Rack_C metrics
- `metrics/held_out_test_metrics.csv`: Rack_D diagnostic metrics
- `metrics/pipeline_timing.csv`: measured application timing
- `metrics/training_results.csv` and `.png`: selected training history
- `metrics/test_*.png` and `.jpg`: selected Rack_D plots and batches
- `predictions/`: selected annotated Rack_D images and YOLO text output
- `poc/`: selected annotated image, JSON, and plain-text review summary
- `baseline_yolo11n_640/`: preserved pre-retraining evidence
