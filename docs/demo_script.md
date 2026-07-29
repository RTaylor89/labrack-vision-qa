# LabRack Vision QA — 3–5 Minute Demo Script

Use this as a narration guide. Record the actual screen and keep the visible
terminal output; do not replace it with fabricated screenshots.

## 0:00–0:30 — Problem and scope

- Introduce LabRack Vision QA as an educational computer-vision prototype for
  staged, non-patient rack images.
- State that every finding is a possible issue requiring human review.
- State that the current model detects racks, caps, and empty slots. Tube
  detection is a roadmap goal because the current images do not support
  consistent tube annotation.

## 0:30–1:05 — What was built

- Show the actual pipeline: validate image → YOLO11s/960 detection → cautious QA
  rules → annotated image + JSON + plain-text summary.
- State that this is object detection with transfer learning in Ultralytics,
  with OpenCV used for image output.

## 1:05–2:20 — Live demo

Run the command shown below, then open all three generated artifacts.

```bash
source .venv/bin/activate
python -m src.run \
  --image data/images/test/Rack_D_image00304_jpg.rf.DzfIrdoLsLIpJhLkkOfw.jpg \
  --output-dir output/demo \
  --model weights/labrack_yolo11s_960.pt \
  --imgsz 960
```

- Show the terminal summary.
- Open the generated annotated image, JSON, and text summary.
- Point out the model path, class counts, confidence-review flag, disclaimer,
  and measured time. Describe all flags as possible issues requiring human
  review.

## 2:20–2:50 — Data

- Show `data/README.md` and `data/split_manifest.csv`.
- Explain the 140 staged images and grouped split: Rack_A/B train, Rack_C
  validation, Rack_D test.
- Credit Kate Leemann for the staged photography, image annotations, and testing
  methodology.

## 2:50–3:35 — Results versus targets

- Show `results/metrics/held_out_test_metrics.csv`.
- Show `results/metrics/candidate_comparison.csv`.
- Report selected validation mAP50 0.825, mAP50–95 0.594, and empty-slot recall
  0.858.
- Report the Rack_D diagnostic result: mAP50 0.937, mAP50–95 0.601, and
  empty-slot recall 0.761.
- State that validation has five images and Rack_D already informed this
  iteration, so neither result establishes broad generalization.
- Compare the actual results with the Blueprint targets: mAP50 0.937 versus
  at least 0.75, 2.07-second full CLI wall time versus under 3 seconds, and all
  three required outputs.

## 3:35–4:05 — Changes, challenges, and fixes

- Explain the Blueprint changes: segmentation to detection, 200–250 planned
  images to 140 staged images, tube deferred, and YOLO11s/960 selected by
  validation.
- Explain the fixes: 960-pixel input for small slots, grouped rack splits, and
  MPS smoke tests with batch 2.

## 4:05–4:35 — Failure case

- Show the label and prediction views for `Rack_D_image00304...`.
- Ground truth: one rack, zero caps, 81 empty slots.
- Prediction at confidence 0.25: one rack, one cap, seven possible empty
  positions.
- Explain that the remaining mismatch requires human review. Do not make a
  clinical claim.

## 4:35–5:00 — Lessons and next steps

- Summarize what works: controlled model selection, complete reproducible
  pipeline, annotated/JSON/text outputs, and test coverage.
- State the remaining limitations: partial dataset, no tube instances, only
  five validation images, and no new pristine physical-rack test group.
- State the next bounded step: add a newly staged rack group with completed
  labels, especially clearer tube examples.
- Remind viewers that the prototype supports human review and never approves a
  rack autonomously.
