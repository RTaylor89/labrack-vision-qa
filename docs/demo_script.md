# LabRack Vision QA — 3–5 Minute Demo Script

We use this as our narration guide. We record the actual screen and keep the
visible terminal output. We do not replace it with fabricated screenshots.

## 0:00–0:30 — Problem and scope

- We introduce LabRack Vision QA as an educational computer-vision prototype for
  staged, non-patient rack images.
- We state that every finding is a possible issue requiring human review.
- We state that our current model detects racks, caps, and empty slots. Tube
  detection is a roadmap goal because the current images do not support
  consistent tube annotation.

## 0:30–1:05 — What was built

- We show our actual pipeline: validate image → YOLO11s/960 detection → cautious QA
  rules → annotated image + JSON + plain-text summary.
- We state that we use object detection with transfer learning in Ultralytics,
  with OpenCV used for image output.

## 1:05–2:20 — Live demo

We run the current M3-selected model with the command below, then open all three
generated artifacts.

```bash
source .venv/bin/activate
python -m src.run \
  --image data/images/test/Rack_D_image00304_jpg.rf.DzfIrdoLsLIpJhLkkOfw.jpg \
  --output-dir output/demo \
  --model weights/labrack_yolo11s_960.pt \
  --imgsz 960
```

- We show the terminal summary.
- We open the generated annotated image, JSON, and text summary.
- We point out the model path, class counts, confidence-review flag, disclaimer,
  and measured time. We describe all flags as possible issues requiring human
  review.

## 2:20–2:50 — Data

- We show `data/README.md` and `data/split_manifest.csv`.
- We explain the 140 staged images and grouped split: Rack_A/B train, Rack_C
  validation, Rack_D test.
- We explain that Roderick handled the programming and debugging. We credit Kate
  Leemann for collecting the staged photos, completing the annotations, and
  helping us design the testing methodology.

## 2:50–3:35 — Results versus targets

- We show `results/metrics/held_out_test_metrics.csv`.
- We show `results/metrics/candidate_comparison.csv`.
- We report selected validation mAP50 0.825, mAP50–95 0.594, and empty-slot recall
  0.858.
- We report the Rack_D diagnostic result: mAP50 0.937, mAP50–95 0.601, and
  empty-slot recall 0.761.
- We state that validation has five images and Rack_D already informed this
  iteration, so neither result establishes broad generalization.
- We compare the actual results with our Blueprint targets: mAP50 0.937 versus
  at least 0.75, 2.07-second full CLI wall time versus under 3 seconds, and all
  three required outputs.

## 3:35–4:05 — Changes, challenges, and fixes

- We explain our Blueprint changes: segmentation to detection, 200–250 planned
  images to 140 staged images, tube deferred, and YOLO11s/960 selected by
  validation.
- We explain our fixes: 960-pixel input for small slots, grouped rack splits, and
  MPS smoke tests with batch 2.

## 4:05–4:35 — Failure case

- We show the label and prediction views for `Rack_D_image00304...`.
- We report Kate's annotation: one rack, zero caps, 81 empty slots.
- We report our model prediction at confidence 0.25: one rack, one cap, seven possible empty
  positions.
- We explain that the remaining mismatch requires human review. We do not make a
  clinical claim.

## 4:35–5:00 — Lessons and next steps

- We summarize what works: controlled model selection, complete reproducible
  pipeline, annotated/JSON/text outputs, and test coverage.
- We state our remaining limitations: partial dataset, no tube instances, only
  five validation images, and no new pristine physical-rack test group.
- We state our next bounded step: add a newly staged rack group with completed
  labels, especially clearer tube examples.
- We remind viewers that our prototype supports human review and never approves a
  rack autonomously.
