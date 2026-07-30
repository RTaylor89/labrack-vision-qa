# LabRack Vision QA — 3–5 Minute Demo Script

I use this as my narration guide. I record my actual screen and keep the visible
terminal output. I do not replace it with fabricated screenshots.

## 0:00–0:30 — Problem and scope

- I introduce LabRack Vision QA as an educational computer-vision prototype for
  staged, non-patient rack images.
- I state that every finding is a possible issue requiring human review.
- I state that my current model detects racks, caps, and empty slots. Tube
  detection is a roadmap goal because the current images do not support
  consistent tube annotation.

## 0:30–1:05 — What was built

- I show my actual pipeline: validate image → YOLO11s/960 detection → cautious QA
  rules → annotated image + JSON + plain-text summary.
- I state that I use object detection with transfer learning in Ultralytics,
  with OpenCV used for image output.

## 1:05–2:20 — Live demo

I run the current M3-selected model with the command below, then open all three
generated artifacts.

```bash
source .venv/bin/activate
python -m src.run \
  --image data/images/test/Rack_D_image00304_jpg.rf.DzfIrdoLsLIpJhLkkOfw.jpg \
  --output-dir output/demo \
  --model weights/labrack_yolo11s_960.pt \
  --imgsz 960
```

- I show the terminal summary.
- I open the generated annotated image, JSON, and text summary.
- I point out the model path, class counts, confidence-review flag, disclaimer,
  and measured time. I describe all flags as possible issues requiring human
  review.

## 2:20–2:50 — Data

- I show `data/README.md` and `data/split_manifest.csv`.
- I explain the 140 staged images and grouped split: Rack_A/B train, Rack_C
  validation, Rack_D test.
- I explain that I handled the programming and debugging. I credit Kate Leemann
  for collecting the staged photos, completing the annotations, and assisting
  me in designing the testing methodology.

## 2:50–3:35 — Results versus targets

- I show `results/metrics/held_out_test_metrics.csv`.
- I show `results/metrics/candidate_comparison.csv`.
- I report selected validation mAP50 0.825, mAP50–95 0.594, and empty-slot recall
  0.858.
- I report the Rack_D diagnostic result: mAP50 0.937, mAP50–95 0.601, and
  empty-slot recall 0.761.
- I state that validation has five images and Rack_D already informed this
  iteration, so neither result establishes broad generalization.
- I compare the actual results with my Blueprint targets: mAP50 0.937 versus
  at least 0.75, 2.07-second full CLI wall time versus under 3 seconds, and all
  three required outputs.

## 3:35–4:05 — Changes, challenges, and fixes

- I explain my Blueprint changes: segmentation to detection, 200–250 planned
  images to 140 staged images, tube deferred, and YOLO11s/960 selected by
  validation.
- I explain my fixes: 960-pixel input for small slots, grouped rack splits, and
  MPS smoke tests with batch 2.

## 4:05–4:35 — Failure case

- I show the label and prediction views for `Rack_D_image00304...`.
- I report Kate's annotation: one rack, zero caps, 81 empty slots.
- I report my model prediction at confidence 0.25: one rack, one cap, seven possible empty
  positions.
- I explain that the remaining mismatch requires human review. I do not make a
  clinical claim.

## 4:35–5:00 — Lessons and next steps

- I summarize what works: controlled model selection, complete reproducible
  pipeline, annotated/JSON/text outputs, and test coverage.
- I state my remaining limitations: partial dataset, no tube instances, only
  five validation images, and no new pristine physical-rack test group.
- I state my next bounded step: add a newly staged rack group with completed
  labels, especially clearer tube examples.
- I remind viewers that my prototype supports human review and never approves a
  rack autonomously.
