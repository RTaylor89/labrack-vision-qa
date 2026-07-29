"""
smoke_train.py — prove the training loop works before you commit real time to it.

A "smoke test" is an InfoSec/SecDevOps best practice: before you trust a system, you run a
tiny, cheap check to see if it catches fire. Here we do a short training run
(default 20 epochs on our labeled images) just to confirm the whole loop is
wired correctly — data is found, my GPU is used, weights come out the other end.
It is NOT meant to produce a good model. That comes later with the full run.

Run it (from the repo root, with your venv active):

    python smoke_train.py

Some common overrides:

    python smoke_train.py --epochs 30 --batch 32
    python smoke_train.py --model yolo11s.pt          # compare a bigger model
    python smoke_train.py --data data/dataset.yaml    # explicit dataset path

Exit codes (so it plays nicely in a pipeline): 0 = ran, 1 = a check failed.
"""

import argparse
import sys
from pathlib import Path

import yaml  # PyYAML, already in requirements.txt


# We keep smoke defaults small and cheap on purpose. Think "does it boot," not
# "is it accurate." Real hyperparameters will live in the full run.
DEFAULT_DATA = "data/dataset.yaml"
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_EPOCHS = 20
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
RUN_NAME = "smoke"


def preflight_gpu():
    """
    Confirm PyTorch can see an available GPU before we train.

    Training on CPU by accident is the classic time-sink: it "works" but crawls.
    Like any good preflight check, we surface the state loudly and let the
    operator decide. Returns the device string YOLO should use ("0", "mps", or
    "cpu").
    """
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch not installed. See RUNBOOK.md §1.", file=sys.stderr)
        raise SystemExit(1)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[ OK ] GPU visible: {name}")
        return "0"  # use the first GPU (my 4090)

    if torch.backends.mps.is_available():
        print("[ OK ] Apple Metal GPU visible (MPS).")
        return "mps"

    # Not a hard stop — you *can* smoke-test on CPU, we just surface it clearly.
    print("[WARN] No CUDA or Apple Metal GPU visible to PyTorch — this will run "
          "on CPU and be slow. See RUNBOOK.md §1 for GPU setup.",
          file=sys.stderr)
    return "cpu"


def preflight_dataset(data_path):
    """
    Sanity-check the dataset before handing it to the trainer.

    Cheap validation up front beats a confusing crash 30 seconds into training.
    We confirm the yaml exists, then that the train/val image folders exist and
    actually contain images. Missing labels are YOLO's job to complain about; we
    just make sure there is something to train on.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"[FAIL] Dataset file not found: {data_path}. Label some images "
              f"and fill in {data_path.name} (RUNBOOK.md §3).", file=sys.stderr)
        raise SystemExit(1)

    with open(data_path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    if not isinstance(spec, dict):
        print(f"[FAIL] Dataset YAML must contain a mapping: {data_path}",
              file=sys.stderr)
        raise SystemExit(1)

    names = spec.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] if index in names else names[str(index)]
                     for index in range(len(names))]
        except KeyError:
            names = []
    elif isinstance(names, list):
        names = list(names)
    else:
        names = []

    expected_names = ["rack", "tube", "cap", "empty_slot"]
    if names != expected_names:
        print("[FAIL] Dataset classes must use the required names and order.",
              file=sys.stderr)
        print(f"       Expected: {expected_names}", file=sys.stderr)
        print(f"       Found:    {names}", file=sys.stderr)
        raise SystemExit(1)

    # Ultralytics resolves a relative "path" from the working directory.
    # Match that behavior so this preflight checks the same folders training uses.
    base = Path(spec.get("path", data_path.parent))
    split_counts = {}
    for split in ("train", "val"):
        folder = base / spec.get(split, f"images/{split}")
        images = []
        if folder.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                images += list(folder.glob(pattern))
        print(f"[ .. ] {split}: {len(images)} image(s) in {folder}")
        split_counts[split] = len(images)

    empty_splits = [name for name, count in split_counts.items() if count == 0]
    if empty_splits:
        print(f"[FAIL] Required dataset split(s) contain no images: "
              f"{', '.join(empty_splits)}. Add labeled images before "
              "smoke-training (RUNBOOK.md §2-3).", file=sys.stderr)
        raise SystemExit(1)

    image_count = sum(split_counts.values())
    print(f"[ OK ] Dataset looks trainable ({image_count} image(s) total).")


def run_smoke_train(args, device):
    """
    Kick off the short training run and report where the weights landed.

    We import ultralytics here (not at the top) so the preflight checks can run
    and fail fast without paying the cost of loading torch/ultralytics twice.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FAIL] ultralytics not installed. Run: pip install -r "
              "requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    print(f"[ .. ] Smoke-training {args.model} for {args.epochs} epoch(s) on "
          f"device={device} ...")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=0,          # reproducible: same run twice = same result
        project=str(Path("runs").resolve()),
        name=RUN_NAME,
        exist_ok=True,   # overwrite runs/smoke so we don't pile up smoke1, smoke2
        verbose=True,
    )

    # save_dir is where YOLO wrote everything for this run.
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    print()
    print("[ OK ] Smoke train finished — the loop works end to end.")
    print(f"       Run folder: {run_dir}")
    if best.exists():
        print(f"       Best weights: {best}")
        print(f"       Try them:  python -m src.run --image <photo> --model {best}")
    print("       This model is NOT tuned. Do the full run per RUNBOOK.md §4.")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Short 'does the training loop work' smoke test for "
                    "LabRack Vision QA.")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"Dataset yaml (default: {DEFAULT_DATA}).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Starting weights (default: {DEFAULT_MODEL}).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Epochs (default: {DEFAULT_EPOCHS}; keep it small).")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Image size (default: {DEFAULT_IMGSZ}).")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH}; -1 = auto).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    print("=== LabRack Vision QA — training smoke test ===")
    device = preflight_gpu()
    preflight_dataset(args.data)
    run_smoke_train(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
