#!/usr/bin/env python3
"""
Run my complete fresh YOLO11 candidate matrix on an Ubuntu RTX 4090 system.

I start every candidate from an isolated official Ultralytics YOLO11
COCO-pretrained checkpoint. I never load a LabRack checkpoint, never resume a
training run, and never overwrite an existing run ID.

I train these six candidates:

    YOLO11n at 640 and 960 pixels
    YOLO11s at 640 and 960 pixels
    YOLO11m at 640 and 960 pixels

I evaluate every candidate on the validation split. I select the candidate with
the highest empty-slot recall, using overall mAP50-95 and mAP50 as tie-breakers.
Only after selection do I evaluate the selected checkpoint on the configured
test split. I describe that test result as diagnostic when the split has
already informed development.

I write raw Ultralytics runs to the external output directory and build a
smaller handoff ZIP containing the evidence I need to update the repository's
README, results, notebook, slides, PDFs, and selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # I add the repository root because this script lives one directory below
    # the reusable src package when I launch it as a file.
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_CLASSES = ["rack", "tube", "cap", "empty_slot"]
OFFICIAL_BASE_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
DEFAULT_DATASET = REPO_ROOT / "data" / "dataset.yaml"
DEFAULT_DEMO_IMAGE = (
    REPO_ROOT
    / "data"
    / "images"
    / "test"
    / "Rack_D_image00304_jpg.rf.DzfIrdoLsLIpJhLkkOfw.jpg"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Candidate:
    """I keep each controlled model/resolution choice in one immutable record."""

    model: str
    imgsz: int
    batch: int

    @property
    def name(self) -> str:
        return f"{Path(self.model).stem}_{self.imgsz}"

    @property
    def model_size(self) -> str:
        return Path(self.model).stem.removeprefix("yolo11")


CANDIDATES = (
    Candidate("yolo11n.pt", 640, 16),
    Candidate("yolo11n.pt", 960, 8),
    Candidate("yolo11s.pt", 640, 16),
    Candidate("yolo11s.pt", 960, 8),
    Candidate("yolo11m.pt", 640, 16),
    Candidate("yolo11m.pt", 960, 8),
)


def sha256_file(path: Path) -> str:
    """I hash inputs and outputs so I can prove exactly which files I used."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_class_names(names: Any) -> list[str]:
    """I normalize either supported YAML class-name representation."""

    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        try:
            return [
                str(names[index] if index in names else names[str(index)])
                for index in range(len(names))
            ]
        except (KeyError, TypeError):
            return []
    return []


def load_dataset_spec(dataset_yaml: Path) -> dict[str, Any]:
    """I validate the required schema and all three dataset splits."""

    if not dataset_yaml.is_file():
        raise ValueError(f"Dataset YAML does not exist: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError("Dataset YAML must contain a mapping.")

    names = normalized_class_names(spec.get("names"))
    if names != EXPECTED_CLASSES:
        raise ValueError(
            f"Dataset classes must be {EXPECTED_CLASSES}; found {names}."
        )

    dataset_root = Path(spec.get("path", dataset_yaml.parent))
    if not dataset_root.is_absolute():
        # This matches how this project runs Ultralytics from the repository root.
        dataset_root = (REPO_ROOT / dataset_root).resolve()

    split_paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        value = spec.get(split)
        if not value:
            raise ValueError(f"Dataset YAML is missing the {split!r} split.")
        split_path = Path(value)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        split_path = split_path.resolve()
        images = [
            path
            for path in split_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ] if split_path.is_dir() else []
        if not images:
            raise ValueError(f"Dataset split contains no images: {split_path}")
        split_paths[split] = split_path

    spec["_resolved_root"] = dataset_root
    spec["_resolved_splits"] = split_paths
    return spec


def dataset_files(dataset_yaml: Path, spec: dict[str, Any]) -> list[Path]:
    """I collect the exact YAML, images, labels, and split manifest I trained on."""

    files = {dataset_yaml.resolve()}
    split_paths: dict[str, Path] = spec["_resolved_splits"]
    for image_dir in split_paths.values():
        files.update(
            path.resolve()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_dir = Path(str(image_dir).replace("/images/", "/labels/"))
        if label_dir.is_dir():
            files.update(path.resolve() for path in label_dir.glob("*.txt"))

    split_manifest = REPO_ROOT / "data" / "split_manifest.csv"
    if split_manifest.is_file():
        files.add(split_manifest.resolve())
    return sorted(files)


def validate_run_id(run_id: str) -> str:
    """I keep run IDs portable and safe to use as directory names."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError(
            "Run ID must be 1-80 characters using letters, numbers, '.', '_', "
            "or '-'."
        )
    return run_id


def select_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """I apply the selection rule declared before training."""

    rows = list(rows)
    if len(rows) != len(CANDIDATES):
        raise ValueError(
            f"I require all {len(CANDIDATES)} completed candidates; "
            f"found {len(rows)}."
        )
    if any(row.get("empty_slot_recall") is None for row in rows):
        raise ValueError("Every candidate must have an empty-slot recall value.")
    return max(
        rows,
        key=lambda row: (
            float(row["empty_slot_recall"]),
            float(row["validation_map50_95"]),
            float(row["validation_map50"]),
        ),
    )


def write_json(path: Path, payload: Any) -> None:
    """I write stable, readable JSON for the handoff."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """I write a CSV with the union of the row fields in first-seen order."""

    if not rows:
        raise ValueError(f"I cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    """I convert numeric values without turning missing metrics into fake zeros."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_index(values: Any, index: int) -> float | None:
    """I read one per-class metric from NumPy, tensor, or list-like output."""

    try:
        return safe_float(values[index])
    except (IndexError, KeyError, TypeError):
        return None


def metric_rows(metrics: Any) -> list[dict[str, Any]]:
    """I convert Ultralytics detection metrics into a stable project schema."""

    box = metrics.box
    names = {
        int(index): str(name)
        for index, name in dict(metrics.names).items()
    }
    counts = getattr(metrics, "nt_per_class", [])
    rows = [
        {
            "class": "all",
            "images": int(getattr(metrics, "seen", 0)),
            "instances": int(sum(counts)) if len(counts) else None,
            "precision": safe_float(box.mp),
            "recall": safe_float(box.mr),
            "map50": safe_float(box.map50),
            "map50_95": safe_float(box.map),
        }
    ]
    for index, name in names.items():
        rows.append(
            {
                "class": name,
                "images": int(getattr(metrics, "seen", 0)),
                "instances": (
                    int(counts[index]) if index < len(counts) else None
                ),
                "precision": safe_index(box.p, index),
                "recall": safe_index(box.r, index),
                "map50": safe_index(box.ap50, index),
                "map50_95": safe_index(box.ap, index),
            }
        )
    return rows


def metric_row_for(rows: list[dict[str, Any]], class_name: str) -> dict[str, Any]:
    """I require the metric row used by the selection rule."""

    for row in rows:
        if row["class"] == class_name:
            return row
    raise ValueError(f"Validation metrics do not contain class {class_name!r}.")


def training_history_summary(results_csv: Path) -> dict[str, Any]:
    """I report completed epochs and the epoch with the highest mAP50-95."""

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {results_csv}")

    metric_field = next(
        (field for field in rows[0] if "mAP50-95" in field),
        None,
    )
    best_epoch = None
    if metric_field:
        valid = [
            (index + 1, safe_float(row.get(metric_field)))
            for index, row in enumerate(rows)
        ]
        valid = [(epoch, value) for epoch, value in valid if value is not None]
        if valid:
            best_epoch = max(valid, key=lambda item: item[1])[0]

    return {
        "epochs_completed": len(rows),
        "best_map50_95_epoch": best_epoch,
    }


def run_command(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """I capture supporting command output without hiding a required failure."""

    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def capture_environment(environment_dir: Path, torch: Any, ultralytics: Any) -> dict[str, Any]:
    """I record the software, repository, CUDA, and GPU state used for training."""

    import cv2

    environment_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0)
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": gpu_name,
    }
    write_json(environment_dir / "environment.json", payload)

    commands = {
        "nvidia-smi.txt": ["nvidia-smi"],
        "pip-freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "git-commit.txt": ["git", "rev-parse", "HEAD"],
        "git-status.txt": ["git", "status", "--short"],
    }
    for filename, command in commands.items():
        result = run_command(command)
        (environment_dir / filename).write_text(
            result.stdout or f"Command exited {result.returncode} with no output.\n",
            encoding="utf-8",
        )
    return payload


def copy_evidence(source: Path, destination: Path) -> None:
    """I copy plots, tables, configuration, and logs while omitting last.pt."""

    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "last.pt":
            continue
        relative = path.relative_to(source)
        if relative == Path("weights") / "best.pt":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def write_checksums(root: Path) -> None:
    """I checksum every handoff file except the checksum list itself."""

    checksum_path = root / "CHECKSUMS.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """I define one explicit command for the authoritative CUDA rerun."""

    default_run_id = datetime.now(timezone.utc).strftime("rtx4090_%Y%m%d_%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLO11n/s/m at 640/960 from isolated official base weights, "
            "evaluate, select, and package the RTX 4090 evidence."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-image", type=Path, default=DEFAULT_DEMO_IMAGE)
    parser.add_argument(
        "--allow-other-cuda-gpu",
        action="store_true",
        help="Allow CUDA hardware whose name does not contain '4090'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable six-candidate plan without training.",
    )
    return parser.parse_args(argv)


def print_plan(args: argparse.Namespace) -> None:
    """I print the exact controlled matrix before I spend GPU time."""

    print("Fresh RTX 4090 candidate matrix")
    print(f"  dataset: {args.data}")
    print(f"  output:  {args.output_root}")
    print(f"  run ID:  {args.run_id}")
    print(
        f"  epochs={args.epochs} patience={args.patience} "
        f"workers={args.workers} device={args.device} seed={args.seed}"
    )
    for candidate in CANDIDATES:
        print(
            f"  - {candidate.name}: base={candidate.model} "
            f"imgsz={candidate.imgsz} batch={candidate.batch}"
        )
    print("  resume=False for every candidate")
    print(
        "  selection=max(empty_slot recall), then max(mAP50-95), "
        "then max(mAP50)"
    )


def main(argv: list[str] | None = None) -> int:
    """I run the full matrix and create the evidence handoff ZIP."""

    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.run_id = validate_run_id(args.run_id)
    args.data = args.data.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.demo_image = args.demo_image.expanduser().resolve()
    print_plan(args)
    if args.dry_run:
        return 0

    if args.epochs < 1 or args.patience < 0 or args.workers < 0:
        raise ValueError("Epochs must be positive; patience/workers cannot be negative.")

    spec = load_dataset_spec(args.data)
    run_root = args.output_root / args.run_id
    handoff_root = args.output_root / f"{args.run_id}_handoff"
    handoff_zip = args.output_root / f"{args.run_id}_handoff.zip"
    handoff_sha256 = handoff_zip.with_suffix(".zip.sha256")
    conflicts = [
        path
        for path in (run_root, handoff_root, handoff_zip, handoff_sha256)
        if path.exists()
    ]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"I refuse to reuse or overwrite an existing RTX run: {joined}"
        )

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. I will not run the final matrix on CPU.")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090" not in gpu_name and not args.allow_other_cuda_gpu:
        raise RuntimeError(
            f"Expected an RTX 4090, but PyTorch reports {gpu_name!r}. "
            "I can override this only with --allow-other-cuda-gpu."
        )

    run_root.mkdir(parents=True)
    base_dir = run_root / "official_base_models"
    train_project = run_root / "train"
    validation_project = run_root / "validation"
    environment_dir = run_root / "environment"
    base_dir.mkdir()
    environment = capture_environment(environment_dir, torch, ultralytics)
    started_at = datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "status": "running",
        "run_id": args.run_id,
        "started_at_utc": started_at,
        "dataset_yaml": str(args.data),
        "dataset_yaml_sha256": sha256_file(args.data),
        "output_root": str(args.output_root),
        "starting_weight_policy": (
            "Isolated official Ultralytics YOLO11 COCO-pretrained checkpoints; "
            "no LabRack checkpoint; resume=False."
        ),
        "selection_rule": [
            "maximum validation empty_slot recall",
            "maximum validation overall mAP50-95 tie-breaker",
            "maximum validation overall mAP50 second tie-breaker",
        ],
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "device": args.device,
            "workers": args.workers,
            "amp": True,
            "seed": args.seed,
            "deterministic": True,
            "cache": False,
            "resume": False,
        },
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
    }
    write_json(run_root / "run_manifest.json", run_manifest)

    dataset_manifest = []
    for path in dataset_files(args.data, spec):
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = str(path)
        dataset_manifest.append({"path": label, "sha256": sha256_file(path)})
    write_json(run_root / "dataset_manifest.json", dataset_manifest)

    candidate_rows: list[dict[str, Any]] = []
    candidate_details: list[dict[str, Any]] = []
    base_hashes: dict[str, str] = {}

    for candidate in CANDIDATES:
        base_path = base_dir / candidate.model
        if candidate.model not in OFFICIAL_BASE_MODELS:
            raise ValueError(f"Unapproved starting checkpoint: {candidate.model}")

        if not base_path.exists():
            print(f"\n[download] {candidate.model} -> {base_path}")
            YOLO(str(base_path))
        base_hash = sha256_file(base_path)
        base_hashes[candidate.model] = base_hash

        train_name = candidate.name
        expected_train_dir = train_project / train_name
        if expected_train_dir.exists():
            raise FileExistsError(f"Candidate directory already exists: {expected_train_dir}")

        print(f"\n[train] {candidate.name} from {base_path}")
        started = time.perf_counter()
        model = YOLO(str(base_path))
        train_result = model.train(
            data=str(args.data),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=candidate.imgsz,
            batch=candidate.batch,
            device=args.device,
            workers=args.workers,
            amp=True,
            seed=args.seed,
            deterministic=True,
            cache=False,
            project=str(train_project),
            name=train_name,
            exist_ok=False,
            resume=False,
            plots=True,
            save=True,
            verbose=True,
        )
        training_seconds = time.perf_counter() - started
        train_dir = Path(train_result.save_dir).resolve()
        if train_dir != expected_train_dir.resolve():
            raise RuntimeError(
                f"Ultralytics changed the run directory to {train_dir}; "
                f"I expected {expected_train_dir}."
            )
        best_path = train_dir / "weights" / "best.pt"
        results_csv = train_dir / "results.csv"
        if not best_path.is_file() or not results_csv.is_file():
            raise RuntimeError(f"Candidate did not produce required artifacts: {train_dir}")

        print(f"[validate] {candidate.name} on validation split")
        best_model = YOLO(str(best_path))
        validation = best_model.val(
            data=str(args.data),
            split="val",
            imgsz=candidate.imgsz,
            batch=candidate.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            save_json=False,
            project=str(validation_project),
            name=train_name,
            exist_ok=False,
            verbose=True,
        )
        validation_dir = Path(validation.save_dir).resolve()
        rows = metric_rows(validation)
        all_metrics = metric_row_for(rows, "all")
        empty_metrics = metric_row_for(rows, "empty_slot")
        history = training_history_summary(results_csv)

        comparison = {
            "candidate": candidate.name,
            "model": f"YOLO11{candidate.model_size}",
            "image_size": candidate.imgsz,
            "batch": candidate.batch,
            **history,
            "training_seconds": round(training_seconds, 3),
            "validation_map50": all_metrics["map50"],
            "validation_map50_95": all_metrics["map50_95"],
            "empty_slot_precision": empty_metrics["precision"],
            "empty_slot_recall": empty_metrics["recall"],
            "empty_slot_map50": empty_metrics["map50"],
            "empty_slot_map50_95": empty_metrics["map50_95"],
            "selected": "no",
        }
        candidate_rows.append(comparison)
        candidate_details.append(
            {
                **asdict(candidate),
                "candidate": candidate.name,
                "official_base_path": str(base_path),
                "official_base_sha256": base_hash,
                "best_checkpoint": str(best_path),
                "best_checkpoint_sha256": sha256_file(best_path),
                "train_directory": str(train_dir),
                "validation_directory": str(validation_dir),
                "validation_metrics": rows,
                **history,
                "training_seconds": round(training_seconds, 3),
            }
        )
        write_csv(validation_dir / "validation_metrics.csv", rows)
        write_json(validation_dir / "validation_metrics.json", rows)
        write_csv(run_root / "candidate_comparison.partial.csv", candidate_rows)
        write_json(run_root / "candidate_details.partial.json", candidate_details)

        # I release each trained model before loading the next official base.
        del model, train_result, best_model, validation
        torch.cuda.empty_cache()

    selected = select_candidate(candidate_rows)
    for row in candidate_rows:
        row["selected"] = "yes" if row["candidate"] == selected["candidate"] else "no"
    selected_detail = next(
        item for item in candidate_details
        if item["candidate"] == selected["candidate"]
    )
    write_csv(run_root / "candidate_comparison.csv", candidate_rows)
    write_json(run_root / "candidate_details.json", candidate_details)

    selected_imgsz = int(selected["image_size"])
    selected_batch = int(selected["batch"])
    selected_best = Path(selected_detail["best_checkpoint"])
    diagnostic_project = run_root / "diagnostic_test"

    print(f"\n[select] {selected['candidate']}")
    print("[test] I now evaluate only the selected checkpoint on the test split.")
    selected_model = YOLO(str(selected_best))
    diagnostic = selected_model.val(
        data=str(args.data),
        split="test",
        imgsz=selected_imgsz,
        batch=selected_batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        save_json=False,
        save_txt=True,
        save_conf=True,
        project=str(diagnostic_project),
        name=selected["candidate"],
        exist_ok=False,
        verbose=True,
    )
    diagnostic_dir = Path(diagnostic.save_dir).resolve()
    diagnostic_rows = metric_rows(diagnostic)
    write_csv(diagnostic_dir / "diagnostic_test_metrics.csv", diagnostic_rows)
    write_json(diagnostic_dir / "diagnostic_test_metrics.json", diagnostic_rows)

    test_images: Path = spec["_resolved_splits"]["test"]
    predictions_project = run_root / "predictions"
    selected_model.predict(
        source=str(test_images),
        imgsz=selected_imgsz,
        conf=0.25,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(predictions_project),
        name=selected["candidate"],
        exist_ok=False,
        verbose=True,
    )
    predictions_dir = predictions_project / selected["candidate"]

    demo_image = args.demo_image
    if not demo_image.is_file():
        demo_image = next(
            path for path in sorted(test_images.iterdir())
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
    poc_dir = run_root / "poc"
    from src.run import process_image

    wall_started = time.perf_counter()
    poc_results, _, poc_paths = process_image(
        demo_image,
        output_dir=poc_dir,
        model_path=str(selected_best),
        confidence_threshold=0.25,
        image_size=selected_imgsz,
    )
    wall_seconds = time.perf_counter() - wall_started
    pipeline_timing = {
        "demo_image": str(demo_image),
        "candidate": selected["candidate"],
        "image_size": selected_imgsz,
        "application_inference_seconds": poc_results["inference_seconds"],
        "application_wall_seconds": wall_seconds,
        "outputs": {key: str(value) for key, value in poc_paths.items()},
    }
    write_json(run_root / "pipeline_timing.json", pipeline_timing)

    selected_dir = run_root / "selected"
    selected_weights_dir = selected_dir / "weights"
    selected_weights_dir.mkdir(parents=True)
    selected_weight_name = (
        f"labrack_{selected['candidate']}_rtx4090.pt"
    )
    selected_weight = selected_weights_dir / selected_weight_name
    shutil.copy2(selected_best, selected_weight)

    selection = {
        "selection_rule": [
            "maximum validation empty_slot recall",
            "maximum validation overall mAP50-95 tie-breaker",
            "maximum validation overall mAP50 second tie-breaker",
        ],
        "selected_candidate": selected,
        "selected_checkpoint": str(selected_weight),
        "selected_checkpoint_sha256": sha256_file(selected_weight),
        "test_split_interpretation": (
            "Diagnostic evidence requiring human interpretation; I do not claim "
            "a pristine test if this split informed development."
        ),
    }
    write_json(run_root / "selection.json", selection)
    run_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_base_sha256": base_hashes,
            "selected_candidate": selected["candidate"],
            "selected_checkpoint_sha256": sha256_file(selected_weight),
        }
    )
    write_json(run_root / "run_manifest.json", run_manifest)

    handoff_root.mkdir()
    shutil.copytree(environment_dir, handoff_root / "environment")
    shutil.copy2(run_root / "run_manifest.json", handoff_root)
    shutil.copy2(run_root / "dataset_manifest.json", handoff_root)
    shutil.copy2(run_root / "candidate_comparison.csv", handoff_root)
    shutil.copy2(run_root / "candidate_details.json", handoff_root)
    shutil.copy2(run_root / "selection.json", handoff_root)
    shutil.copy2(run_root / "pipeline_timing.json", handoff_root)

    for detail in candidate_details:
        candidate_name = detail["candidate"]
        evidence_dir = handoff_root / "candidates" / candidate_name
        copy_evidence(Path(detail["train_directory"]), evidence_dir / "train")
        copy_evidence(
            Path(detail["validation_directory"]),
            evidence_dir / "validation",
        )
        weights_dir = evidence_dir / "weights"
        weights_dir.mkdir()
        shutil.copy2(
            Path(detail["best_checkpoint"]),
            weights_dir / "best.pt",
        )

    selected_handoff = handoff_root / "selected"
    shutil.copytree(selected_weights_dir, selected_handoff / "weights")
    copy_evidence(diagnostic_dir, selected_handoff / "diagnostic_test")
    shutil.copytree(predictions_dir, selected_handoff / "predictions")
    shutil.copytree(poc_dir, selected_handoff / "poc")

    handoff_readme = f"""# LabRack Vision QA — RTX 4090 evidence handoff

I produced this package with a fresh six-candidate CUDA run.

- Run ID: `{args.run_id}`
- GPU: `{environment['gpu_name']}`
- Dataset YAML: `{args.data}`
- Selected candidate: `{selected['candidate']}`
- Selected checkpoint: `selected/weights/{selected_weight_name}`
- Selection rule: maximum validation empty-slot recall, then overall mAP50-95,
  then overall mAP50

I started every candidate from the isolated official Ultralytics YOLO11 base
checkpoint recorded in `candidate_details.json`. I used `resume=False` and did
not load any LabRack checkpoint from the earlier Apple MPS experiments.

I use `candidate_comparison.csv`, the per-candidate validation evidence,
`selection.json`, the selected diagnostic-test evidence, predictions, POC
outputs, and timing when I update the final repository deliverables.

Every QA finding remains a possible issue requiring human review. This
educational prototype is not validated for clinical, diagnostic, or production
laboratory use.
"""
    (handoff_root / "HANDOFF_README.md").write_text(
        handoff_readme,
        encoding="utf-8",
    )
    write_checksums(handoff_root)
    shutil.make_archive(
        str(handoff_zip.with_suffix("")),
        "zip",
        root_dir=handoff_root.parent,
        base_dir=handoff_root.name,
    )
    archive_hash = sha256_file(handoff_zip)
    handoff_sha256.write_text(
        f"{archive_hash}  {handoff_zip.name}\n",
        encoding="utf-8",
    )
    (run_root / "COMPLETE.txt").write_text(
        f"Completed: {datetime.now(timezone.utc).isoformat()}\n"
        f"Handoff ZIP: {handoff_zip}\n"
        f"Checksum file: {handoff_sha256}\n"
        f"SHA-256: {archive_hash}\n",
        encoding="utf-8",
    )

    print("\n[complete] Fresh RTX 4090 matrix finished.")
    print(f"  Selected: {selected['candidate']}")
    print(f"  Handoff:  {handoff_zip}")
    print(f"  Checksum: {handoff_sha256}")
    print(f"  SHA-256:  {archive_hash}")
    print("  I will bring this ZIP back for the final documentation update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
