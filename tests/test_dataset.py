"""Structural checks for the staged YOLO detection dataset."""

from pathlib import Path


DATA_ROOT = Path(__file__).parents[1] / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
CLASS_IDS = {0, 1, 2, 3}


def test_every_image_has_one_matching_label():
    for split in SPLITS:
        images = {
            path.stem
            for path in (DATA_ROOT / "images" / split).iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {
            path.stem
            for path in (DATA_ROOT / "labels" / split).glob("*.txt")
        }
        assert images == labels


def test_detection_labels_are_normalized_and_unique():
    for split in SPLITS:
        for label_path in (DATA_ROOT / "labels" / split).glob("*.txt"):
            rows = label_path.read_text(encoding="utf-8").splitlines()
            assert len(rows) == len(set(rows)), f"Duplicate row in {label_path}"

            for line_number, row in enumerate(rows, 1):
                fields = row.split()
                assert len(fields) == 5, (
                    f"{label_path}:{line_number} must contain five fields"
                )
                class_id = int(fields[0])
                cx, cy, width, height = map(float, fields[1:])
                assert class_id in CLASS_IDS
                assert 0 <= cx <= 1
                assert 0 <= cy <= 1
                assert 0 < width <= 1
                assert 0 < height <= 1
                assert cx - width / 2 >= -1e-6
                assert cx + width / 2 <= 1 + 1e-6
                assert cy - height / 2 >= -1e-6
                assert cy + height / 2 <= 1 + 1e-6
