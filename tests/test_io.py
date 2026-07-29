"""Tests for input validation and output-path logic — no model required."""

import pytest

from src import config
from src.detector import RackDetector, validate_image_path


def test_output_paths_derive_from_stem():
    paths = config.output_paths("input/rack_001.jpg", output_dir="output")
    assert paths["annotated"].name == "rack_001_annotated.jpg"
    assert paths["json"].name == "rack_001_results.json"
    assert paths["summary"].name == "rack_001_summary.txt"
    assert str(paths["annotated"].parent) == "output"


def test_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError):
        validate_image_path(missing)


def test_unsupported_extension_raises_value_error(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        validate_image_path(bad)


def test_supported_extension_passes(tmp_path):
    good = tmp_path / "rack_001.PNG"  # case-insensitive
    good.write_bytes(b"\x89PNG\r\n\x1a\n")
    resolved = validate_image_path(good)
    assert resolved == good


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        validate_image_path(tmp_path)


def test_detector_passes_selected_image_size_to_yolo(tmp_path, monkeypatch):
    image_path = tmp_path / "rack.jpg"
    image_path.write_bytes(b"not decoded by the fake model")

    class FakeResult:
        boxes = None
        names = {}
        orig_shape = (480, 640)

    class FakeModel:
        def __init__(self):
            self.predict_kwargs = None

        def predict(self, **kwargs):
            self.predict_kwargs = kwargs
            return [FakeResult()]

    fake_model = FakeModel()
    monkeypatch.setattr("src.detector._load_model", lambda _: fake_model)

    detector = RackDetector(model_path="fake.pt", image_size=960)
    output = detector.detect(image_path)

    assert fake_model.predict_kwargs["imgsz"] == 960
    assert output["inference_image_size"] == 960
