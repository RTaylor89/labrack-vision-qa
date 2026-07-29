"""Tests for training dataset preflight logic — no model required."""

from pathlib import Path

import pytest
import yaml

from smoke_train import preflight_dataset


NAMES = {
    0: "rack",
    1: "tube",
    2: "cap",
    3: "empty_slot",
}


def write_dataset_yaml(root):
    data_dir = root / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump({
            "path": "data",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": NAMES,
        }),
        encoding="utf-8",
    )
    return dataset_path


def test_preflight_uses_ultralytics_root_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_dataset_yaml(tmp_path)
    for split in ("train", "val"):
        image_dir = tmp_path / "data" / "images" / split
        image_dir.mkdir(parents=True)
        (image_dir / f"{split}.jpg").write_bytes(b"test")

    preflight_dataset(Path("data/dataset.yaml"))


def test_preflight_rejects_empty_validation_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset_path = write_dataset_yaml(tmp_path)
    train_dir = tmp_path / "data" / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "train.jpg").write_bytes(b"test")

    with pytest.raises(SystemExit):
        preflight_dataset(dataset_path)
