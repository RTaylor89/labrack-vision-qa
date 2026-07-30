"""Tests for my RTX 4090 orchestration logic; no model or GPU is required."""

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_rtx4090_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("run_rtx4090_matrix", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_matrix_contains_every_requested_model_and_resolution():
    actual = {
        (candidate.model, candidate.imgsz)
        for candidate in MODULE.CANDIDATES
    }
    expected = {
        (f"yolo11{size}.pt", imgsz)
        for size in ("n", "s", "m")
        for imgsz in (640, 960)
    }
    assert actual == expected
    assert len(MODULE.CANDIDATES) == 6


def test_selection_prioritizes_empty_slot_recall():
    rows = []
    for index, candidate in enumerate(MODULE.CANDIDATES):
        rows.append(
            {
                "candidate": candidate.name,
                "empty_slot_recall": 0.50 + index / 100,
                "validation_map50_95": 0.90 - index / 100,
                "validation_map50": 0.95 - index / 100,
            }
        )

    selected = MODULE.select_candidate(rows)

    assert selected["candidate"] == MODULE.CANDIDATES[-1].name


def test_selection_uses_declared_tie_breakers():
    rows = []
    for candidate in MODULE.CANDIDATES:
        rows.append(
            {
                "candidate": candidate.name,
                "empty_slot_recall": 0.80,
                "validation_map50_95": 0.60,
                "validation_map50": 0.70,
            }
        )
    rows[2]["validation_map50_95"] = 0.65
    rows[3]["validation_map50_95"] = 0.65
    rows[3]["validation_map50"] = 0.75

    selected = MODULE.select_candidate(rows)

    assert selected["candidate"] == rows[3]["candidate"]


def test_selection_rejects_an_incomplete_matrix():
    with pytest.raises(ValueError, match="all 6 completed candidates"):
        MODULE.select_candidate(
            [{
                "candidate": "yolo11n_640",
                "empty_slot_recall": 0.5,
                "validation_map50_95": 0.5,
                "validation_map50": 0.5,
            }]
        )


@pytest.mark.parametrize(
    "run_id",
    ["rtx4090_final", "rtx4090-20260815", "rtx4090.2026_08_15"],
)
def test_run_id_accepts_portable_names(run_id):
    assert MODULE.validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    ["", "../escape", "contains spaces", "/absolute/path"],
)
def test_run_id_rejects_unsafe_names(run_id):
    with pytest.raises(ValueError):
        MODULE.validate_run_id(run_id)
