"""Tests for the QA rules — no model required, just hand-written detections."""

from src.qa_rules import count_by_class, find_low_confidence, run_qa_checks


def make_detection(class_name, confidence=0.9, box=None):
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "box": box or [0, 0, 10, 10],
    }


def test_count_by_class_counts_each_name():
    detections = [
        make_detection("tube"),
        make_detection("tube"),
        make_detection("cap"),
        make_detection("rack"),
    ]
    counts = count_by_class(detections)
    assert counts == {"tube": 2, "cap": 1, "rack": 1}


def test_find_low_confidence_uses_threshold():
    detections = [
        make_detection("tube", confidence=0.9),
        make_detection("tube", confidence=0.4),
        make_detection("cap", confidence=0.49),
    ]
    low = find_low_confidence(detections, review_confidence=0.5)
    assert len(low) == 2


def test_no_detections_warns_and_recommends_review():
    result = run_qa_checks([])
    assert result["review_recommended"] is True
    assert any(flag["level"] == "warning" for flag in result["flags"])
    assert result["core_counts"] == {
        "rack": 0, "tube": 0, "cap": 0, "empty_slot": 0}


def test_missing_rack_raises_warning():
    detections = [make_detection("tube"), make_detection("cap")]
    result = run_qa_checks(detections)
    messages = " ".join(flag["message"] for flag in result["flags"])
    assert "No rack detected" in messages


def test_fewer_caps_than_tubes_flags_possible_uncapped():
    detections = [make_detection("rack")]
    detections += [make_detection("tube") for _ in range(5)]
    detections += [make_detection("cap") for _ in range(3)]
    result = run_qa_checks(detections)
    review_flags = [f for f in result["flags"] if f["level"] == "review"]
    assert any("uncapped" in f["message"] for f in review_flags)
    assert result["review_recommended"] is True


def test_caps_equal_tubes_no_uncapped_flag():
    detections = [make_detection("rack")]
    detections += [make_detection("tube") for _ in range(4)]
    detections += [make_detection("cap") for _ in range(4)]
    result = run_qa_checks(detections)
    assert not any("uncapped" in f["message"] for f in result["flags"])


def test_empty_slots_reported_as_possible_issue_for_review():
    detections = [
        make_detection("rack"),
        make_detection("tube"),
        make_detection("cap"),
        make_detection("empty_slot"),
        make_detection("empty_slot"),
    ]
    result = run_qa_checks(detections)
    review_flags = [f for f in result["flags"] if f["level"] == "review"]
    assert any("2 possible empty position" in f["message"]
               for f in review_flags)
    assert result["review_recommended"] is True


def test_clean_rack_recommends_no_review():
    detections = [make_detection("rack")]
    detections += [make_detection("tube") for _ in range(3)]
    detections += [make_detection("cap") for _ in range(3)]
    result = run_qa_checks(detections)
    assert result["review_recommended"] is False
    assert result["flags"] == []
