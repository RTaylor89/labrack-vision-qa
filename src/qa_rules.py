"""
qa_rules.py — The "thinking" part of the project.

The detector tells me WHAT is in the image. This file decides WHAT THAT MEANS
for quality assurance: are there fewer caps than tubes? Any empty slots? Any
detections I am unsure about?

Two important principles for this file:
  1. It never loads a model. It only works on the list of detection dicts. That
     means I can test it with made-up detections (see tests/test_qa_rules.py),
     which is fast and reliable.
  2. It always speaks in careful language — "possible", "requires review". This
     is a prototype that flags things for my review, not automated approval. It
     must never claim it has
     "confirmed a defect."

Flag levels I use:
    "warning" — the image or capture itself looks wrong (e.g. no rack found)
    "review"  — a possible issue a person should look at
    "info"    — a neutral fact worth recording (e.g. number of empty slots)
"""

from collections import Counter  # Counter is a dict that counts things for us

from . import config


def count_by_class(detections):
    """
    Count how many detections there are of each class name.

    Counter does the heavy lifting: give it the class name of every detection
    and it returns something like Counter({"tube": 22, "cap": 21}). I convert
    it to a normal dict before returning so the output is simple and predictable.
    """
    counts = Counter(det["class_name"] for det in detections)
    return dict(counts)


def find_low_confidence(detections, review_confidence=config.REVIEW_CONFIDENCE):
    """
    Return only the detections whose confidence is below the review cutoff.

    These already passed the detector's own threshold (so they are shown), but
    they are not confident enough to fully trust, so I collect them here to be
    flagged for my review.
    """
    return [det for det in detections if det["confidence"] < review_confidence]


def run_qa_checks(detections,
                  core_classes=config.CORE_CLASSES,
                  review_confidence=config.REVIEW_CONFIDENCE):
    """
    Run every QA check and return a single summary dictionary.

    Returned shape:
        {
            "counts":       {class_name: n, ...},   # every class I saw
            "core_counts":  {rack, tube, cap, empty_slot},  # always all four
            "flags":        [{"level": ..., "message": ...}, ...],
            "review_recommended": True/False,
        }

    I build "core_counts" separately from "counts" so downstream code can
    always rely on the four project classes being present (as 0 if not seen),
    even when the model reports other, unrelated classes.
    """
    counts = count_by_class(detections)
    # For each core class, look up its count, defaulting to 0 if it was not seen.
    core_counts = {name: counts.get(name, 0) for name in core_classes}
    flags = []  # I append findings here as I go

    # --- Special case: the model found nothing at all -----------------------
    # If there are zero detections, further checks are pointless. I warn and
    # return early so the rest of the function can assume detections exist.
    if not detections:
        flags.append({
            "level": "warning",
            "message": "No objects detected. Verify the image and the model.",
        })
        return {
            "counts": counts,
            "core_counts": core_counts,
            "flags": flags,
            "review_recommended": True,
        }

    # --- Check 1: is there a rack at all? -----------------------------------
    if core_counts.get("rack", 0) == 0:
        flags.append({
            "level": "warning",
            "message": "No rack detected. Verify the image shows a rack and "
                       "that a rack-aware model is loaded.",
        })

    # --- Check 2: fewer caps than tubes => possible uncapped tubes ----------
    tube_count = core_counts.get("tube", 0)
    cap_count = core_counts.get("cap", 0)
    # This is meaningful only if I actually saw tubes. I use careful wording.
    if tube_count > 0 and cap_count < tube_count:
        missing = tube_count - cap_count
        flags.append({
            "level": "review",
            "message": f"Fewer caps ({cap_count}) than tubes ({tube_count}) — "
                       f"{missing} possible uncapped tube(s), requires review.",
        })

    # --- Check 3: possible empty positions => request human review ----------
    empty_count = core_counts.get("empty_slot", 0)
    if empty_count > 0:
        flags.append({
            "level": "review",
            "message": f"{empty_count} possible empty position(s) detected — "
                       "requires human review.",
        })

    # --- Check 4: shaky detections => ask a human to look -------------------
    low_conf = find_low_confidence(detections, review_confidence)
    if low_conf:
        flags.append({
            "level": "review",
            "message": f"{len(low_conf)} detection(s) below the "
                       f"{review_confidence:.2f} confidence threshold — "
                       f"manual review recommended.",
        })

    # I recommend review if ANY flag is a warning or review (info is neutral).
    # any() returns True as soon as it finds one matching item.
    review_recommended = any(flag["level"] in ("warning", "review")
                             for flag in flags)

    return {
        "counts": counts,
        "core_counts": core_counts,
        "flags": flags,
        "review_recommended": review_recommended,
    }
