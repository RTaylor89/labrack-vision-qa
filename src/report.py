"""
report.py — turn the results into the two saved files a person reads.

We produce two things from the same data:
  1. A JSON file — structured and machine-readable, good for later analysis.
  2. A plain-text summary — short and human-readable, good for a quick glance.

Keeping both in one module means the JSON and the summary can never drift apart:
the summary is always built from the same results dict that gets saved as JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def build_results(detection_output, qa_result, inference_seconds=None):
    """
    Merge the detector's output and the QA result into one dictionary.

    This combined dict is the "single record" of everything that happened for
    one image. We add a UTC timestamp so results are self-dating, and optionally
    the inference time (used to check our "under 3 seconds" speed target).
    """
    results = {
        "image_path": detection_output["image_path"],
        "image_size": detection_output["image_size"],
        "model": detection_output["model"],
        # isoformat() gives a standard, sortable timestamp like
        # "2026-07-14T21:00:00+00:00". We use UTC so it is unambiguous.
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": qa_result["counts"],
        "core_counts": qa_result["core_counts"],
        "detections": detection_output["detections"],
        "flags": qa_result["flags"],
        "review_recommended": qa_result["review_recommended"],
        "disclaimer": config.DISCLAIMER,
    }
    # Only include the timing if we were given it (the QA-only tests do not run
    # the model, so they have no timing to report).
    if inference_seconds is not None:
        results["inference_seconds"] = round(inference_seconds, 3)
    return results


def build_summary(results):
    """
    Render the short, human-readable summary text from a results dict.

    We build the text line by line into a list, then join it with newlines at
    the end. Building a list and joining once is cleaner (and faster) than
    repeatedly adding strings together.
    """
    lines = []
    lines.append("LabRack Vision QA — Summary")
    lines.append("=" * 32)  # a row of 32 '=' characters as an underline
    lines.append(f"Image:  {Path(results['image_path']).name}")
    lines.append(f"Model:  {results['model']}")
    if "inference_seconds" in results:
        lines.append(f"Time:   {results['inference_seconds']:.3f} s")
    lines.append("")  # blank line

    # The four core class counts, always shown. The :<12 pads each name to 12
    # characters so the numbers line up in a neat column.
    lines.append("Counts (core classes):")
    for name, count in results["core_counts"].items():
        lines.append(f"  {name:<12} {count}")

    # If the model reported any OTHER classes (common with the pretrained model),
    # list them separately so it is obvious they are not our project classes.
    other = {k: v for k, v in results["counts"].items()
             if k not in results["core_counts"]}
    if other:
        lines.append("")
        lines.append("Other detected classes:")
        for name, count in sorted(other.items()):
            lines.append(f"  {name:<12} {count}")

    # The QA flags, or a clear "none" if there were none.
    lines.append("")
    if results["flags"]:
        lines.append("QA flags:")
        for flag in results["flags"]:
            lines.append(f"  [{flag['level'].upper()}] {flag['message']}")
    else:
        lines.append("QA flags: none.")

    # A one-line verdict. Even when nothing is flagged we still remind the reader
    # that a human must review — this tool never signs off on its own.
    lines.append("")
    verdict = ("Manual review recommended."
               if results["review_recommended"]
               else "No issues flagged. Human review still required.")
    lines.append(verdict)
    lines.append("")
    lines.append(results["disclaimer"])

    return "\n".join(lines)


def save_results(results, json_path, summary_path):
    """
    Write both output files to disk and return the summary text.

    We create the output folders if needed, dump the results dict as pretty
    (indented) JSON, and write the rendered summary. We return the summary text
    so the caller (run.py) can also print it to the screen without rebuilding it.
    """
    json_path = Path(json_path)
    summary_path = Path(summary_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # encoding="utf-8" makes sure characters like the "—" dash save correctly.
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    summary_text = build_summary(results)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text + "\n")

    return summary_text
