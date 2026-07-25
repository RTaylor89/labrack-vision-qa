"""
config.py — Central place for every setting the project uses.

Why do we have a config file at all?
    As the project grows, the same values (which model, what confidence cutoff,
    what the classes are called) get used in several files. If those values were
    copy-pasted around, changing one would mean hunting through the whole
    codebase. Instead we put them here once and every other module imports them.
    This is a common enterprise pattern: "single source of truth."

Nothing in here does real work — it only holds values and one small helper.
"""

from pathlib import Path  # pathlib gives us nice, OS-safe file paths


# --- Model -----------------------------------------------------------------
# Which model weights to load. We start with the pretrained "nano" YOLO11 model
# (yolo11n.pt) because it is small, fast, and downloads automatically the first
# time it is used. It is only a placeholder: its built-in classes are everyday
# COCO objects (person, bottle, etc.), NOT lab racks and tubes.
#
# Later, after we train our own model, we just change this ONE line to point at
# our trained weights, for example:  MODEL_PATH = "weights/labrack_yolo11n.pt"
MODEL_PATH = "yolo11n.pt"


# --- Detection thresholds --------------------------------------------------
# A detector gives every box a "confidence" score between 0.0 and 1.0.
#
# CONFIDENCE_THRESHOLD: boxes scoring below this are thrown away completely —
#   we assume they are noise. 0.25 is a common, forgiving starting point.
CONFIDENCE_THRESHOLD = 0.25

# REVIEW_CONFIDENCE: boxes that DO pass the threshold but score below this are
#   still kept, but the QA layer flags them so a human double-checks them. Think
#   of it as "good enough to show, not good enough to fully trust."
REVIEW_CONFIDENCE = 0.50


# --- Classes ---------------------------------------------------------------
# The object types this project actually cares about. We keep them in a tuple
# (an unchangeable list) because these names are fixed labels, not data that
# should be edited at runtime.
#
# Important teaching point: the pretrained yolo11n model will never output these
# names, so their counts will be zero until we use a trained model. The QA layer
# is written to handle that honestly instead of pretending the objects exist.
CORE_CLASSES = ("rack", "tube", "cap", "empty_slot")


# --- Input validation ------------------------------------------------------
# The only image types we accept. We check a file's extension against this list
# before trying to open it, so we can give a clear error instead of crashing.
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --- Annotation colors -----------------------------------------------------
# Colors used when we draw boxes on the image. NOTE: OpenCV orders colors as
# (Blue, Green, Red), NOT the usual (Red, Green, Blue). This trips up almost
# everyone the first time, so we spell it out here.
CLASS_COLORS = {
    "rack": (180, 120, 40),        # blue-teal
    "tube": (60, 180, 75),         # green
    "cap": (0, 165, 255),          # orange
    "empty_slot": (60, 60, 220),   # red
}
# If we ever detect a class that is not in the dict above (e.g. a COCO class
# from the pretrained model), we fall back to grey instead of crashing.
DEFAULT_COLOR = (200, 200, 200)


# --- Output ----------------------------------------------------------------
# Where results are written by default. Path() makes this work on any OS.
DEFAULT_OUTPUT_DIR = Path("output")

# This disclaimer is attached to every report. The project is an educational
# prototype, so we say so clearly and consistently, everywhere.
DISCLAIMER = (
    "Educational prototype only. Not validated for clinical, diagnostic, or "
    "production laboratory use. Human review is required."
)


def output_paths(image_path, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Work out the three output filenames for a given input image.

    We keep this here (in config) so the naming rule lives in one place. If we
    ever want to rename outputs, we change it once and every caller follows.

    Example:
        input/rack_001.jpg  ->  output/rack_001_annotated.jpg
                                output/rack_001_results.json
                                output/rack_001_summary.txt

    Path(image_path).stem grabs just "rack_001" (no folder, no extension).
    We then build three sibling filenames inside output_dir and return them in
    a dict so callers can ask for them by name (paths["json"], etc.).
    """
    stem = Path(image_path).stem
    output_dir = Path(output_dir)
    return {
        "annotated": output_dir / f"{stem}_annotated.jpg",
        "json": output_dir / f"{stem}_results.json",
        "summary": output_dir / f"{stem}_summary.txt",
    }
