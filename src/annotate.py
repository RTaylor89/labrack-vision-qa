"""
annotate.py — This is where we draw the detection boxes onto the image and save it.

This is the show our work step: it turns the list of detections into a
picture a person can look at, with a colored rectangle and a label on each
detected object.

We use OpenCV (imported as cv2), a very common computer-vision library. Two
OpenCV quirks to keep in mind:
  - Colors are ordered (Blue, Green, Red), not (Red, Green, Blue).
  - cv2.imread returns None instead of raising when it cannot read a file, so we
    have to check for None ourselves.
"""

from pathlib import Path

import cv2

from . import config


def _label_for(detection):
    """Build the short text drawn above a box, e.g. 'tube 0.87'."""
    return f"{detection['class_name']} {detection['confidence']:.2f}"


def annotate_image(image_path, detections, save_path):
    """
    Draw every detection on the image and write the result to save_path.

    Returns the save_path (as a Path). Raises a clear error if the image cannot
    be read — this can happen if a file has a valid ".jpg" name but is actually
    corrupt, which our earlier extension check cannot catch.
    """
    # Load the original image from disk. cv2 gives back a NumPy array of pixels,
    # or None if it failed.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Could not read image data from {image_path}. "
            f"The file may be corrupt or not a real image."
        )

    # Draw one rectangle + label per detection.
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        # Pick the class color, or fall back to grey for unknown classes.
        color = config.CLASS_COLORS.get(detection["class_name"],
                                        config.DEFAULT_COLOR)

        # The bounding box itself. thickness=2 draws a 2-pixel-wide outline.
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # Now the label. First measure how big the text will be, so we can draw
        # a filled colored strip behind it — otherwise white text can vanish
        # against light parts of the photo.
        label = _label_for(detection)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Put the strip just above the box. max(..., 0) stops it going off the
        # top edge of the image when a box is near the top.
        top = max(y1 - text_height - baseline - 4, 0)
        cv2.rectangle(image, (x1, top), (x1 + text_width + 4, y1), color, -1)

        # Finally the text, in white, on top of the strip. LINE_AA = smoother
        # (anti-aliased) text.
        cv2.putText(image, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Make sure the output folder exists, then save. parents=True creates any
    # missing parent folders; exist_ok=True means "don't error if it's there."
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # cv2.imwrite returns False on failure instead of raising, so we check it.
    if not cv2.imwrite(str(save_path), image):
        raise IOError(f"Failed to write annotated image to {save_path}")
    return save_path
