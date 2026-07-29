"""
detector.py — The ONLY file that "talks" to the YOLO model.

Design idea:
    We deliberately keep "running the model" separate from deciding what the
    results mean. That second part will live in qa_rules.py. This separation has
    two big payoffs:
      1. We can test the QA logic without ever loading a heavy model.
      2. If we swap YOLO for something else later, only this file changes.

What this file produces:
    A "detection" is just a plain Python dictionary. We use a dict (instead of a
    fancy class) because it is easy to read, easy to save as JSON, and easy to
    build by hand in a test. Each detection looks like:

        {
            "class_id": 1,                 # the model's numeric label
            "class_name": "tube",          # the human-readable name
            "confidence": 0.87,            # how sure the model is (0.0 - 1.0)
            "box": [x1, y1, x2, y2],       # corner pixels of the box
        }
"""

from pathlib import Path

from . import config  # our own settings module


def validate_image_path(image_path):
    """
    Make sure the file is one we can actually process, BEFORE we load a model.

    We check three things in order and raise a clear, specific error for each:
      - the file exists,
      - it is a file (not a folder),
      - its extension is one we support.

    Returning the resolved Path on success lets callers reuse it. This function
    is intentionally light: it imports nothing heavy, so tests can call it
    without needing PyTorch/ultralytics installed.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    # .suffix is the extension (".JPG"); .lower() makes the check case-insensitive
    extension = path.suffix.lower()
    if extension not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(config.SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type '{extension}' for {path.name}. "
            f"Supported types: {supported}"
        )

    return path


def _load_model(model_path):
    """
    Load a YOLO model from a weights file.

    The import sits INSIDE the function, not at the top of the file. This
    is on purpose: importing ultralytics also loads PyTorch, which is large and
    slow. By importing it only when we truly load a model, the rest of the code
    (and all our QA tests) can run without that heavy dependency installed.

    The leading underscore in the name (_load_model) is a Python convention
    meaning: internal helper, not meant to be used from outside this file.
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:
        # Re-raise with a friendlier message that tells the user how to fix it.
        raise ImportError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from error
    return YOLO(model_path)


class RackDetector:
    """
    A small wrapper around one YOLO11 model.

    Why a class instead of a plain function? Because loading the model is slow,
    and we only want to do it once. We load it in __init__ (when the object is
    created) and then reuse that loaded model every time detect() is called.
    """

    def __init__(self, model_path=config.MODEL_PATH,
                 confidence_threshold=config.CONFIDENCE_THRESHOLD,
                 image_size=config.INFERENCE_IMAGE_SIZE):
        # Remember the settings so we can record them later in the report.
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        # Load the model once, up front.
        self.model = _load_model(model_path)

    def detect(self, image_path):
        """
        Run the model on one image and return a tidy result dictionary.

        Steps:
          1. Validate the path (fail early with a clear error if it is bad).
          2. Ask the model to predict, passing our confidence threshold.
          3. Walk through each detected box and copy the useful bits into our
             own plain-dict format.
          4. Return everything the rest of the pipeline needs.
        """
        path = validate_image_path(image_path)

        # predict() returns a list (one entry per image). We gave it one image,
        # so we take element [0]. verbose=False keeps it from printing to screen.
        results = self.model.predict(
            source=str(path),
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        result = results[0]
        '''
        Gemini helped here. I was struggling to understand that the model's output is
        a tensor(boxes[box]). So I had Gemini help me troubleshoot and explain this portion via comments
        and then rewrote the comments to make sure I understood.
        '''
        detections = []
        boxes = result.boxes  # can be None if the model found nothing
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls)
                # box.xyxy[0] is a tensor of [x1, y1, x2, y2] as floats (The box corners). 
                # Here you convert each to a whole-number pixel coordinate.
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist()) #.tolist() makes a list of floats
                # detections.append builds 1 detection dict per box
                detections.append({
                    "class_id": class_id,
                    # result.names maps a class_id number to its text label.
                    "class_name": result.names[class_id],
                    "confidence": round(float(box.conf), 4),
                    "box": [x1, y1, x2, y2],
                })

        # ultralytics reports the image shape as (height, width). I flip it to
        # (width, height) because that is the order that makes the most sense to me (Roderick).
        height, width = result.orig_shape
        return {
            "image_path": str(path),
            "image_size": [width, height],
            "model": str(self.model_path),
            "inference_image_size": self.image_size,
            "detections": detections,
        }
