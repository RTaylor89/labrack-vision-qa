"""
run.py — the front door of the whole program.

This is the file I run from the terminal. It wires the other modules
together in order: validate -> detect -> apply QA rules -> annotate -> report.

Run it like this:
    python -m src.run --image input/rack_001.jpg
    python -m src.run --image input/rack_001.jpg --output-dir output --conf 0.3
    python -m src.run --image input/rack_001.jpg --imgsz 960

For input/rack_001.jpg it writes three files:
    output/rack_001_annotated.jpg
    output/rack_001_results.json
    output/rack_001_summary.txt
"""

import argparse   # standard library tool for reading command-line options
import sys        # lets us read command-line args and set the exit code
import time       # used to measure how long inference takes

from . import config
from .annotate import annotate_image
from .detector import RackDetector, validate_image_path
from .qa_rules import run_qa_checks
from .report import build_results, save_results


def process_image(image_path, output_dir=config.DEFAULT_OUTPUT_DIR,
                  model_path=config.MODEL_PATH,
                  confidence_threshold=config.CONFIDENCE_THRESHOLD,
                  image_size=config.INFERENCE_IMAGE_SIZE):
    """
    Run the complete pipeline on ONE image and write all three output files.

    This function is kept separate from the command-line handling (main) so it
    can be called directly from a notebook or another script, or tested, without
    going through argparse.

    Returns a tuple: (results dict, summary text, output paths dict).
    """
    # Fail early if the input is bad, before loading the (slow) model.
    validate_image_path(image_path)
    paths = config.output_paths(image_path, output_dir)

    # Create the detector. This is where the selected local checkpoint loads.
    detector = RackDetector(model_path=model_path,
                            confidence_threshold=confidence_threshold,
                            image_size=image_size)

    # Time just the detection step. perf_counter is a high-resolution timer.
    start = time.perf_counter()
    detection_output = detector.detect(image_path)
    inference_seconds = time.perf_counter() - start

    # Apply the QA rules to the detections, then assemble the full results.
    qa_result = run_qa_checks(detection_output["detections"])
    results = build_results(detection_output, qa_result, inference_seconds)

    # Save the annotated picture and the two text/JSON reports.
    annotate_image(image_path, detection_output["detections"], paths["annotated"])
    summary_text = save_results(results, paths["json"], paths["summary"])

    return results, summary_text, paths


def _parse_args(argv):
    """
    Define and read the command-line options.

    argparse handles --help for us automatically and reports friendly errors if
    someone passes a bad option. Each add_argument line below is one option.
    """
    parser = argparse.ArgumentParser(
        description="LabRack Vision QA — analyze one staged rack image.")
    parser.add_argument("--image", required=True,
                        help="Path to a JPG or PNG image.")
    parser.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR),
                        help="Directory for annotated image, JSON, and summary.")
    parser.add_argument("--model", default=config.MODEL_PATH,
                        help="Model weights (default: selected project model).")
    parser.add_argument("--conf", type=float,
                        default=config.CONFIDENCE_THRESHOLD,
                        help="Detection confidence threshold (0-1).")
    parser.add_argument("--imgsz", type=int,
                        default=config.INFERENCE_IMAGE_SIZE,
                        help="YOLO inference image size in pixels.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    The entry point. Returns an exit code: 0 means success, 1 means a handled
    error (the shell can use this to know whether the run worked).
    """
    # If argv is None I read the real command-line arguments. Passing argv in
    # explicitly is handy for testing.
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    try:
        results, summary_text, paths = process_image(
            args.image,
            output_dir=args.output_dir,
            model_path=args.model,
            confidence_threshold=args.conf,
            image_size=args.imgsz,
        )
    except (FileNotFoundError, ValueError) as error:
        # These are the "expected" problems: a missing path, or an unsupported
        # or corrupt file. I print a clean message to stderr and exit with 1
        # instead of dumping a scary stack trace. Any OTHER kind of error is a
        # real bug, so I let it crash loudly rather than hide it.
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Show the summary and where everything was saved.
    print(summary_text)
    print()
    print(f"Wrote: {paths['annotated']}")
    print(f"Wrote: {paths['json']}")
    print(f"Wrote: {paths['summary']}")
    return 0


# This block runs only when the file is executed directly (python -m src.run),
# not when it is imported. SystemExit passes my exit code back to the shell.
if __name__ == "__main__":
    raise SystemExit(main())
