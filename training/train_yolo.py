"""Train local YOLO detection weights from an exported, labeled Roboflow data.yaml."""

import argparse
import shutil
from pathlib import Path

from robotics_lab.paths import MODELS, RUNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=MODELS/"yolo26n.pt")
    parser.add_argument("--output", type=Path, default=MODELS/"custom_object.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Both --data data.yaml and --weights .pt must be existing files")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output filename")
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("epochs, imgsz, and batch must be positive")
    import polars  # checkpoint/metrics dependency: fail before spending time training
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    if model.task != "detect":
        parser.error("Use object-detection weights")
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                project=str(RUNS/"training"), name="custom_detector", exist_ok=False)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training finished without a best.pt file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print("Model saved:", args.output.resolve())
    print("Classes:", model.names)


if __name__ == "__main__":
    main()
