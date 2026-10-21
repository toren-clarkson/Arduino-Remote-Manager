# Model weights

The [local webapp](../webapp/README.md) Model library supports uploads,
display names, notes, provider selection, task/class inspection, file rename,
download and recoverable Trash. Its inference/training forms declare task
compatibility; cataloging an arbitrary artifact does not make it YOLO-compatible.

`yolo26n.pt` is the existing pretrained COCO detection model, moved here from the project root. It supports common classes including `bottle`, `cup`, `person`, and `sports ball`.

Place trained custom detection weights here as `custom_object.pt`, `best.pt`, or another descriptive name, then pass that path with `--weights` to the main mission. Large weight files are ignored by Git. No custom object model was trained during project cleanup because no labeled custom dataset was supplied.

Train with the helper in [training/README.md](../training/README.md). Runtime class names must match the trained model's names exactly, ignoring case. A Roboflow URL, dataset ZIP or `data.yaml` is not a weights file.
