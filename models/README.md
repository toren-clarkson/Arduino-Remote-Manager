# Model weights

Existing weights are retained. `red-ball-v1.pt` is used by default in
[stop-and-stare.py](../stop-and-stare.py), targeting its `Red-Ball` class.
`yolo26n.pt` is the pretrained COCO model for classes such as bottle, cup and person.

Pass another local model with `--weights models/my_object.pt --target-class my_object`
to the stop-and-stare mission, or `--model` to `default-yolo-tracking.py`.
Class names must match the model. Dataset YAML/ZIP files are not model weights.
See [training](../training/README.md) for optional training commands.
