# Local datasets

Existing images, videos, YOLO labels and Roboflow exports are preserved here.
Use VS Code or File Explorer to copy and edit files. Dataset contents are ignored
by Git; instructions remain tracked.

Suggested labeled structure: `my_object/data.yaml`, `train/images`, `train/labels`,
`valid/images`, `valid/labels`, and optional `test/` splits. See
[training/README.md](../training/README.md) for frame extraction, annotation, export
and training. `data.yaml` describes a dataset, not a trained detector.
