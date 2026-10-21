# Local datasets

Use the [local webapp](../webapp/README.md) Datasets page to upload media,
sample video frames, import labeled ZIPs or download a Roboflow version.
Browse/edit images, labels and YAML through Files & editor.

Keep unlabelled images, extracted video frames, and exported Roboflow versions here. Dataset contents are ignored by Git; source code and instructions remain tracked.

Suggested layout:

```text
unlabeled/session_a/*.jpg
my_object/data.yaml
my_object/train/images/ and labels/
my_object/valid/images/ and labels/
my_object/test/images/ and labels/
```

See [training/README.md](../training/README.md) for labeling/export/training instructions. `data.yaml` describes a dataset; it is not a trained model and cannot be passed to the robot's `--weights` option.
