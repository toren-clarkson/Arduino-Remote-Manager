# Roboflow dataset -> trained YOLO -> robot

The [local webapp](../webapp/README.md) provides forms for uploads, frame extraction,
Roboflow/ZIP import, model inspection, training and logs. Its training executor
also supports segmentation, pose and OBB with matching task-specific weights and
labels; the command-line helper below remains focused on detection.

On this Windows ARM64 host running x64 Python, Polars 2.0 fails with
`unknown feature flag: 'sse3'` when saving training results. The project pins
`polars[rtcompat]==1.35.2` on this platform, verified with a real one-epoch training
run. Use the editable install to apply that constraint; do not bypass CPU checks.

## What needs training?

The **YOLO detector** learns the object's visual class from labeled images. A video is optional: it is a convenient source of images, but the detector still needs bounding-box labels. **ByteTrack** associates the detector's boxes between frames; this project does not separately train ByteTrack for each object. The AprilTag detector decodes a known printed family/ID and needs no example training video.

If the object is already in COCO (for example `bottle`, `cup`, or `sports ball`), try `models/yolo26n.pt` first and skip custom training. Use a custom detector when the intended object is not recognized adequately by those classes.

## 1. Collect images

Take photos of the object from the distances, viewing angles, backgrounds and lighting the robot will encounter. Include examples with only part of the object visible, plus representative empty scenes. Keep the object class name consistent. The stop-and-stare program uses sharp stationary observations, so include images representative of that setup.

If collecting video, sample images locally:

```powershell
.\.venv\Scripts\python.exe -m training.extract_frames path\to\session_a.mp4 --output datasets/unlabeled/session_a --fps 2
```

`extract_frames.py` reads video frame-rate metadata and saves JPEGs at the requested sample rate. It does not annotate or train anything and does not overwrite existing output filenames. Use another output directory for each session. You can also use images saved by the mission's `--save-frames` option; those are unannotated stationary frames, not labeled training data.

Avoid putting nearly identical neighboring frames from the same recording into both training and validation/test sets. Prefer splitting entire capture sessions before sampling so validation reflects new scenes rather than duplicates.

## 2. Label and version with Roboflow

1. Create a Roboflow **Object Detection** project.
2. Upload your photos or extracted JPEGs.
3. Draw a tight bounding box around every visible instance of the chosen object and assign a consistent name such as `my_object`. Review annotations for consistency and retain relevant negative images.
4. Create training, validation and test splits with separate scene/session coverage.
5. Generate a dataset version, then export an Ultralytics-compatible bounding-box format, such as **YOLOv8**. That dataset format can be used with YOLO26 training; the format name does not force a YOLOv8 model.
6. Download and unzip the export to `datasets/my_object/`, so `datasets/my_object/data.yaml` exists along with its images and labels.

Roboflow handles annotation and dataset export in this workflow. Runtime detection stays local; no account key is needed on the moving robot program. See the official [dataset export documentation](https://docs.roboflow.com/datasets/exporting-data).

### Optional scripted download

`download_dataset.py` calls Roboflow's documented dataset-export HTTP endpoint; it does not require installing the Roboflow SDK into this camera environment. Set the account key locally in PowerShell, then run:

```powershell
$env:ROBOFLOW_API_KEY = "your-private-key"
.\.venv\Scripts\python.exe -m training.download_dataset --workspace YOUR_WORKSPACE --project YOUR_PROJECT --version 1 --output datasets/my_object
```

Replace those placeholders with your own export details. Generate the version/export in the web app first. The utility downloads YOLO-format data to a **new** directory and validates archive paths before extraction. It does not upload images, create annotations, or train the model. The key is read from the environment and not written to source files or printed in request errors. No private export was downloaded during project setup because no account details/key were provided.

## 3. Train YOLO

From the project root:

```powershell
.\.venv\Scripts\python.exe -m training.train_yolo --data datasets/my_object/data.yaml --weights models/yolo26n.pt --epochs 50 --imgsz 416 --batch 8 --device cpu --output models/my_object.pt
```

The helper fine-tunes the pretrained detection model using Ultralytics, stores outputs under `runs/training/custom_detector*/`, and copies the best weights to the requested model filename. It uses a Windows-safe main guard and zero data-loader workers. Existing destination model files are not overwritten; choose another filename for another experiment.

CPU training is supported but can be slow. `--device 0` selects a compatible CUDA GPU **only if** your PyTorch installation supports it. Alternatively run equivalent Ultralytics training in a GPU notebook, then download `best.pt` into `models/`. A trained `.pt` file is different from the dataset's `data.yaml`.

The defaults are starting values, not a guarantee of accuracy. Inspect validation images, missed objects, false positives, and class names before driving the robot. Keep a separate held-out test set. Ultralytics' official [training guide](https://docs.ultralytics.com/modes/train/) explains training arguments and output metrics.

## 4. Preview your model with the mission

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --weights models/my_object.pt --target-class my_object --tag-id 0 --save-frames
```

The program prints model classes and rejects a requested class that is not present. It starts in preview mode with no Arduino connection. Observe the confirmed boxes, area percentages and planned actions, then calibrate the proximity threshold and discrete steps. To enable the actual robot after checking configuration, add `--drive --arduino-ip YOUR_BOARD_IP`.

YOLO + ByteTrack is called only on stationary observation bursts. Track IDs are reset after a camera movement; selecting the same class afterward cannot guarantee identity between visually identical objects. Training a distinctive class or keeping one target instance in view helps make the task unambiguous. See [the full mission guide](../missions/stop_and_stare/README.md).

## Implemented helpers

| File | Inputs | Outputs |
|---|---|---|
| `extract_frames.py` | Video, output folder, sampling FPS | Unlabeled JPEGs |
| `download_dataset.py` | Workspace/project/version, local API-key environment variable | Validated dataset export in a new folder |
| `train_yolo.py` | Labeled `data.yaml`, initial weights, training settings | Training results and a new `.pt` model |

No custom model has been trained automatically: custom training requires your labeled dataset and an explicit training run. The included pretrained model already works for a demo class such as `bottle`.

Tracking reference: [Ultralytics Track mode and ByteTrack](https://docs.ultralytics.com/modes/track/).
