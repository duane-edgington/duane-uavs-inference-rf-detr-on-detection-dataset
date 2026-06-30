# duane-uavs-inference-rf-detr-on-detection-dataset

ipynb files that are uploaded to this repo have been stripped of output to save space

```
jupyter nbconvert --clear-output --inplace rf_detr_onnx_sahi_inference.ipynb
```

the repo on ICEFISH where staging is done is

```
duane@ICEFISH duane-uavs-inference-rf-detr-on-detection-dataset
```

## `rf_detr_onnx_sahi_sweep_v4.ipynb`

RF-DETR ONNX + SAHI inference on 42 MP drone images, with a confidence
threshold sweep computed from a single inference pass. This version
evaluates Sarah's RFDETRNano checkpoint (trained under `rfdetr==1.7.1`)
and removes the diagnostic cells used during pipeline debugging.

Run order: cells 1–2, then **Runtime → Restart session**, then cells 4
onward in order.

### Cell 1 — Check GPU
`!nvidia-smi`. Confirms a GPU is attached before any installs run.

### Cell 2 — Install dependencies
Installs `rfdetr`, `supervision`, `roboflow`, `sahi[roboflow]`, the
`rfdetr[onnxexport]` extras, pins `onnxruntime-gpu==1.19.2` (uninstalling
any prior onnxruntime build first), and pins `Pillow==10.4.0` to avoid a
known version conflict. Ends by printing a reminder to restart the
runtime — required because onnxruntime-gpu does not reliably register
its CUDA provider without a fresh process.

### Cell 3 (markdown) — Restart warning
Instructs the user to use **Runtime → Restart session**, not "Restart
and run all", since the latter would re-run the pip installs unnecessarily.

### Cell 4 — Verify GPU and ONNX Runtime after restart
Confirms `torch.cuda.is_available()`, prints the GPU name, the installed
onnxruntime version, and asserts `CUDAExecutionProvider` is present in
`ort.get_available_providers()`. Also prints the installed `rfdetr`
package version via `pip show rfdetr`, which is the version check that
previously caught the deprecated-checkpoint warning.

### Cell 5 — Global config
All editable parameters live here:
- `SWEEP_BASE_THRESHOLD = 0.10` — minimum confidence kept during the
  single inference pass; every value in `SWEEP_THRESHOLDS` must be ≥ this.
- `SWEEP_THRESHOLDS` — the 13 confidence values evaluated in the sweep
  (0.10 through 0.95).
- `IOU_THRESHOLD = 0.5` — IoU cutoff used for TP/FP/FN matching,
  independent of confidence threshold.
- `SLICE_SIZE = 384` and `IMAGE_SIZE = 384` — SAHI tile size and the
  resolution the model expects as input. For RFDETRNano these should
  match the model's native training resolution.
- `OVERLAP_RATIO = 0.20` — SAHI tile overlap, applied to both height
  and width.
- `PT_WEIGHTS` — path to Sarah's checkpoint
  (`sarah_checkpoint_best_total.pth`).
- `ONNX_MODEL_PATH` — path to the exported ONNX file
  (`sara_rf_detr_nano.onnx`).
- `ANNOTATIONS_PATH`, `TEST_IMAGES_DIR` — local paths the test set is
  copied to in Cell 6.
- `OUTPUT_DIR`, `DRIVE_OUTPUT`, `CSV_PATH` — where results are written
  locally and mirrored to Drive.

### Cell 6 — Mount Drive and copy data to Colab
Mounts Google Drive, creates the local working directories, then
`rsync`s the test images from
`/content/drive/MyDrive/uavs2/images/` into
`/content/datasets/test/images/`, copies `_annotations.coco.json` into
place, and copies `data.yaml` (the YOLO-format config containing the
class list) into `/content/datasets/`. Prints a confirmation showing
whether the annotations file was found and how many images were copied.

### Cell 7 — GPU memory cleanup utility
Defines `cleanup_gpu_memory(obj=None, verbose=False)`. Optionally takes
a weak reference to an object being discarded, runs `gc.collect()`,
`torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`, and can print
allocated/reserved memory before and after for debugging.

### Cell 8 — Export PT checkpoint to ONNX
Reads class names from the COCO annotations file and builds
`CATEGORY_MAPPING`. If `ONNX_MODEL_PATH` already exists on Drive, the
export is skipped. Otherwise loads `RFDETRNano(pretrain_weights=...,
resolution=IMAGE_SIZE, num_classes=len(CLASS_NAMES))`, calls `.export()`,
and copies the resulting `output/rfdetr-nano.onnx` to the Drive path.

### Cell 9 — Load ONNX model into a SAHI `DetectionModel` subclass
Defines `RFDetrOnnxDetectionModel(DetectionModel)`:
- `load_model()` creates an `onnxruntime.InferenceSession` with
  `CUDAExecutionProvider` (falling back to CPU if no GPU is available)
  and records the input/output tensor names.
- `perform_inference(image)` resizes the tile to `self.image_size`,
  normalises with ImageNet mean/std, and runs the ONNX session. RF-DETR
  ONNX output format: `dets` is `[1, 300, 4]` normalised `cx,cy,w,h`;
  `labels` is `[1, 300, 1]` raw logits (confidence = sigmoid(logit)).
- `convert_original_predictions(shift_amount, full_shape)` converts each
  of the 300 candidate boxes from normalised tile-local coordinates to
  pixel coordinates, clips to tile bounds, **manually adds
  `shift_amount` to translate from tile-local to full-image
  coordinates** (SAHI's `ObjectPrediction` does not apply this shift
  automatically — confirmed by direct testing), and constructs an
  `ObjectPrediction` with `shift_amount=[0, 0]` so the shift is not
  applied a second time downstream.

Also defines `sahi_result_to_sv_detections()`, which converts a SAHI
`PredictionResult` into a `supervision.Detections` object for use with
the `supervision` metrics and visualisation API.

Finally instantiates `detection_model` with
`confidence_threshold=SWEEP_BASE_THRESHOLD` (kept low so the single
inference pass captures every candidate the sweep might need) and loads
it.

### Cell 10 — Load the dataset
Loads the test set as a `supervision.DetectionDataset` via
`sv.DetectionDataset.from_coco()`, and prints the image count and class
list.

### Cell 11 — Run inference once at the base threshold
Iterates over every image in the dataset, runs
`sahi.predict.get_sliced_prediction()` with `SLICE_SIZE` /
`OVERLAP_RATIO` from the config, converts each result to
`sv.Detections`, clips `class_id` into a valid range, and stores results
in two parallel lists: `targets` (ground truth) and `all_raw_preds`
(predictions at `SWEEP_BASE_THRESHOLD`). This is the only inference pass
in the notebook — every threshold in the sweep below re-filters these
stored predictions rather than re-running the model.

### Cell 12 — Threshold sweep
For each value in `SWEEP_THRESHOLDS`: filters `all_raw_preds` down to
predictions at or above that confidence, matches predictions to ground
truth using Hungarian assignment on IoU (`IOU_THRESHOLD`), and computes
TP / FP / FN, precision, recall, F1, and FP/TP ratio. `mAP50` and
`mAP50:95` are computed **once**, before the loop, on the full
unfiltered `all_raw_preds` — this avoids the earlier bug where computing
mAP separately at each threshold from pre-filtered predictions produced
an artificially truncated precision-recall curve. A confusion matrix
image is saved per threshold, and the full set of metrics is written to
`CSV_PATH` (`threshold_sweep_RF-DETR_nano_detector_extended.csv`) and
copied to Drive.

### Cell 13 — Single-image visualisation
Picks one image, filters its predictions at a `VISUALISE_THRESHOLD`,
draws ground truth boxes side-by-side with predicted boxes, and displays
the comparison using `sv.plot_images_grid()`.

### Cell 14 — Full dataset inference output: annotated images + COCO JSON
Re-uses the predictions stored in Cell 11 (no re-inference), filters
each image's detections at `SAVE_THRESHOLD = 0.45`, and produces two
outputs:
- A COCO-format predictions file (`predictions_coco.json`) with image
  metadata and one annotation entry per surviving detection, including
  score.
- An annotated copy of every test image with predicted boxes and
  confidence-labelled text drawn on top, saved to
  `OUTPUT_DIR/annotated_images/`.

Both outputs are copied to `DRIVE_OUTPUT` at the end.

### Cell 15 — Detection count summary
Reads back `predictions_coco.json`, prints the total detection count,
image count, and mean detections per image, and lists the five images
with the most detections — useful for spotting images where the model
is firing an unusually high number of (likely spurious) predictions.

### Cell 16 — Preview one annotated image
Displays the first annotated image inline in the notebook as a quick
visual sanity check.

### Cell 17 — Ground truth object size statistics
Computes mean/median/min width and height (in pixels) of every ground
truth box across the dataset. Used to compare the actual object scale
present in the test images against the tile size and model input
resolution, to judge whether `SLICE_SIZE` / `IMAGE_SIZE` are appropriate
for the objects being detected.
