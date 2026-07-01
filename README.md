# duane-uavs-inference-rf-detr-on-detection-dataset

ipynb files that are uploaded to this repo have been stripped of output to save space

```
jupyter nbconvert --clear-output --inplace rf_detr_onnx_sahi_inference.ipynb
```

the repo on ICEFISH where staging is done is

```
duane@ICEFISH duane-uavs-inference-rf-detr-on-detection-dataset
```

## `rf_detr_onnx_sahi_sweep_v5.ipynb`

RF-DETR ONNX + SAHI inference on 42 MP drone images, with a confidence
threshold sweep computed from a single inference pass. This version
evaluates Sarah's RFDETRNano checkpoint (trained under `rfdetr==1.7.1`).

Compared to v4, the ingest path now reads data with **supervision** and
**auto-detects YOLO vs COCO** format: input lives in a Drive folder
`uavs3` containing `images.tar.gz`, `labels.tar.gz`, and `data.yaml`
(the tarballs each expand to `train/ val/ test/`, one or two of which may
be empty). It also folds in a set of correctness/robustness fixes over
the earlier fixed pass — noted inline in the cells below — covering the
ONNX class-column handling, filename-keyed prediction lookup, RGB input
conversion, the ONNX-export filename, output provenance, and a few
out-of-range/empty-input guards.

Pure-inference runs with no labels only need the inference and save
cells; the sweep, confusion-matrix, and object-size cells require
ground-truth labels.

## rf_detr_onnx_sahi_sweep_v5_2.ipynb  updated version with output files tar into one file. Much faster store on google drive

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
package version via `%pip show rfdetr`, which is the version check that
previously caught the deprecated-checkpoint warning.

### Cell 5 — Global config
All editable parameters live here:
- `SWEEP_BASE_THRESHOLD = 0.10` — minimum confidence kept during the
  single inference pass; every value in `SWEEP_THRESHOLDS` must be ≥ this.
  Lowering it captures more of the low-confidence tail (a more complete
  precision-recall curve for mAP) at the cost of more memory during the
  sweep.
- `SWEEP_THRESHOLDS` — the 13 confidence values evaluated in the sweep
  (0.10 through 0.95).
- `IOU_THRESHOLD = 0.5` — IoU cutoff used for TP/FP/FN matching,
  independent of confidence threshold.
- `SLICE_SIZE = 640` and `IMAGE_SIZE = 640` — SAHI tile size and the
  resolution the model expects as input. The inline comment notes that
  384 matches the RFDETRNano native training resolution (896 for
  RFDETRLarge); adjust if the model input resolution differs.
- `OVERLAP_RATIO = 0.20` — SAHI tile overlap, applied to both height
  and width.
- `DATA_FORMAT = 'auto'` — `'yolo'`, `'coco'`, or `'auto'` (detect from
  what is present in `DRIVE_DATA_DIR`).
- `SPLIT = 'test'` — which split to run on. If the chosen split has no
  images, the next non-empty split is used instead (test → val → train).
- `DRIVE_DATA_DIR = '/content/drive/MyDrive/uavs3'` — the Drive folder
  holding the tarballs / annotations and `data.yaml`.
- `PT_WEIGHTS` — path to Sarah's checkpoint
  (`sarah_checkpoint_best_total.pth`).
- `ONNX_MODEL_PATH` — path to the exported ONNX file
  (`sarah_rf_detr_nano.onnx`).
- `DATASET_ROOT`, `DATA_YAML_PATH`, `ANNOTATIONS_PATH` (COCO only),
  `TEST_IMAGES_DIR`, `LABELS_DIR` (YOLO only) — local paths; the last
  three are populated by the ingest cell.
- `OUTPUT_DIR`, `DRIVE_OUTPUT`, `CSV_PATH` — where results are written
  locally and mirrored to Drive.

### Cell 6 — Mount Drive, auto-detect format, and stage data locally
Mounts Google Drive and creates the local working directories, then
resolves the data format (unless `DATA_FORMAT` forces one): YOLO if the
image/label tarballs or only a `data.yaml` are present, COCO if a
`*coco*.json` is present.
- **YOLO path:** extracts `images.tar.gz` and `labels.tar.gz` into
  `DATASET_ROOT/images` and `DATASET_ROOT/labels` (a helper collapses an
  accidental doubly-nested `images/images/` layer), copies `data.yaml`,
  locates the split directory (tolerating one level of nesting), and if
  the requested `SPLIT` has no images falls back to the next non-empty
  split. Sets `TEST_IMAGES_DIR` and `LABELS_DIR`, allows an empty labels
  dir for pure inference, and prints the split with image/label counts.
- **COCO path (legacy):** `rsync`s a flat images directory into
  `DATASET_ROOT/test/images`, copies `_annotations.coco.json`, and copies
  `data.yaml` if present.

### Cell 7 — GPU memory cleanup utility
Defines `cleanup_gpu_memory(obj=None, verbose=False)`. Optionally takes
a weak reference to an object being discarded, runs `gc.collect()`,
`torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`, and can print
allocated/reserved memory before and after for debugging.

### Cell 8 — Export PT checkpoint to ONNX
Reads class names from `data.yaml` (YOLO) or the COCO annotations file
(COCO) and builds `CLASS_NAMES` / `CATEGORY_MAPPING`. If `ONNX_MODEL_PATH`
already exists on Drive, the export is skipped. Otherwise loads
`RFDETRNano(pretrain_weights=..., resolution=IMAGE_SIZE,
num_classes=len(CLASS_NAMES))`, calls `.export()`, and copies the result
to the Drive path. Rather than assuming a fixed output filename, it globs
`output/*.onnx` and copies the newest match (the exported name varies by
`rfdetr` version), raising a clear error if the export produced no
`.onnx`.

### Cell 9 — Load ONNX model into a SAHI `DetectionModel` subclass
Defines `RFDetrOnnxDetectionModel(DetectionModel)`:
- `load_model()` creates an `onnxruntime.InferenceSession` with
  `CUDAExecutionProvider` (falling back to CPU if no GPU is available),
  records the input/output tensor names, and prints the active provider.
- `perform_inference(image)` resizes the tile to `self.image_size`,
  normalises with ImageNet mean/std, and runs the ONNX session. RF-DETR
  ONNX output format: `dets` is `[1, 300, 4]` normalised `cx,cy,w,h`;
  `labels` is `[1, 300, num_classes]` raw logits (confidence =
  sigmoid(logit)). The head has one logit column per class the model was
  **trained** with, which is not necessarily 1 even for a single-class
  dataset. A module-level `OBJECT_CLASS_COLUMN` (default `0`) selects
  which column is read as the object class, and a one-time `[INFO]`
  message is printed if the head has more than one column, so a wrong
  column choice is visible rather than silent. (For Sarah's checkpoint the
  head is `[1, 300, 2]`; column 0 is the object class and column 1 is an
  unused/reserved column — verified with the per-column sigmoid
  diagnostic — so the default of 0 is correct.)
- `convert_original_predictions(shift_amount, full_shape)` converts each
  of the 300 candidate boxes from normalised tile-local coordinates to
  pixel coordinates, clips to tile bounds, **manually adds
  `shift_amount` to translate from tile-local to full-image
  coordinates** (SAHI's `ObjectPrediction` does not apply this shift
  automatically — confirmed by direct testing), and constructs an
  `ObjectPrediction` with `shift_amount=[0, 0]` so the shift is not
  applied a second time downstream. Confidence is the sigmoid of the
  selected column and every detection is assigned `category_id=0`
  (single-class evaluation).

Also defines `sahi_result_to_sv_detections()`, which converts a SAHI
`PredictionResult` into a `supervision.Detections` object for use with
the `supervision` metrics and visualisation API.

Finally instantiates `detection_model` with
`confidence_threshold=SWEEP_BASE_THRESHOLD` (kept low so the single
inference pass captures every candidate the sweep might need) and loads
it.

### Cell 10 — Load the dataset
Loads the test set as a `supervision.DetectionDataset`, matching
`DATA_FORMAT`: `sv.DetectionDataset.from_yolo()` (images dir, labels dir,
`data.yaml`) or `sv.DetectionDataset.from_coco()` (images dir,
annotations JSON). Prints the image count and class list.

### Cell 11 — Run inference once at the base threshold
Iterates over every image in the dataset, opens each as RGB (an explicit
`.convert('RGB')` so grayscale / RGBA / palette inputs don't break the
ImageNet normalisation), runs `sahi.predict.get_sliced_prediction()` with
`SLICE_SIZE` / `OVERLAP_RATIO` from the config, converts each result to
`sv.Detections`, clips `class_id` into a valid range, and stores results
in three parallel lists: `targets` (ground truth), `all_raw_preds`
(predictions at `SWEEP_BASE_THRESHOLD`), and `all_raw_names` (the image
basename for each entry, kept aligned 1:1 so later cells can look
predictions up by filename). This is the only inference pass in the
notebook — every threshold in the sweep below re-filters these stored
predictions rather than re-running the model.

### Cell 12 — Threshold sweep
For each value in `SWEEP_THRESHOLDS`: filters `all_raw_preds` down to
predictions at or above that confidence, matches predictions to ground
truth using Hungarian assignment on IoU (`IOU_THRESHOLD`), and computes
TP / FP / FN, precision, recall, F1, and FP/TP ratio. `mAP50` and
`mAP50:95` are computed **once**, before the loop, on the full
`all_raw_preds` — this avoids the earlier bug where computing mAP
separately at each threshold from pre-filtered predictions produced an
artificially truncated precision-recall curve. (Because `all_raw_preds`
is itself floored at `SWEEP_BASE_THRESHOLD`, the reported mAP is a mild
underestimate; lower the base threshold for a more complete curve.) A
confusion matrix image is saved per threshold, and the full set of
metrics is written to `CSV_PATH`
(`threshold_sweep_RF-DETR_nano_detector_extended.csv`) and copied to
Drive under the same filename.

### Cell 13 — Single-image visualisation
Picks one image (`IMAGE_INDEX`, clamped to the dataset size so an
out-of-range index falls back to the last image rather than raising),
opens it as RGB, filters its predictions at a `VISUALISE_THRESHOLD`,
draws ground truth boxes side-by-side with predicted boxes, and displays
the comparison using `sv.plot_images_grid()`.

### Cell 14 — Full dataset inference output: annotated images + COCO JSON
Re-uses the predictions stored in Cell 11 (no re-inference), looking each
image's detections up **by filename** (robust to any ordering or
extension mismatch between the dataset iteration and the on-disk file
list; `.tif` is included alongside `.tiff`), filters each image's
detections at `SAVE_THRESHOLD = 0.45`, and produces two outputs:
- A COCO-format predictions file (`predictions_coco.json`) with image
  metadata and one annotation entry per surviving detection, including
  score. The model provenance in the file is labelled `RFDETRNano ONNX`.
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
truth box across the dataset (and prints a short message instead of
erroring when there are no ground-truth boxes, e.g. a pure-inference
split). Used to compare the actual object scale present in the test
images against the tile size and model input resolution, to judge
whether `SLICE_SIZE` / `IMAGE_SIZE` are appropriate for the objects
being detected.
