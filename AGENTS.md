# AGENTS.md

Living notes for agents working in this workspace. Update this file after every codebase, tooling, dataset-prep, or training-pipeline change.

## Project Context

- Main repo: `lgp-main`
- Main application: Windows C# WinForms AOI host-computer app for optical filter defect inspection.
- Machine-control stack: Hikvision/MVS cameras, HALCON/HalconDotNet, MySQL, serial/Modbus, STM32.
- YOLO sidecar: `aoi_yolo`, used for offline dataset conversion, training, and evaluation without modifying the production C# machine-control app.

## Current Objectives

1. Understand and preserve the existing industrial AOI control application.
   - Treat the C# WinForms app as the machine-control host computer.
   - Avoid destabilizing camera sequencing, STM32/Modbus commands, MySQL writes, tray state, and operator UI.
   - Replace or augment only the inspection backend after offline validation.

2. Build a lightweight, fast, reliable YOLO-style inspection backend.
   - Primary motivation: current HALCON/deep-learning inspection is too slow and recent tinkering reduced success rate.
   - Keep the model lightweight enough for production AOI throughput on the RTX 5060.
   - Prefer segmentation over plain boxes because scratches and pinholes require geometry/measurement, not just classification.

3. Use the provided BMP/JSON test batches as ground truth.
   - BMP files are the canonical source images.
   - JSON files are LabelMe-style metadata with polygon defect annotations.
   - Ignore `imageData` for training because it is a redundant PNG-encoded copy of the BMP pixels.
   - Compare predictions against JSON polygons to measure accuracy.

4. Improve model accuracy toward production-grade performance.
   - The first broad 9-class model is not near the desired 95-98% target.
   - Current best long-run broad model: box mAP50 about 61.9%, mask mAP50 about 47.5%.
   - Next experiment should focus on well-represented core classes: `pinhole`, `inpinhole`, and `scratch`.
   - Rare classes (`chipping`, `inswab`, `fingerprint`, `splash`, `corrosion`, etc.) should be merged, excluded, or handled later after more examples are available.
   - Consider excluding `edge` unless it is explicitly needed for localization or product-boundary logic.

5. Prepare for eventual integration into the C# AOI app.
   - Do not wire YOLO into production control until offline metrics are strong.
   - Target integration point is the existing detection call chain around `Detectionui.Start_Detection()` and `DetectionWithDL2.ImageProcess(...)`.
   - Future integration should return normalized defect records: image name, class, confidence, mask/box, centroid, area/length/width, and pixel coordinates.
   - Reuse existing coordinate mapping, database writes, UI display, and machine-control decisions where possible.

## Immediate Next Steps

1. Build a conservative edge-policy + product-face masking stage.
   - Do not use arbitrary 300x300 crops for defect training; they lose stream/edge context.
   - Separate likely edge-contact defects from ordinary interior defects.
   - Edge-contact defects should be assigned only after a pixel dry-sweep detects a real object edge and the JSON edge/defect metadata cross-check agrees.
   - Defects merely near an edge, but visibly separated from the edge, should remain full-image inspection cases.
   - Corners are a separate case: a valid corner defect must contact both a vertical and horizontal edge.
   - Current conservative default: `edge_gap_px=5`, plus matching JSON edge orientation and distance.

2. Build a scalable product-face / inspectable-surface masking stage.
   - Goal: automatically mask out background, fixture, black voids, non-inspected side regions, and dust outside the real product face before defect scoring.
   - Do not rely on hand-drawn ROI masks at production scale.
   - Use offline auto-annotation / pseudo-labeling first, then train a lightweight real-time mask model.

3. Create a new sidecar folder, likely `aoi_yolo/product_mask/`.
   - Suggested files:
     - `auto_generate_product_masks.py`
     - `train_product_mask.py`
     - `evaluate_masked_defects.py`
     - `dataset_product_mask/`
     - `runs/`
   - Keep it separate from defect model experiments and from the production C# AOI app.

4. Generate first-pass product-face masks.
   - Use existing `edge` annotations where helpful.
   - Combine with image-processing cues: thresholding, contours, bright/dark region separation, morphology, largest plausible rectangular/surface region, and known image size.
   - Save generated masks and overlays for visual review.
   - If local tools or network access allow later, consider SAM/SAM2/CVAT/Label Studio style auto-annotation to bootstrap better masks, but do not run heavy SAM-like models in production.

5. Train a lightweight real-time product-mask model.
   - Candidate model: small YOLO segmentation or compact UNet-style model.
   - Input: full BMP.
   - Output: inspectable product-face mask.
   - Production path should be:
     `product mask model -> defect YOLO only inside mask -> Option A threshold/area cascade`.

6. Re-evaluate defect precision after applying the product-face mask.
   - Compare current Option A baseline:
     recall about 83.56%, precision about 83.22%.
   - Test whether masking removes isolated dust/background false positives without removing true defects.
   - Keep reporting class-level recall/precision for merged `pinhole` and `scratch`.

7. Keep the C# AOI app untouched until offline product-mask + defect metrics and latency are acceptable.

## Changelog

### #12 - GitHub Repo Decoupling And Data Exclusion

- Prepared the migrated workspace to become its own GitHub repository separate from `lgp-main`.
- Clarified repo boundary: `lgp-main` remains vital to local AOI development and future integration work, but is excluded from this GitHub repo only to decouple the upload/history from the production host-computer app.
- Added root `.gitignore` rules to keep the production C# app folder, source BMP batches, LabelMe/image metadata JSON, generated YOLO datasets, product-mask pseudo-label outputs, review overlays, model weights, training runs, virtualenvs, and caches out of Git.
- Added root `requirements.txt` for the AOI YOLO sidecar Python dependencies, with a note to install CUDA-matched PyTorch wheels first on the production GPU machine.
- Added tracked documentation/template files inside `test images/` so a clone shows where to place `0001.bmp` / `0001.json` style local batches, while real BMP files and per-image JSON metadata remain ignored.
- Added root `README.md` describing the public sidecar repo boundary, local data policy, and dependency entry point before the first GitHub push.
- Intended tracked scope is sidecar code/docs/config only unless a future change explicitly moves public-safe sample data or release artifacts into a dedicated tracked folder.

### #11 - Product-Face Pseudo-Label Masks Generated After Migration

- Read the old C: migration artifacts as requested:
  - `C:\Users\Catnip\Documents\visual micro measurements\NOTICE.md`
  - `C:\Users\Catnip\Documents\visual micro measurements\migration_to_H_robocopy.log`
- Confirmed the active migrated workspace is `H:\visual micro measurements`; the robocopy log reports 62,751 files, 6,592 directories, about 71.693 GB, and 0 failed copies.
- Verified that `H:\visual micro measurements\.git` exists but is empty/unusable: `git rev-parse --show-toplevel` and `git status --short --branch` both report `fatal: not a git repository`.
- Started the product-face masking sidecar task in `aoi_yolo/product_mask/`.
- Updated `aoi_yolo/product_mask/auto_generate_product_masks.py`:
  - Added bounded working-resolution mask selection with `--max-dim` to keep full-resolution outputs while reducing expensive morphology work.
  - Added `--resume` support so interrupted generation can skip completed mask/review/annotation triplets and still write a summary.
  - Added JSON-edge-aware pseudo-labeling for slender horizontal/vertical edge annotations.
  - Added conservative filtering for broad/non-axis edge polygons so they fall back to image processing instead of creating slivers.
  - Added paired-edge handling so two parallel edges produce a between-edges product-face band.
- Generated first-pass product-face pseudo-label outputs:
  - `aoi_yolo/product_mask/dataset_product_mask/masks/`: 907 PNG masks.
  - `aoi_yolo/product_mask/dataset_product_mask/review_bmp/`: 907 baked BMP overlays.
  - `aoi_yolo/product_mask/dataset_product_mask/annotations/`: 907 LabelMe-style JSON files with `imageData: null`.
  - `aoi_yolo/product_mask/dataset_product_mask/summary.csv`: 907 rows.
- Final QA summary:
  - Method distribution includes 239 `percentile_20`, 133 `otsu_inverse`, 129 `edge_json_vertical_right`, 99 `edge_json_horizontal_down`, 98 `edge_json_horizontal_up`, 75 `edge_json_vertical_left`, 25 `edge_json_horizontal_between`, and 11 `edge_json_vertical_between`.
  - Area ratio range is about 2.45% to 97.76%, average about 60.56%.
  - 0 masks are over 98% area after paired-edge correction; 2 masks are under 5% and should be manually reviewed before training.
  - Spot-checked `0001.bmp` and `1407.bmp` overlays; `1407.bmp` now uses `edge_json_horizontal_between` instead of the earlier full-frame mask.
- Next recommended step: visually review low-area and high-area overlays from `dataset_product_mask/review_bmp`, then convert accepted pseudo-labels into a YOLO/UNet product-mask training dataset. Do not wire this into the production C# AOI app yet.

### #10 - Workspace Migrated To H Drive

- Workspace was copied from `C:\Users\Catnip\Documents\visual micro measurements` to `H:\visual micro measurements` using `robocopy`.
- Copy summary: 62,751 files, 6,592 directories, about 71.693 GB, 0 failed.
- The old C: workspace was cleaned after migration and now retains only `NOTICE.md` and `migration_to_H_robocopy.log`.
- Future work should be opened from `H:\visual micro measurements`; the old C: thread/workspace does not automatically retarget itself.

### #9 - Dust / Benign Speck Hard-Negative Sidecar Started

- User identified dust/benign specks on the product as a likely source of false positives.
- Direction: build a sidecar hard-negative dataset rather than contaminating existing YOLO datasets.
- Added `aoi_yolo/hard_negatives_dust/prepare_dust_filter_dataset.py` before migration.
- Intended outputs:
  - `aoi_yolo/hard_negatives_dust/source_clean_test_images/`: backup/copy of original BMP/JSON source with JSON `imageData` removed.
  - `aoi_yolo/hard_negatives_dust/candidate_filter_dataset/`: classification crops split into `real_defect` and `dust_or_benign`.
- Mining strategy:
  - Use current best core defect model `aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt`.
  - Run low-confidence inference on the cleaned source copy.
  - Predictions that do not overlap JSON core defects become `dust_or_benign` hard negatives.
  - JSON core defects become `real_defect` positives.
- The first full mining command was interrupted before completion; rerun from the H: workspace.
- Candidate command from `H:\visual micro measurements`:
  ```powershell
  $env:YOLO_CONFIG_DIR='H:\visual micro measurements\aoi_yolo\ultralytics_config'
  $env:MPLCONFIGDIR='H:\visual micro measurements\aoi_yolo\matplotlib_config'
  .\.venv_yolo\Scripts\python.exe aoi_yolo\hard_negatives_dust\prepare_dust_filter_dataset.py --source "test images" --out aoi_yolo\hard_negatives_dust --weights aoi_yolo\runs\core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best\weights\best.pt --imgsz 1280 --conf 0.05 --fp-iou 0.05 --crop-size 192 --pad 48 --balance-positives
  ```

### #8 - ROI-Mask Training Attempt From Crop Windows

- User changed tactics: keep original full BMP context and use the cropped/annotated windows as masks showing the AI which parts of the image deserve close inspection.
- Added `aoi_yolo/edge_policy_to_roi_masks.py`.
- Added `--images-dir` support to `aoi_yolo/labelme_to_yolo_seg.py` so LabelMe JSON labels can live separately from images.
- Generated `aoi_yolo/dataset_roi_masks_labelme`:
  - 776 original full-size BMPs.
  - 103 images with reconstructed `edge_roi` masks from edge-policy crop windows.
  - 673 images with `full_roi` masks for non-edge/interior inspection.
  - JSON `imageData` remains `null`.
  - Review BMPs show baked ROI overlays.
- Converted to `aoi_yolo/dataset_roi_masks_yolo` and trained:
  - Run: `aoi_yolo/runs/roi_mask_yolo11n_seg_e20_i640`.
  - Classes: `edge_roi`, `full_roi`.
  - Result: overall mask mAP50 about 47.0%, but this was dominated by `full_roi`; validation showed `full_roi` mask mAP50 about 93.9% and `edge_roi` mask mAP50 about 0.9%.
- Converted to `aoi_yolo/dataset_edge_roi_yolo` with only `edge_roi` positives and full/interior images as empty-label backgrounds, then trained:
  - Run: `aoi_yolo/runs/edge_roi_yolo11n_seg_e30_i640`.
  - Result: edge ROI mask recall about 10%, mask mAP50 about 0.15%; not useful.
- Conclusion: using defect-triggered crop windows as ROI masks is not sufficient to train a reliable part/attention model. The masks describe sparse defect windows, not stable product-part geometry.
- Next direction should be explicit part-context/product-surface labels:
  - `interior_face`
  - `edge_vertical`
  - `edge_horizontal`
  - `corner`
  - `background/fixture/void`
  - Then route defect detection: no edge/corner context means use full/interior inspection, not crop strips.

### #7 - Edge-Policy Dataset And Cleanup

- User clarified that arbitrary 300x300 crops are unsafe for picture-stream inspection because defects may need full edge context.
- Added `aoi_yolo/prepare_edge_policy_dataset.py`.
- New policy:
  - Run a pixel dry-sweep of each BMP to detect likely horizontal/vertical object edges.
  - Use JSON `edge` annotations only as cross-reference metadata, not as the primary crop chooser.
  - If a defect truly contacts a detected edge and the JSON edge orientation/distance agrees, save an edge-running strip and mark it `auto_pass_defective=True`.
  - If no true edge-contact defect is present, keep the original full BMP and write a cleaned LabelMe JSON with `imageData: null`.
  - Reserve a separate `edge_corner_defects` category for defects that contact both vertical and horizontal edges.
- Tightened edge classification after visual review of `0002_edge000_vertical_x2333_y0.bmp`; that pinhole was near/crossed by a false vertical line but was not an edge defect.
- Conservative default is now `edge_gap_px=5` with matching JSON edge orientation and `json_edge_max_distance=120`.
- Generated `aoi_yolo/dataset_edge_policy`:
  - 135 edge defect strips with baked review BMPs.
  - 673 full-image no-edge-defect cases.
  - 0 corner cases under the current conservative threshold.
  - 1 malformed non-LabelMe JSON skipped: `drive_manifest.json`.
- Cleaned obsolete experimental datasets from this thread:
  - `aoi_yolo/dataset_crops_300`
  - `aoi_yolo/dataset_crops_300_dryrun`
  - `aoi_yolo/dataset_strips_300`
  - `aoi_yolo/dataset_strips_300_dryrun`
  - `aoi_yolo/dataset_edge_strips_300_dryrun`
  - `aoi_yolo/dataset_edge_policy_dryrun`

### #6 - Next Direction: Scalable Product-Face Masking

- User observed that many test images show only one side/region of an assumed rectangular part, with substantial irrelevant background/fixture/void area.
- Agreed that hand-drawn masks are not scalable; next step should be an automated product-face / inspectable-surface masking stage.
- Proposed scalable pipeline:
  `heavy/offline auto-labeler or image-processing pseudo-labels -> reviewed product-face masks -> lightweight real-time product-mask model -> defect YOLO inside mask -> Option A cascade`.
- Mentioned relevant ecosystem concepts for the next chat to investigate or use if useful: Segment Anything/SAM-style auto-annotation, CVAT/Label Studio assisted annotation, pseudo-labeling, and lightweight YOLO/UNet segmentation for production inference.
- The production AOI line should not run a heavy auto-labeler; it should run a trained lightweight mask model.
- Added immediate next steps to create `aoi_yolo/product_mask/` and evaluate whether product-face masking improves defect precision by removing dust/background false positives.

### #5 - Hybrid Option A And Option B Comparison

- Created `aoi_yolo/hybrid_models/option_a_threshold_cascade` for the no-training threshold/area cascade.
- Created `aoi_yolo/hybrid_models/option_b_crop_classifier` for the YOLO-candidate crop classifier experiment.
- Added `aoi_yolo/hybrid_models/option_b_crop_classifier/train_and_eval_crop_filter.py`.
- Option A balanced result on `dataset_core2_bg_v2`: recall about 83.56%, precision about 83.22%.
- Option B generated 5289 train candidates and 1258 validation candidates from the YOLO base model at `conf=0.05`.
- Trained a small binary CNN crop filter for 12 epochs and saved `candidate_filter.pt`.
- Corrected Option B final evaluation to use one-to-one JSON matching so duplicate candidates cannot inflate recall.
- Option B best comparable point at filter threshold 0.50: recall about 83.01%, precision about 63.52%.
- Option B higher thresholds did not beat Option A: threshold 0.60 reached recall about 60.00%, precision about 76.04%; threshold 0.70 reached recall about 37.40%, precision about 80.06%.
- Current recommendation: use Option A for the precision-focused hybrid. Option B needs richer inputs such as larger context, shape/geometry features, class-specific classifiers, or hard-negative mining.
- Added `aoi_yolo/hybrid_models/COMPARISON.md` with the side-by-side results.
- Added ROI and edge-exclusion filtering support to `aoi_yolo/evaluate_json_polygons.py`.
- Tested edge-margin masking on Option A balanced settings: 10 px edge exclusion reached recall about 81.64%, precision about 83.59%; 25 px edge exclusion reached recall about 79.32%, precision about 83.67%.
- Edge-margin masking alone is not enough because it removes true defects too. Next masking experiment should define the actual inspectable product face/side, likely from product geometry or machine coordinates.

### #4 - Expanded Training Data And Precision Diagnosis

- User added a much larger labeled batch to `test images`.
- New audit summary: 907 usable JSON files, 907 BMP files, 1816 `pinhole`, 1593 `inpinhole`, 543 `scratch`, 1051 `edge`, plus more rare classes.
- Created `aoi_yolo/dataset_core2_bg_v2` with original `inpinhole` merged into `pinhole`, `scratch` kept separate, and empty-label background images included.
- New split: 726 train images, 181 val images, classes `pinhole` and `scratch`.
- Fine-tuned from `core2_bg_yolo11n_seg_e100_i1280_mask2_from_core2/weights/best.pt` into `aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best`.
- Used `batch=1` at `imgsz=1280` to avoid prior RTX 5060 VRAM failures.
- Training reached epoch 44 before the foreground command timed out; saved `best.pt` and `last.pt`.
- Expanded-data AOI-style eval for `best.pt` at `conf=0.05`, `hit_iou=0.05`: overall recall about 95.48%, precision about 55.41%; `pinhole` recall 96.20%, precision 56.72%; `scratch` recall 90.82%, precision 47.85%.
- At `conf=0.10`, overall recall about 92.74%, precision about 64.91%; at `conf=0.20`, recall about 88.08%, precision about 76.82%.
- Telemetry showed only 11 of 561 false positives at `conf=0.05` overlapped ignored labels at IoU >= 0.05, so low precision is mostly real over-calling rather than hidden rare-class annotations.
- Simple class-specific confidence and geometry sweeps could not preserve 95% recall while materially improving precision; best recall-guarded precision was about 57.98%.
- False-positive diagnosis: scratch has many duplicate/overlapping predictions around real scratches, but pinhole false positives are mostly isolated normal specks. Next direction should be hard-negative mining, a second-stage candidate classifier/filter, or explicit normal/benign-speck labeling.
- Added class-specific confidence and area filtering support to `aoi_yolo/evaluate_json_polygons.py`.
- Precision-focused operating points on `core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt`:
  - Balanced: `pinhole conf >= 0.30`, `scratch conf >= 0.20`, `pinhole area >= 150`, `scratch area >= 300`; overall recall about 83.56%, precision about 83.22%.
  - High-confidence: `pinhole conf >= 0.50`, `scratch conf >= 0.40`, same area filters; overall recall about 69.59%, precision about 93.21%.
  - Precision-first: `pinhole conf >= 0.60`, `scratch conf >= 0.45`, same area filters; overall recall about 60.27%, precision about 95.86%.

### #3 - Focused Core Training Reached AOI-Style 95% Recall Ballpark

- Created `aoi_yolo/dataset_core3` for `pinhole`, `inpinhole`, and `scratch`.
- Trained `core3_yolo11n_seg_e250_i1024_from_defects` from the previous broad defect checkpoint.
- Trained higher-resolution `core3_yolo11n_seg_e120_i1280_mask2` with `imgsz=1280` and `mask_ratio=2`.
- Found that the 3-class model often detects defects but confuses `pinhole` and `inpinhole`: class-agnostic recall reached about 97.45% at low confidence, while class-specific precision/recall remained poor.
- Added `--rename old=new` support to `aoi_yolo/labelme_to_yolo_seg.py`.
- Created `aoi_yolo/dataset_core2`, merging original `inpinhole` into `pinhole` and keeping `scratch` separate.
- Trained `aoi_yolo/runs/core2_yolo11n_seg_e120_i1280_mask2_from_core3/weights/best.pt`.
- Added `aoi_yolo/evaluate_json_polygons.py` to compare YOLO predicted masks against original LabelMe JSON polygons and report AOI-style hits, misses, false positives, recall, and precision.
- Current best AOI-style operating point for the merged model: `conf=0.03`, `hit_iou=0.05`, `imgsz=1280`, with overall recall about 95.08% and precision about 56.85%.
- High-recall screening setting: `conf=0.01`, `hit_iou=0.05`, with overall recall about 97.96% but precision about 45.87%.
- Installed ONNX export dependencies into `.venv_yolo`: `onnx`, `onnxruntime`, and `onnxslim`.
- Exported the merged focused checkpoint to `aoi_yolo/runs/core2_yolo11n_seg_e120_i1280_mask2_from_core3/weights/best.onnx`.
- RTX 5060 PyTorch validation latency at `imgsz=1280`: about 0.8 ms preprocess, 8.6 ms inference, and 1.8 ms postprocess per image.
- This is not production-ready yet: scratch recall is about 91.89% at the 95.08% overall setting, and false positives remain too high.

### #2 - Persisted Current And Future Objectives

- Added `Current Objectives` so future chats can resume the AOI/YOLO migration plan.
- Added `Immediate Next Steps` for the next model-training and evaluation experiments.
- Documented that BMP files are canonical images and JSON `imageData` is redundant PNG data.
- Documented the current target integration strategy: keep machine control stable and replace only the inspection backend after offline validation.

### #1 - YOLO AOI Sidecar And Initial Training Pipeline

- Added `aoi_yolo/README.md` documenting the offline YOLO pilot workflow.
- Added `aoi_yolo/audit_dataset.py` to summarize LabelMe-style JSON labels and polygon statistics.
- Added `aoi_yolo/labelme_to_yolo_seg.py` to convert BMP/JSON pairs into YOLO segmentation datasets.
- Created `aoi_yolo/dataset_defects` from the provided test images, excluding `edge` as a likely boundary/positioning label.
- Created `.venv_yolo` and installed Ultralytics plus CUDA-enabled PyTorch.
- Verified CUDA training on RTX 5060 with PyTorch `2.11.0+cu128`.
- Ran YOLO11n segmentation smoke training for 5 epochs at `imgsz=640`.
- Ran a longer YOLO11n segmentation training attempt at `imgsz=1024`, reaching epoch 147 before interruption.
- Current best long-run weights: `aoi_yolo/runs/defects_yolo11n_seg_e150_i1024/weights/best.pt`.
- Current long-run validation summary: box mAP50 about 61.9%, mask mAP50 about 47.5%; not production-grade yet.
- Next recommended step: train a focused model on the well-represented classes such as `pinhole`, `inpinhole`, and `scratch`, then compare against JSON ground truth.

