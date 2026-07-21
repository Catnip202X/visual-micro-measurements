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

## Dataset Sanitation And Statistical Validation Policy

### Representation And Dataset Versions

- Audit both annotation-instance counts and independent positive-image counts for every class before training.
- Keep the training set and diagnostic validation folds approximately class-balanced, while retaining a separate locked test set that reflects real production prevalence.
- Do not balance a class only by duplicating or augmenting the same rare images. Representation must cover independent cameras, trays, acquisition sessions, lighting, product regions, defect sizes, and appearances.
- Merge or defer classes that lack enough independent examples for reliable learning and evaluation. Keep `pinhole` and `inpinhole` merged unless a production decision requires separation and sufficient independent evidence exists.
- Any rebalance, relabel, source addition, or split change creates a new versioned dataset with immutable manifests and a documented class/image audit.

### Split And Shuffle Rules

- Shuffle deterministically with a recorded random seed after grouping related images.
- Use stratified group splitting by defect class and acquisition group. Images from the same product, tray, sequence, camera session, or near-duplicate capture group must remain in one split.
- Run duplicate and near-duplicate checks before freezing manifests.
- Never reshuffle a completed experiment's train, validation, or test membership. Compare models on the same manifests.

### Deep Validation Protocol

- Evaluate the complete holdout population in operational batches of about 300-400 images and aggregate all batches before calculating final metrics.
- Run five grouped cross-validation folds so distinct acquisition groups form each holdout and every group serves as holdout data once.
- Do not treat five repeated inference passes on identical weights and images as additional independent validation evidence. Repeated training seeds may measure optimization variance, but do not replace grouped folds.
- After model and threshold selection, evaluate once on the untouched production-distribution test set.
- Match predictions one-to-one against original LabelMe JSON polygons. Report AOI event detection at `hit_iou=0.05` and stricter localization metrics such as mask IoU and mask mAP50 separately.

### Per-Class Statistical Reporting

- For every class, report annotated objects, independent images, true positives, false positives, false negatives, recall, precision, F1, AP50/mask AP50, and mask IoU where applicable.
- Report 95% confidence intervals, the mean and variation across the five grouped folds, class confusion, and breakdowns by defect size, camera, product region, and edge/interior context.
- Mark a class statistically inconclusive when its holdout support is too small. Operational batches are processing units, not independent experiments.

### Tile And Multi-Stage Evaluation

- Before reporting fine-model accuracy, publish candidate/ground-truth coverage for any coarse-to-fine or dynamic-tile experiment.
- Categorize every missed object as a coarse-stage coverage failure, fine-stage miss, class confusion, or mask-coverage failure.
- Compare tile and full-image systems on identical manifests, JSON ground truth, thresholds, and one-to-one matching.

### Changelog Timestamp Rule

- Starting with `#20`, every new changelog heading must end with the local completion time formatted as `YYYY-MM-DD HH:MM +/-HH:MM`.
- Do not retrofit `#1`-`#19`, because their exact completion times are not reliably documented.

## Immediate Next Steps

1. Version and audit the dataset before further training.
   - Record per-class annotation instances, independent positive images, acquisition groups, defect-size bins, camera/source domains, and known hard negatives.
   - Audit the other team's 3,085-image dataset for provenance, label semantics, duplicates, camera/domain compatibility, and permission before considering any merge.

2. Create grouped, class-aware dataset manifests.
   - Build approximately balanced training and diagnostic-validation folds using stratified grouping by product/tray/sequence/camera session.
   - Reserve a locked test set that retains real production class prevalence.

3. Strengthen hard-negative coverage.
   - Manually review dust, benign specks, edge reflections, background bright points, glue/support texture, fixture/void regions, and fragmented bright regions.
   - Keep product-face masks as context features rather than hard gates because direct gating reduced recall to about 52%.

4. Improve independent representation for weak classes.
   - Prioritize thin/weak `scratch` examples across size, direction, contrast, edge distance, camera, and acquisition batch.
   - Merge or defer rare classes until each intended production class has enough independent train and holdout support.

5. Run a controlled native-detail tile experiment.
   - Compare full-image inference with dynamic tiles that guarantee candidate coverage, not merely a minimum tile count.
   - Report coarse-stage coverage, fine-stage misses, class confusion, mask coverage, latency, and final one-to-one AOI metrics separately.

6. Consider one thesis-style segmentation benchmark after data corrections.
   - Test a compact U-Net using multi-scale input, SE attention, and multi-scale feature fusion only as a controlled offline comparison.
   - Use the same manifests, JSON ground truth, AOI metrics, strict mask metrics, and latency reporting as YOLO.

7. Perform deep validation before model selection.
   - Run five grouped folds, process complete holdouts in batches of about 300-400 images, aggregate the results, and report per-class confidence intervals and failure categories.
   - Evaluate the selected model once on the untouched production-distribution test set.

8. Keep the production C# AOI application untouched until offline accuracy, statistical confidence, and latency gates are acceptable.

## Changelog

### #23 - Historical-Best Model Two-Run Deep Evaluation Direction (2026-07-21 16:07 -04:00)

- Approved a deep offline evaluation of the historical best model at `aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt`, using its frozen 1280-pixel input and historical balanced cascade that previously produced 83.56% recall and 83.22% precision.
- Prepare two mutually exclusive, group-aware randomized datasets of exactly 500 images each from canonical BMP/LabelMe pairs that are excluded from the model's historical train and validation membership. Preserve production prevalence and freeze seeds, hashes, groups, and manifests.
- Simulate production timing in a dedicated first pass: load each image before timing, then time model preprocessing, synchronized inference, prediction/mask postprocessing, record conversion, and the frozen confidence/area cascade. Do not perform JSON ground-truth work inside or between timed images.
- Run inference exactly once per measured image. Cache predictions and timing rows, then perform a separate offline scoring loop that consumes the cache without invoking the model again.
- For each run and defect type, report dataset image/object support, TP/FP/FN, slipped defects, recall, precision, F1, mean matched mask IoU, and 95% Wilson intervals.
- Report production-path timing distributions, fixed timing-bracket counts, and timing grouped by pinhole-only, scratch-only, mixed, and no-target/context images. Test relationships with defect category, object counts, prediction counts, dimensions, and pixel count.
- Compare the two independent runs using distribution tests, bootstrap differences, effect sizes, bracket-composition tests, per-defect metric differences, and correction for multiple category comparisons. Composition differences must be separated from evidence of model instability.
- The approved design is `docs/superpowers/specs/2026-07-21-historical-best-two-run-deep-evaluation-design.md`. This entry records direction only and does not claim that manifests, inference, statistics, or reports are complete.
- The executable implementation plan is `docs/superpowers/plans/2026-07-21-historical-best-two-run-deep-evaluation.md`; it enforces test-first manifest exclusion, exactly-once production inference, cached CPU-only scoring, per-run timing/accuracy analysis, and between-run statistical comparison.
- Implementation progress: added the isolated `aoi_yolo/historical_best_deep_eval/` package with frozen historical weights/thresholds, two deterministic seeds, fixed timing brackets, portable `AOI_DATA_ROOT` resolution for private local data, and a SHA-256 experiment identity. Configuration tests pass.
- Leakage audit hashed all 907 historical train/validation images and found 2,170 eligible canonical images outside that membership. Frozen group-aware manifests contain exactly 500 images each with zero shared stems, groups, or hashes. Run 1 contains 935 pinhole and 103 scratch objects across 295/62 positive images; run 2 contains 674 pinhole and 81 scratch objects across 285/62 positive images. No inference has run yet.
- Implemented and tested the exactly-once production-pass cache. Images are decoded before timing; the measured interval covers model preprocessing, synchronized GPU inference, model postprocessing/mask work, prediction conversion, and the frozen confidence/area cascade. Warm-up images are excluded, every measured stem is inferred once, staged timing and raw/kept predictions are written atomically, and completed or partial run directories cannot be silently overwritten. No measured inference has run yet.
- Implemented cache-only LabelMe polygon scoring without an Ultralytics/model dependency. It validates cache identity and exact manifest order, merges `inpinhole` into `pinhole`, performs class-aware maximum-IoU one-to-one matching, reconciles TP/FP/FN, writes slipped-defect rows, and reports per-class/overall recall, precision, F1, mean matched mask IoU, support, and 95% Wilson intervals. Synthetic one-to-one and reconciliation tests pass; no measured inference has run yet.
- Implemented per-run production timing analysis from the frozen cache: staged/total distribution summaries, exact counts and percentages in fixed `<25`, `25-50`, `50-100`, `100-200`, `200-500`, `500-1000`, and `>=1000 ms` brackets, pinhole-only/scratch-only/mixed/no-target grouping, Spearman relationships with defect presence/object count/prediction count/image dimensions, a Kruskal-Wallis context test, and a two-panel timing graph. All rows are reconciled to the manifest; no measured inference has run yet.
- Keep the production C# AOI application untouched.

### #22 - CLAHE 200-Per-Class Training And 500-Image Validation Completed (2026-07-21 14:02 -04:00)

- Approved the next controlled offline direction: train a focused YOLO11n segmentation model for merged `pinhole`/`inpinhole` and `scratch` using exactly 200 independent positive training images per target class, without satisfying quotas through duplicate or augmented copies.
- Reserve a deterministic, group-aware random 500-image validation holdout that is completely disjoint from training and approximately retains canonical production prevalence.
- Apply deterministic CLAHE preprocessing with `clipLimit=2.0` and an `8x8` tile grid to training and inference images to strengthen weak-scratch visibility, with conservative brightness, contrast, and gamma augmentation only during training.
- Reuse the current 1280-pixel YOLO11n-seg training family and fixed historical balanced confidence/area cascade; initialize from clean base weights so the new holdout is not exposed through an older AOI checkpoint.
- Validate all 500 images against original LabelMe polygons with class-aware one-to-one matching at `hit_iou=0.05` and report per-class and aggregate TP, FP, FN, slipped-defect identities, precision, recall, F1, strict mask evidence, and 95% confidence intervals.
- Warm up the GPU, record synchronized staged and end-to-end timing for every holdout image, and generate a reproducible 500-image timing graph plus p50, p95, maximum, and defect-context timing statistics.
- This entry records an approved experiment direction only. It does not claim that dataset preparation, training, evaluation, or timing has completed, and it does not modify the production C# AOI application.
- Recorded the approved design at `docs/superpowers/specs/2026-07-20-clahe-200-per-class-500-holdout-design.md` and the executable plan at `docs/superpowers/plans/2026-07-20-clahe-200-per-class-500-holdout.md`.
- Implementation is isolated under `aoi_yolo/clahe_200x500/` and currently covers audited LabelMe loading, deterministic group/quota selection, CLAHE preprocessing, derived YOLO data, frozen training arguments, one-to-one matching, Wilson intervals, slipped-defect rows, and timing plots with automated tests. Full canonical manifest preparation, smoke training, full training, and 500-image measured results remain pending and must not be inferred from this progress note.
- Perceptual-hash threshold review found that automatic pHash unioning is invalid for these visually homogeneous AOI frames: DCT Hamming distance `<=6` created a false 2,698-image component, distance `<=2` created a 1,405-image component, and even exact pHash equality produced components up to 128 images. The experiment therefore records pHash values and the rejected threshold evidence but freezes roles using the previously SHA-deduplicated canonical set plus conservative 25-image sequence groups.
- Frozen seed-`20260720` manifests now contain exactly 500 disjoint holdout images and 1,071 training images. Training representation is exactly 200 independent positive images for merged `pinhole` and 200 for `scratch`, with 779 additional context/background images; holdout prevalence contains 292 pinhole-positive, 75 scratch-positive, and 190 context images.
- Generated and reconciled 1,071 CLAHE training PNG/label pairs and 500 CLAHE holdout PNG/label pairs with 1,571 derived checksum rows. Ultralytics path preflight resolves both split directories from the portable dataset YAML.
- The full-membership one-epoch smoke run completed in 497.779 seconds, saved and reloaded `best.pt`/`last.pt`, and evaluated all 500 holdout images containing 1,070 target objects. Ultralytics reported about 7.0 ms model inference per image, but the smoke model's accuracy is intentionally immature and is not a final quality claim.
- Added `run_training_only.cmd` and launched the unchanged 120-epoch configuration through a detached Windows process after proving that a healthy foreground trainer was terminated only when its command-task lifecycle ended. Preserved both incomplete attempts as `runs/full_stalled_pre_epoch_20260720_2050` and `runs/full_interrupted_turn_end_20260720_2134`; neither contained a resumable checkpoint. The detached run uses the same frozen manifests/config, has exactly one verified trainer command, and is actively advancing through epoch 1.
- At epoch 23 the detached trainer stopped abruptly without a Python or CUDA traceback. Windows System Event Log recorded NVIDIA driver provider `nvlddmkm` event ID 153 at `2026-07-21 00:53:49 -04:00`, exactly matching the final log write, so this is treated as an external GPU-driver interruption rather than a model/config failure. The epoch-22 `last.pt` checkpoint was retained; the launcher now forwards command-line arguments so recovery can use the existing locked `--resume` path without changing manifests or training arguments.
- Completed all 120 epochs after the single checkpoint recovery. Approximate summed trainer time was 54,378 seconds (15 h 06 m), excluding the interruption/relaunch gap. Reloaded `best.pt` validation reported box mAP50 64.1% and mask mAP50 38.9% across all 500 holdout images and 1,070 objects.
- Completed class-aware one-to-one polygon evaluation at `hit_iou=0.05`. Raw confidence 0.05 produced TP/FP/FN `892/901/178`, recall 83.36% (95% CI 81.01-85.48%), precision 49.75% (47.44-52.06%), and F1 62.31%. The historical balanced cascade produced `537/149/533`, recall 50.19% (47.20-53.18%), precision 78.28% (75.04-81.20%), and F1 61.16%.
- Balanced per-class results were pinhole `439/24/501`, recall 46.70%, precision 94.82%, F1 62.58%; scratch `98/125/32`, recall 75.38%, precision 43.95%, F1 55.52%. Raw per-class results were pinhole `776/379/164` and scratch `116/522/14`.
- Reconciled both operating points to exactly 1,070 ground-truth objects. Slipped-defect CSVs contain exactly 178 raw FN rows and 533 balanced FN rows, with Wilson intervals present at aggregate and per-class levels.
- Recorded exactly 500 unique timing rows and generated `aoi_yolo/clahe_200x500/TIMING_500_IMAGES.png`. End-to-end timing was mean 469.71 ms, p50 207.18 ms, p95 1,365.07 ms, maximum 20,047.00 ms; model inference alone was mean 12.38 ms and p95 17.52 ms.
- The extreme timing tail comes from Python polygon matching/postprocessing on dense images (maximum 19,518.02 ms), not GPU inference. This validation evaluator must be optimized before its end-to-end timing is treated as production throughput.
- The experiment does not meet the 95-98% production target: the balanced cascade slips too many pinholes, while raw inference creates too many false positives. Full commands, confidence intervals, limitations, and artifact paths are recorded in `aoi_yolo/clahe_200x500/CLAHE_200X500_RESULTS.md`.
- Final verification passed all 19 focused tests; the production C# AOI application remained untouched.

### #21 - Extended LabelMe Test Dataset Merged And Deduplicated (2026-07-15 14:51 -04:00)

- Combined `test images extended` into the canonical `test images` dataset while preserving BMP/LabelMe JSON pairs.
- Verified all 907 same-name overlaps byte-for-byte with SHA-256 for both BMP images and JSON metadata, moved 2,177 new pairs, and removed the verified redundant source copies.
- Ran a full SHA-256 image audit and found seven additional pixel-identical pairs under different filenames.
- Because those seven pairs contained different annotation revisions, retained the copy with more labeled shapes; for equal shape counts, retained the later-numbered revision. Removed `2390`, `2797`, `2806`, `2860`, `2861`, `3403`, and `3423` with their paired JSON files.
- Final canonical dataset contains 3,077 uniquely named, content-unique BMP images and 3,077 corresponding LabelMe JSON files. The unrelated `drive_manifest.json` and `metadata_template.json` support files remain in place.
- `test images extended` retains only its non-dataset `classes.txt`; no BMP or LabelMe JSON pairs remain there.

### #20 - Thesis And Team Findings Converted Into Dataset And Validation Policy (2026-07-14 12:18 -04:00)

- Reviewed `滤光片瑕疵检测装备硕士论文.pdf` and the team summary `7.13.docx`, then converted supported findings into dataset, training, validation, and reporting instructions.
- The thesis used 1,120 annotated images across eight more-balanced defect classes and reported improved U-Net table metrics of mean precision 0.951, recall 0.907, F1 0.938, and IoU 0.583 after multi-scale input, SE attention, and multi-scale feature fusion. Its prose separately states IoU 66.9%, so the discrepancy must remain explicit.
- The thesis's 95.7% result is product-level quality-grading accuracy, not defect-level precision/recall. Its unresolved dust-versus-pinhole problem supports the current reviewed-hard-negative direction.
- The team pipeline used 3,085 full images split into 2,468 train, 308 validation, and 309 test images, but those data must not be merged until provenance, semantics, duplicates, domains, and permission are audited.
- The team's coarse-to-fine pipeline missed 85 of 921 ground-truth objects at the fine stage, inadequately classified or covered 109, and failed to place 20 objects in any fine-stage tile. Future tile experiments must therefore report candidate/GT coverage before fine-model accuracy.
- The team observed 65 mutual `pinhole`/`inpinhole` confusions, supporting the current merged core label. It also reported `splash` recall 92.57% with precision only 6.24% and only one `watermark` test object, showing why class counts and uncertainty must accompany metrics.
- The team's front/back experiment showed that `imgsz=512` loses weak and small defect detail, while edge reflections, bright background points, glue/support texture, and fragmented bright regions require explicit hard-negative coverage.
- The current source audit contains 3,409 merged `pinhole`/`inpinhole` annotations versus 543 `scratch` annotations, while individual rare classes have only 3-87 annotations. This imbalance is now treated as a primary dataset limitation rather than something that repeated augmentation can solve.
- Added a permanent policy requiring approximately balanced training and diagnostic validation, deterministic stratified group splitting, immutable manifests, duplicate checks, and a separate locked production-distribution test set.
- Deep validation must cover the complete holdout data in batches of about 300-400 images across five grouped folds. Repeated inference on the same weights and images does not increase independent statistical evidence.
- Per-class reporting must include support counts, TP/FP/FN, recall, precision, F1, strict mask metrics, 95% confidence intervals, fold variation, confusion, and size/camera/context breakdowns; under-supported classes must be marked statistically inconclusive.
- Product masks remain contextual rather than hard rejection gates. Dynamic tiles and a thesis-style compact U-Net remain controlled offline experiments, not assumed replacements for the current YOLO baseline.
- This documentation-only update does not modify datasets, model weights, training code, evaluation code, or the production C# AOI application.
- Starting with this entry, future changelog headings must include local completion time as `YYYY-MM-DD HH:MM +/-HH:MM`; entries `#1`-`#19` remain undated because their exact completion times are not reliably documented.

### #19 - Weekly Group Meeting Report Package Completed And Verified

- Completed the presentation-ready package at `weekly_group_meeting_report_2026-07-12/`: seven stage folders, seven `stage_notes.md` files, 14 original BMP backups, 14 annotated PNGs, 14 provenance rows in `source_index.csv`, and two summary PNGs.
- Added evidence-backed annotation primitives, the seven-stage manifest, deterministic material generation, presentation narrative, root navigation, and summary graphics. Event-only sources remain explicitly non-geometric; candidate polygons and existing overlays retain exact provenance.
- Added `summary.md` as the cloud/Gemini assembly entry point. Each stage note now lists only its own selected images using portable `~\weekly_group_meeting_report_2026-07-12\<stage>\annotated\<image>.png` paths and no Markdown image syntax.
- Simplified stage one to the formal nine-class YOLO11n-seg long run (147 recorded epochs, box mAP50 about 61.9%, mask mAP50 about 47.5%); the five-epoch Smoke Test is excluded from presentation content.
- Standardized all seven presentation headings in Chinese. Presentation-facing uses of “Option A” were replaced with “stage balanced baseline” (`阶段性平衡基准`) and “class-confidence/area cascade” (`类别置信度与面积级联筛选`); real source paths containing `option_a` remain unchanged for provenance.
- Corrected named edge-strip overlays so they are composited at their encoded crop origin on the full-resolution BMP, regenerated the affected `0073.png`, added regression coverage, and documented report and baked-review colors in `COLOR_KEY.md`.
- Generation command: `.\.venv_yolo\Scripts\python.exe weekly_group_meeting_report_2026-07-12\generate_report_materials.py --manifest weekly_group_meeting_report_2026-07-12\report_manifest.json --output weekly_group_meeting_report_2026-07-12`.
- Final QA visually inspected at least one annotated image from every stage, including full-image and edge/context examples; Chinese text, borders, semantic contours, and mask opacity were readable and did not obscure the relevant evidence.
- Cross-checked all headline metrics against the authoritative comparison reports, including Option A balanced (83.56% recall / 83.22% precision), multi-scale threshold 0.10 (83.29% / 83.75%), HALCON selected rules (83.15% / 83.61%), product-mask gating (52.33% / 82.86%), and DAMNet threshold 0.01 (83.84% / 64.42%).
- Final focused verification passed all 21 tests. The production C# AOI application and original experiment files were not modified, and no Git operations were performed.

### #18 - HALCON-Style Two-Stage Filter Baseline Added

- Added `aoi_yolo/hybrid_models/option_b_crop_classifier/halcon_style_filter.py` as an offline two-stage detector/filter baseline.
- Stage 1 uses existing YOLO Option A candidates from `aoi_yolo/hybrid_models/option_b_crop_classifier/multiscale_geometry_filter/train_candidates.csv` and `val_candidates.csv`; it does not re-run YOLO by default.
- Stage 2 computes OpenCV/Numpy equivalents of HALCON-style measurements from original BMP pixels and predicted masks:
  - mask area, bbox dimensions, bbox long/short side, elongation, fill ratio;
  - contour perimeter, compactness, convexity;
  - local ring mean/std intensity around the candidate;
  - absolute, dark, bright, and z-normalized local contrast;
  - Sobel gradient magnitude and gradient lift versus the local ring.
- The script grid-searches interpretable class-specific rules on train candidates, then evaluates on the validation split with the same one-to-one JSON polygon matching.
- Default recall-preserving run output:
  - Output folder: `aoi_yolo/hybrid_models/halcon_style_filter`
  - Option A baseline: recall 83.56%, precision 83.22%, 123 false positives.
  - Selected HALCON-style rules: recall 83.15%, precision 83.61%, 119 false positives.
  - Relaxed HALCON-style rules: recall 83.29%, precision 83.40%, 121 false positives.
- Precision-leaning run with `--min-train-row-recall 0.97`:
  - Output folder: `aoi_yolo/hybrid_models/halcon_style_filter_recall97`
  - Selected HALCON-style rules: recall 82.47%, precision 83.96%, 115 false positives.
  - Relaxed rules match the default relaxed result: recall 83.29%, precision 83.40%, 121 false positives.
- Interpretation:
  - The deterministic HALCON-style measurements can remove a small number of pinhole false positives.
  - They do not materially improve scratch precision; scratch selected rules effectively reduce one prediction and one hit, leaving 22 scratch false positives.
  - Fixed rules are too blunt for the current target. The measurement features are still useful, but should feed a learned second-stage model or a reviewed hard-negative workflow.
- Updated `aoi_yolo/hybrid_models/option_b_crop_classifier/README.md` with the script entry point and results.

### #15 - DAMNet-Style Pyramid Baseline Added

- Started a DAMNet-style defect segmentation sidecar under `aoi_yolo/damnet/` to test whether a dense multi-scale pyramid model can beat the current YOLO segmentation baseline.
- Added `aoi_yolo/damnet/train_damnet.py`:
  - Reads the existing `dataset_core2_bg_v2` train/val split while rasterizing masks directly from the original BMP/LabelMe JSON files in `test images`.
  - Merges `inpinhole` into `pinhole` by default and keeps `scratch` as the second class.
  - Uses a compact encoder/decoder with dense pyramid blocks containing parallel `1x1`, `3x3`, `5x5`, and dilated `3x3` branches.
  - Trains with BCE + Dice loss and class-balanced positive weights.
  - Converts validation probability maps into connected components and evaluates with AOI-style one-to-one mask matching at `hit_iou=0.05`.
  - Writes `best.pt`, `last.pt`, `metrics.csv`, `eval_summary.csv`, `eval_events.csv`, and sample overlays.
  - Supports `--limit-train` and `--limit-val` for quick smoke tests.
- Added `aoi_yolo/damnet/README.md` with smoke-test and longer benchmark commands.
- Smoke test passed with `--limit-train 8 --limit-val 4 --imgsz 256 --base-channels 8`: the script completed one epoch, saved checkpoints, reloaded `best.pt`, wrote eval CSVs, and generated sample overlays. The resulting 0 recall/precision is expected from the tiny one-epoch smoke and is not a meaningful model-quality result.
- This is still an offline experiment only. Do not integrate DAMNet into the C# AOI app unless it beats YOLO on the same JSON validation split and has acceptable latency.

### #16 - DAMNet Benchmark Run Compared Against YOLO

- Ran the DAMNet-style pyramid benchmark:
  - Run: `aoi_yolo/damnet/runs/damnet_pyramid_e80_i640`
  - Target command: 80 epochs, `imgsz=640`, `batch=2`, `base_channels=24`, CUDA.
  - The foreground command hit the 2-hour command window after 23 completed epochs, but wrote usable `best.pt` and `last.pt`.
  - Best validation loss in the partial run was around epoch 21.
- Added `--eval-only` and `--weights` support to `aoi_yolo/damnet/train_damnet.py` so saved DAMNet checkpoints can be evaluated without retraining.
- Evaluated DAMNet `best.pt` with connected-component extraction from semantic probability maps:
  - Threshold 0.35: overall recall 73.01%, precision 76.14%.
  - Threshold 0.20: overall recall 76.44%, precision 73.91%.
  - Threshold 0.10: overall recall 78.63%, precision 71.57%.
  - Threshold 0.05: overall recall 80.14%, precision 69.64%.
  - Threshold 0.01: overall recall 83.84%, precision 64.42%.
- Re-ran the YOLO Option A balanced reference on the same validation split and JSON ground truth:
  - Weights: `aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt`
  - Settings: `imgsz=1280`, `conf=0.05`, `hit_iou=0.05`, `inpinhole=pinhole`, `pinhole conf >= 0.30`, `scratch conf >= 0.20`, `pinhole area >= 150`, `scratch area >= 300`.
  - Result: overall recall 83.56%, precision 83.22%; pinhole recall/precision 84.34%/84.07%; scratch recall/precision 78.57%/77.78%.
- Conclusion: this DAMNet baseline does not beat YOLO. Its closest recall-matched threshold, 0.01, reaches 83.84% recall but only 64.42% precision, and scratch recall remains poor at 41.84%.
- Added `aoi_yolo/damnet/DAMNET_VS_YOLO_COMPARISON.md` with the threshold sweep and conclusion.
- If DAMNet is revisited, change the setup rather than simply continuing the same run: try higher resolution, scratch oversampling/class weighting, thin-boundary losses, or tile/patch training with full-image reconstruction.

### #17 - DAMNet Long High-Resolution Continuation

- Updated `aoi_yolo/damnet/train_damnet.py` for longer DAMNet experiments:
  - Added `--resume` support for continuing from a DAMNet checkpoint.
  - New checkpoints now include optimizer and scheduler state.
  - Added `--oversample-label` and `--oversample-factor` for repeating scratch-heavy training images.
  - Added `--class-loss-weight` for class-specific BCE/Dice weighting.
- Smoke-tested the new training controls with `--limit-train 8 --limit-val 4 --imgsz 256 --base-channels 8 --oversample-label scratch --oversample-factor 3 --class-loss-weight scratch=3`; training/eval completed.
- Ran a longer high-resolution DAMNet continuation:
  - Run: `aoi_yolo/damnet/runs/damnet_pyramid_resume_e120_i1024_scratchw4`
  - Started from `aoi_yolo/damnet/runs/damnet_pyramid_e80_i640/best.pt`
  - Config: `imgsz=1024`, `batch=1`, `base_channels=24`, `lr=0.0003`, scratch oversampling factor 4, scratch loss weight 4.
  - First 2-hour foreground chunk completed epochs 22-31; best validation loss improved to about 0.3698 at epoch 29.
  - Second 2-hour foreground chunk completed through epoch 40; it did not beat the epoch-29 best checkpoint.
- Evaluated the 1024 scratch-weighted DAMNet checkpoints:
  - `best.pt`, threshold 0.35: recall 57.67%, precision 83.37%.
  - `best.pt`, threshold 0.10: recall 64.79%, precision 81.98%.
  - `best.pt`, threshold 0.03: recall 67.53%, precision 79.64%.
  - `best.pt`, threshold 0.01: recall 70.27%, precision 78.44%.
  - `last.pt`, threshold 0.01: recall 72.47%, precision 64.43%; scratch recall 79.59% but scratch precision only 27.86%.
  - `last.pt`, class thresholds `pinhole=0.01`, `scratch=0.10`: recall 72.05%, precision 71.27%.
- Conclusion: running DAMNet longer at higher resolution with scratch weighting still does not approach the target or beat YOLO. The later checkpoint can lift scratch recall, but only with too many scratch false positives, and pinhole recall remains too low.
- Updated `aoi_yolo/damnet/DAMNET_VS_YOLO_COMPARISON.md` with the long-run results.
- Recommendation: stop spending time on this exact DAMNet setup. A further DAMNet-style attempt needs a different formulation, likely native-detail tile/patch training, explicit hard-negative mining, boundary/thin-scratch losses, or a two-stage proposal/filter design.

### #14 - Multi-Scale Geometry Defect/Dust Filter Implemented

- Reworked `aoi_yolo/hybrid_models/option_b_crop_classifier/train_and_eval_crop_filter.py` into a multi-scale geometry candidate filter.
- Pipeline now follows:
  `YOLO conf=0.05 -> Option A prefilter -> 1x/3x/5x crop pyramid + geometry features -> binary real-defect filter -> threshold sweep`.
- Option A prefilter defaults:
  - `pinhole conf >= 0.30`
  - `scratch conf >= 0.20`
  - `pinhole area >= 150`
  - `scratch area >= 300`
- Each candidate now saves aligned `1x`, `3x`, and `5x` crops, resized to 128x128 with black padding outside the image.
- Added geometry features:
  - YOLO confidence
  - mask area
  - bbox width/height
  - aspect ratio
  - elongation
  - fill ratio
  - contour compactness
  - JSON edge-distance feature
  - product-mask overlap feature from `aoi_yolo/product_mask/dataset_product_mask/masks`
- Model is a shared three-branch CNN encoder plus normalized geometry features, trained with weighted BCE.
- Smoke test passed with `--limit 4`, one epoch, GPU classifier path, crop generation, probability CSV, and threshold-sweep evaluation.
- Full run completed:
  - Output: `aoi_yolo/hybrid_models/option_b_crop_classifier/multiscale_geometry_filter`
  - Candidates after Option A prefilter: 3170 train, 733 val.
  - Train split: 2803 real-defect candidates, 367 dust/benign-like candidates.
  - Val split: 620 real-defect candidates, 113 dust/benign-like candidates.
  - Checkpoint: `multiscale_geometry_filter.pt`
  - Probability CSV: `val_candidates_with_filter_probs.csv`
  - Comparison report: `MULTISCALE_GEOMETRY_COMPARISON.md`
- Validation threshold sweep:
  - Threshold 0.10: recall 83.29%, precision 83.75%, F1 83.52%, 118 false positives.
  - Threshold 0.20: recall 82.05%, precision 85.33%, F1 83.66%, 103 false positives.
  - Threshold 0.50: recall 72.88%, precision 90.78%, F1 80.85%, 54 false positives.
  - Threshold 0.80: recall 50.68%, precision 97.63%, F1 66.73%, 9 false positives.
- Interpretation:
  - The formal recall-biased/F1 selection rule chose threshold 0.20.
  - For production-style recall preservation, threshold 0.10 is probably better: it is only 0.27 recall points below Option A balanced and improves precision by 0.53 points while reducing false positives from 123 to 118.
  - This is a small but real improvement over Option A balanced, not a large breakthrough.
- Next recommendation: inspect the remaining false positives/false negatives from the threshold 0.10 and 0.20 probability CSVs, then manually tag a small reviewed dust/benign-speck set to strengthen the negative class.

### #13 - Product Mask Model Trained And Dust-Like Filter Compared

- Converted product-face pseudo-labels into YOLO segmentation format:
  - Source annotations: `aoi_yolo/product_mask/dataset_product_mask/annotations`
  - Source images: `test images`
  - YOLO dataset: `aoi_yolo/product_mask/dataset_product_mask_yolo`
  - Split: 726 train images, 181 val images, class `product_face_auto`.
- Trained lightweight product-face mask model:
  - Run: `aoi_yolo/product_mask/runs/product_face_yolo11n_seg_e25_i640_w0`
  - Weights: `aoi_yolo/product_mask/runs/product_face_yolo11n_seg_e25_i640_w0/weights/best.pt`
  - Used `workers=0` because Windows multiprocessing fails when launching Ultralytics from inline stdin.
  - Validation against pseudo-label masks: box precision/recall/mAP50 about 0.960/0.972/0.975; mask precision/recall/mAP50 about 0.943/0.956/0.964.
  - Validation speed about 0.5 ms preprocess, 3.8 ms inference, 1.2 ms postprocess per image.
- Added product-mask gating support to `aoi_yolo/evaluate_json_polygons.py`:
  - New `--product-mask-dir`
  - New `--product-mask-min-overlap`
  - Keeps predictions only if enough predicted mask pixels overlap a product-face mask.
- Compared current defect filtering methods on `dataset_core2_bg_v2` validation split, 181 images, `hit_iou=0.05`, `imgsz=1280`, `inpinhole=pinhole`:
  - Option A balanced: recall 83.56%, precision 83.22%.
  - Product-mask gated with Option A and 1%-10% mask overlap: recall 52.33%, precision 82.86%.
  - Product-mask gated with 75% overlap: recall 52.05%, precision 82.97%.
  - Conclusion: current product-face masks are too aggressive for direct defect gating and remove too many true defects.
- Attempted the dedicated dust hard-negative miner in `aoi_yolo/hard_negatives_dust/prepare_dust_filter_dataset.py`; it exceeded the 2-hour command window and produced no `candidate_filter_dataset/manifest.csv`.
- Trained a dust-like / false-positive crop filter using the existing Option B candidate-filter script:
  - Run/output: `aoi_yolo/hybrid_models/option_b_crop_classifier/current_dust_like_filter`
  - Generated 5289 train candidates and 1258 validation candidates.
  - Filter threshold 0.20: recall 95.48%, precision 55.63%.
  - Filter threshold 0.50: recall 85.75%, precision 60.95%.
  - Filter threshold 0.70: recall 20.27%, precision 77.89%.
  - Conclusion: this automatic crop filter still does not beat Option A balanced.
- Reconfirmed precision-focused Option A operating points:
  - High-confidence: `pinhole conf >= 0.50`, `scratch conf >= 0.40`, same area filters; recall 69.59%, precision 93.21%.
  - Precision-first: `pinhole conf >= 0.60`, `scratch conf >= 0.45`, same area filters; recall 60.27%, precision 95.86%.
- Added `aoi_yolo/product_mask/MASK_AND_DUST_FILTER_COMPARISON.md` with the comparison table and next experiment notes.
- Next recommendation: do not use the current binary product-face mask as a hard defect gate. Improve context labels into `interior_face`, `edge_band`, `between_edges`, and `background/void`, and build a reviewed dust/benign-speck crop dataset before training another filter.

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

