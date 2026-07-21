# Historical-Best Model Two-Run Deep Evaluation Design

## Status

Approved on 2026-07-21. This design covers offline evaluation only and must not modify the production C# AOI application.

## Objective

Deeply evaluate the historically best segmentation model, previously measured at 83.56% recall and 83.22% precision, on two new mutually exclusive randomized 500-image datasets. Measure production-path image-processing latency once per image, then reuse the cached predictions for all ground-truth scoring. Determine per-defect performance, latency distributions, relationships between latency and defect composition, and whether the two independent runs differ statistically.

## Frozen model and operating point

- Weights: `aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt`.
- Image size: 1280.
- Raw inference confidence: 0.05.
- AOI matching IoU: 0.05.
- Merge `inpinhole` into `pinhole`.
- Historical production-style cascade:
  - pinhole confidence at least 0.30 and predicted mask area at least 150 pixels;
  - scratch confidence at least 0.20 and predicted mask area at least 300 pixels.
- The model, thresholds, mask rules, and matching policy must remain identical in both runs.

## Dataset protocol

1. Audit the historical model dataset and exclude every train and validation member from the candidate pool.
2. Exclude exact duplicates using SHA-256 and keep sequence/acquisition-related captures in one group.
3. Select two mutually exclusive group-aware random samples of exactly 500 images each from the remaining canonical LabelMe BMP/JSON pairs.
4. Use fixed, recorded seeds and immutable manifests. Each manifest records image/JSON paths, hashes, group, dimensions, source labels, merged target labels, annotated-object counts, and defect context.
5. Preserve the available production prevalence; do not balance classes or require positive images.
6. Verify that each run has 500 unique images, the runs share zero images or groups, and neither run overlaps the historical training/validation membership.

If fewer than 1,000 eligible independent images remain after exclusions, stop and report the audited shortfall rather than weakening leakage controls.

## Exactly-once production timing pass

Each run processes all 500 images continuously. Disk decode is performed before the timed region. The timer starts immediately before the decoded image is passed to the model and ends only after:

1. Ultralytics/model preprocessing;
2. GPU inference with synchronization;
3. model NMS and mask postprocessing;
4. conversion to normalized prediction records;
5. the frozen class-confidence and mask-area cascade.

The timed loop must not open LabelMe JSON files, rasterize ground truth, calculate IoU, match polygons, or write per-image charts. Warm-up images are executed before measurement and are not counted among the 500 timing rows.

Inference occurs exactly once for each measured image. The pass caches predictions and stage timings in an immutable per-run artifact. The later accuracy pass must read this cache and must never invoke the model again.

Recorded timing fields include model preprocess, inference, model postprocess, AOI cascade/record conversion, total production-path latency, image dimensions, raw/kept prediction counts, and GPU/device metadata.

## Offline scoring pass

After a run's complete production timing pass, a separate loop reads its prediction cache and original LabelMe JSON polygons. It performs class-aware one-to-one matching at IoU 0.05. Ground-truth work is outside all production timing measurements.

For every target defect and overall, report:

- annotated objects and independent positive images;
- TP, FP, and FN;
- recall, precision, and F1;
- 95% Wilson confidence intervals for recall and precision;
- mean matched-mask IoU;
- slipped-defect rows containing image, class, source label, polygon identity, area, and best available IoU/confidence evidence.

## Timing summaries and brackets

For total production-path latency and each stage, report count, mean, standard deviation, median, p90, p95, p99, maximum, and bootstrap 95% confidence intervals for mean and median.

Use fixed total-latency brackets so the runs are directly comparable:

- under 25 ms;
- 25 to under 50 ms;
- 50 to under 100 ms;
- 100 to under 200 ms;
- 200 to under 500 ms;
- 500 to under 1,000 ms;
- 1,000 ms or more.

For every bracket, report the number and percentage of all 500 images plus composition by ground-truth category and predicted-object count.

Group timing by:

- pinhole-only;
- scratch-only;
- mixed pinhole and scratch;
- no-target/context.

Report category counts and timing summaries. Analyze timing relationships with ground-truth defect category, ground-truth object count, predicted object count, kept prediction count, image dimensions, and total pixels.

## Statistical analysis

Within each run:

- use Spearman correlation for latency versus object counts and image size;
- use Kruskal-Wallis for latency across the four defect categories;
- if significant, use corrected pairwise Mann-Whitney comparisons and Cliff's delta;
- report effect sizes and confidence intervals, not p-values alone.

Between runs:

- compare total latency distributions with Mann-Whitney U and two-sample Kolmogorov-Smirnov tests;
- report Cliff's delta and bootstrap differences in mean and median;
- compare bracket distributions with a chi-square or exact alternative when expected cells are too small;
- compare per-defect recall and precision using effect estimates and 95% confidence intervals;
- compare defect/image composition first so prevalence changes are not misreported as model instability;
- apply Holm correction to families of category-level comparisons.

The two runs are separate independent samples, not repeated inference on the same images. Statistical significance alone does not imply operational importance.

## Outputs

Create an isolated evaluation package for changelog #23 containing:

- frozen run-1 and run-2 manifests and audit summaries;
- exactly one prediction/timing cache per run;
- exactly 500 timing rows per run;
- per-run event, summary, and slipped-defect files;
- per-run latency histogram, empirical CDF, timing-bracket chart, and defect-category timing plot;
- a combined run-to-run comparison table and plots;
- `HISTORICAL_BEST_DEEP_EVALUATION_RESULTS.md` with commands, provenance, metrics, confidence intervals, tests, effect sizes, limitations, and verification evidence.

Changelog #23 begins as a direction entry and is finalized only after both runs and all reconciliation checks complete.

## Verification and failure handling

- Unit-test manifest exclusions, mutual exclusivity, timing-bracket boundaries, cached-prediction round trips, one-to-one matching, Wilson intervals, correlations, and run-comparison calculations.
- Fail if inference is called during the scoring pass.
- Fail if a run does not contain exactly 500 unique timing rows.
- Fail if TP + FN does not equal annotated target objects, or slipped-defect row count does not equal FN.
- Fail if either manifest overlaps historical model membership or the other run's images/groups.
- Preserve partial timing caches for diagnosis, but do not publish partial-run statistics.
- Record GPU-driver interruptions and rerun only the incomplete run from its beginning; never mix timing rows from before and after a device interruption.

## Explicit non-goals

- No retraining or threshold tuning.
- No CLAHE unless it was part of the historical model's original production path; this evaluation preserves that model's established input strategy.
- No production C# integration or machine-control changes.
- No inference rerun for accuracy scoring.
