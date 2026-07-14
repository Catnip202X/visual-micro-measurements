# AGENTS Research, Dataset Sanitation, and Validation Policy Design

**Date:** 2026-07-14

## Purpose

Update `AGENTS.md` so future work is guided by three evidence sources:

1. the existing AOI YOLO experiment history;
2. the thesis `滤光片瑕疵检测装备硕士论文.pdf`;
3. the team-meeting summary `7.13.docx`.

The update must convert those sources into operational instructions for dataset preparation, training, validation, and reporting. It must not treat external results as if they were produced by the current sidecar or compare incompatible metrics directly.

## Scope

The implementation will modify only `AGENTS.md`.

It will:

- replace the outdated `Immediate Next Steps` with a current prioritized sequence;
- add a permanent dataset-sanitation and statistical-validation policy;
- add changelog entry `#20` documenting the paper and team-meeting findings and why they change the project direction;
- timestamp changelog `#20` and require timestamps on all future changelog entries, without inventing dates for historical entries `#1`-`#19`;
- preserve the production C# application and existing experiment outputs;
- keep the thesis PDF and meeting DOCX as local evidence sources unless the user separately authorizes committing those binary files.

No training, dataset regeneration, model evaluation, production integration, or unrelated report modification is part of this change.

## Evidence To Preserve

### Current project evidence

- The current balanced AOI baseline remains recall 83.56% and precision 83.22% under one-to-one LabelMe polygon matching at `hit_iou=0.05`.
- The current expanded source has 3,409 merged `pinhole`/`inpinhole` annotations and 543 `scratch` annotations. Rare classes have only 3-87 annotations each, so broad-class training and evaluation are not statistically stable.
- Product-mask hard gating reduced recall to about 52%, so the mask should be treated as context rather than a binary rejection gate.
- Multi-scale geometry and HALCON-style candidate filters produced only small gains. The current DAMNet formulation did not beat YOLO.

### Thesis evidence

- The thesis used 1,120 annotated images across eight defect classes with a substantially more balanced class distribution than the current broad dataset.
- Its improved U-Net combines multi-scale input, SE attention, and multi-scale feature fusion.
- Table 3-7 reports mean precision 0.951, recall 0.907, F1 0.938, and IoU 0.583. The surrounding prose states IoU 66.9%, so the table/prose discrepancy must be noted rather than silently resolved.
- The reported 95.7% result is product-level quality-grading accuracy, not defect-level precision or recall.
- The thesis itself identifies dust-versus-pinhole confusion as unfinished work.
- Its product-edge fitting, defect localization, and quality-grading logic are relevant to later integration, but they do not justify immediate production wiring.

### Team-meeting evidence

- The other team pipeline used 3,085 full images split into 2,468 train, 308 validation, and 309 test images. These data must not be mixed into the current dataset until provenance, label semantics, duplicates, camera domains, and authorization are audited.
- Its coarse-to-fine method dynamically selects suspicious regions, guarantees at least four tiles per image, applies an `inpinhole`-strengthened fine model, then uses class-specific thresholds and slight `inswab` mask expansion.
- Of 921 ground-truth objects, 85 were completely missed by the fine stage, 109 were misclassified or inadequately covered, and 20 were not covered by any fine-stage tile.
- `pinhole` and `inpinhole` were mutually confused 65 times, supporting the current merged core label unless production requirements demand separation.
- `splash` reached recall 92.57% but precision only 6.24%; `watermark` had only one test object. These results demonstrate why per-class sample counts and uncertainty must accompany headline metrics.
- The front/back experiment shows that `imgsz=512` loses weak and small details and that edge reflection, background bright points, glue/support texture, and fragmented bright regions are important hard negatives.

## AGENTS.md Structure

### 1. Revised Immediate Next Steps

Replace the current mask-first sequence with this priority order:

1. version and audit the dataset before more training;
2. create stratified, acquisition-group-aware manifests;
3. collect and review hard negatives, especially dust, benign specks, reflections, fixture/glue texture, and background bright points;
4. increase independent representation for `scratch` and any class intended for separate production decisions;
5. run a controlled full-image versus coverage-audited tile experiment at native detail;
6. consider a thesis-style compact U-Net benchmark only after dataset corrections, using the same ground truth and AOI evaluation as YOLO;
7. perform five-fold grouped deep validation, followed by a single locked production-distribution test;
8. keep the C# production application unchanged until accuracy and latency gates pass.

### 2. Dataset Sanitation And Statistical Validation Policy

Add a durable policy covering the following requirements.

#### Representation

- Count both annotation instances and independent positive images per class.
- Make the training set and diagnostic validation folds approximately class-balanced.
- Do not balance solely by duplicating or augmenting the same rare images; representation must include independent cameras, trays, capture sessions, illumination conditions, product regions, defect sizes, and appearances.
- Merge or defer classes that cannot support independent learning and validation.
- Preserve the current `pinhole` + `inpinhole` merge unless a production decision explicitly requires separation and sufficient independent data exist.
- Maintain a separate locked test set that reflects production prevalence. Balanced diagnostic validation and production-prior testing answer different questions and both are required.

#### Split and shuffle rules

- Shuffle deterministically with a recorded seed and immutable manifests.
- Use stratified group splitting by class and acquisition group.
- Keep images from the same product, tray, sequence, camera session, or near-duplicate capture group in a single split.
- Do not reshuffle completed experiments. Any rebalance creates a new versioned dataset and new manifests.
- Run duplicate and near-duplicate checks before finalizing splits.

#### Deep validation

- Evaluate the complete holdout population in operational batches of about 300-400 images and aggregate the batches before calculating final metrics.
- Do not count five repeated inference passes on identical weights and data as five independent validation runs.
- Use five grouped cross-validation folds so distinct acquisition groups form the holdout data in each run and every acquisition group serves as holdout data once.
- Repeated training seeds may be added to measure optimization variance, but they do not replace the five grouped holdout folds and do not create additional independent validation images.
- Keep a final locked production-distribution test set untouched during threshold selection and model comparison, then evaluate it once after model selection.
- Match predictions one-to-one against original LabelMe JSON polygons.
- Preserve both AOI event detection at `hit_iou=0.05` and stricter localization metrics such as mask IoU and mask mAP50.

#### Statistical reporting

For every class, report:

- annotated objects and independent source images;
- true positives, false positives, and false negatives;
- recall, precision, F1, AP50/mask AP50, and mask IoU where applicable;
- 95% confidence intervals;
- mean and variation across the five grouped folds, plus seed variation only when repeated-seed training is separately performed;
- confusion with other classes;
- breakdowns by defect size, camera, product region, and edge/interior context.

Mark results statistically inconclusive when holdout support is too small. Operational batches are processing units, not independent experiments.

#### Tile and multi-stage evaluation

- A coarse-to-fine experiment must publish candidate/GT coverage before fine-model accuracy.
- Guaranteeing a minimum tile count is not sufficient; every missed ground-truth object must be categorized as coarse-stage coverage failure, fine-stage miss, class confusion, or mask-coverage failure.
- Compare the tile system with the full-image baseline on the same manifests, JSON ground truth, thresholds, and one-to-one matching.

### 3. Changelog Entry #20

Use the heading format `### #20 - <title> (YYYY-MM-DD HH:MM +/-HH:MM)`, recording the local workspace time with its current numeric UTC offset. All future changelog headings must follow the same timestamp format. Do not retrofit `#1`-`#19` because exact completion times are not reliably documented.

The changelog will record:

- the two evidence files reviewed;
- the relevant thesis results and metric-compatibility warning;
- the team method and failure counts;
- the resulting dataset sanitation, grouped splitting, deep-validation, and reporting rules;
- the decision to keep product masks as context, preserve the `pinhole` merge, and treat tiles or thesis-style U-Net as controlled experiments rather than assumed replacements;
- that this documentation-only change does not modify training data, model weights, or the production C# application.

## Git Strategy

- Commit this design specification separately.
- After specification review, create an implementation plan using the Superpowers writing-plans workflow.
- Stage only the approved planning document and `AGENTS.md` for the implementation commit.
- Do not stage pre-existing model, report, code, dataset, PDF, DOCX, or temporary-file changes.
- Push the resulting commits to the current branch only after verification.

## Verification And Acceptance Criteria

The change is complete when:

- `AGENTS.md` contains a current `Immediate Next Steps` sequence;
- the permanent sanitation/validation policy contains all four user requirements with the five-run correction;
- changelog `#20` accurately distinguishes current, thesis, and team results;
- changelog `#20` has a local ISO-style timestamp with a numeric UTC offset, and the file instructs future entries to do the same;
- no paper metric is presented as directly comparable to the current AOI event metrics;
- no repeated inference pass is described as additional independent validation evidence;
- no unrelated file is staged or committed;
- Markdown headings, numbering, terminology, paths, and headline metrics pass a focused diff review and `git diff --check`;
- the approved commits are pushed to the configured Git remote.
