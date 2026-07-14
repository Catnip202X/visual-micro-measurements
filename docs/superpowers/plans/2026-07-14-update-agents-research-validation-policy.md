# AOI Research and Validation Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `AGENTS.md` with evidence-backed dataset sanitation, grouped splitting, deep validation, per-class reporting, current next steps, and timestamped changelog guidance.

**Architecture:** This is an isolated documentation change. Replace the obsolete mask-first next-step list, add one durable project policy section, and prepend one timestamped changelog entry that distinguishes current-project, thesis, and team-meeting evidence.

**Tech Stack:** Markdown, PowerShell verification, Git.

## Global Constraints

- Modify only `AGENTS.md`; create and commit this implementation plan alongside it.
- Preserve all existing user changes in `AGENTS.md`, including changelog `#19`.
- Do not stage the thesis PDF, `7.13.docx`, temporary renders, report materials, model outputs, datasets, or unrelated code changes.
- Keep the production C# AOI application untouched.
- Keep the current AOI baseline at recall 83.56% and precision 83.22% under one-to-one LabelMe matching at `hit_iou=0.05`.
- Do not compare the thesis's product-level 95.7% grading result directly with defect-level recall or precision.
- Require five grouped cross-validation folds; repeated inference on identical weights and images is not independent evidence.
- Process deep-validation holdouts in batches of about 300-400 images, then aggregate all batches before final statistics.
- Timestamp changelog `#20` and every future entry as `YYYY-MM-DD HH:MM +/-HH:MM`; do not invent timestamps for `#1`-`#19`.

---

### Task 1: Update AGENTS Guidance and Record the Change

**Files:**
- Modify: `AGENTS.md:9-120`
- Create: `docs/superpowers/plans/2026-07-14-update-agents-research-validation-policy.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-14-agents-research-and-validation-policy-design.md`, `滤光片瑕疵检测装备硕士论文.pdf`, `7.13.docx`, and the existing `AGENTS.md` experiment history.
- Produces: Future-agent instructions for dataset versioning, representation, grouped shuffling, validation, reporting, tile evaluation, and timestamped changelogs.

- [ ] **Step 1: Verify the target structure before editing**

Run:

```powershell
rg -n "^## Current Objectives$|^## Immediate Next Steps$|^## Changelog$|^### #19" AGENTS.md
```

Expected: all four existing headings are found, and no `Dataset Sanitation And Statistical Validation Policy` or `#20` heading exists.

- [ ] **Step 2: Add the permanent dataset-sanitation and validation policy**

Insert a new `## Dataset Sanitation And Statistical Validation Policy` section between `Current Objectives` and `Immediate Next Steps`. It must contain these subsections and requirements:

```markdown
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
```

- [ ] **Step 3: Replace `Immediate Next Steps` with the current priority order**

Replace only the contents between `## Immediate Next Steps` and `## Changelog` with:

```markdown
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
```

- [ ] **Step 4: Prepend timestamped changelog entry `#20`**

Insert immediately below `## Changelog`:

```markdown
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
```

- [ ] **Step 5: Verify content and Markdown structure**

Run:

```powershell
rg -n "^## Dataset Sanitation And Statistical Validation Policy$|^### Representation And Dataset Versions$|^### Split And Shuffle Rules$|^### Deep Validation Protocol$|^### Per-Class Statistical Reporting$|^### Tile And Multi-Stage Evaluation$|^### Changelog Timestamp Rule$|^## Immediate Next Steps$|^### #20 .*\(2026-07-14 12:18 -04:00\)$" AGENTS.md
rg -n "five repeated inference|five grouped|300-400|95% confidence|production-distribution|pinhole.*inpinhole" AGENTS.md
git diff --check -- AGENTS.md docs/superpowers/plans/2026-07-14-update-agents-research-validation-policy.md
```

Expected: every required heading and policy phrase is found; `git diff --check` exits successfully with no whitespace errors.

- [ ] **Step 6: Review and stage only the authorized files**

Run:

```powershell
git diff -- AGENTS.md docs/superpowers/plans/2026-07-14-update-agents-research-validation-policy.md
git add -- AGENTS.md docs/superpowers/plans/2026-07-14-update-agents-research-validation-policy.md
git diff --cached --check
git diff --cached --name-only
```

Expected staged paths, and no others:

```text
AGENTS.md
docs/superpowers/plans/2026-07-14-update-agents-research-validation-policy.md
```

- [ ] **Step 7: Commit and synchronize `main`**

Run:

```powershell
git commit -m "docs: define AOI dataset validation workflow"
git push origin main
git status --short --branch
```

Expected: commit succeeds, the push updates `origin/main`, and the branch is no longer ahead. Pre-existing unrelated working-tree changes remain unstaged and unchanged.
