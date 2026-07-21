# Historical-Best Two-Run Deep Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the historical-best AOI segmentation model on two mutually exclusive leakage-free 500-image samples using exactly one production-timed inference per image, then score cached predictions and statistically compare accuracy and timing.

**Architecture:** A dedicated `aoi_yolo/historical_best_deep_eval` package freezes model/rules and manifests, runs a production-only timing pass that writes compressed prediction caches, and performs LabelMe matching afterward without importing or calling the model. Per-run analysis feeds a separate comparison/report layer.

**Tech Stack:** Python 3.12, Ultralytics YOLO, PyTorch/CUDA, OpenCV, NumPy, SciPy, Matplotlib, CSV/JSON/gzip, pytest.

## Global Constraints

- Fixed weights: `aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt`.
- Use `imgsz=1280`, raw confidence `0.05`, hit IoU `0.05`, and merge `inpinhole` into `pinhole`.
- Historical cascade remains pinhole `0.30 / 150 px` and scratch `0.20 / 300 px`.
- Create two mutually exclusive group-aware 500-image manifests with no historical train/validation overlap.
- Decode before timing; time model preprocessing through prediction conversion/cascade. Exclude disk I/O and all ground-truth work.
- Invoke inference exactly once per measured image. Accuracy scoring reads cached predictions only.
- Restart an entire run after a device interruption; never combine timing fragments.
- Keep the production C# application untouched.

---

### Task 1: Frozen configuration and experiment identity

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/__init__.py`
- Create: `aoi_yolo/historical_best_deep_eval/config.yaml`
- Create: `aoi_yolo/historical_best_deep_eval/config.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `ThresholdRule`, `ExperimentConfig`, `load_config(path) -> ExperimentConfig`, and `experiment_identity(config, weights) -> dict`.

- [ ] **Step 1: Write failing tests**

```python
def test_frozen_operating_point(config):
    assert config.imgsz == 1280
    assert config.rules["pinhole"] == ThresholdRule(.30, 150)
    assert config.rules["scratch"] == ThresholdRule(.20, 300)
    assert config.run_seeds == (2026072101, 2026072102)
    assert config.timing_brackets_ms == (0, 25, 50, 100, 200, 500, 1000, float("inf"))
```

- [ ] **Step 2: Verify failure**

Run: `.\.venv_yolo\Scripts\python.exe -m pytest aoi_yolo\historical_best_deep_eval\tests\test_config.py -q`

Expected: import failure because `config.py` does not exist.

- [ ] **Step 3: Implement typed immutable configuration**

```python
@dataclass(frozen=True)
class ThresholdRule:
    confidence: float
    minimum_area: int

@dataclass(frozen=True)
class ExperimentConfig:
    weights: Path
    canonical_source: Path
    historical_dataset: Path
    imgsz: int
    raw_confidence: float
    hit_iou: float
    rules: dict[str, ThresholdRule]
    run_seeds: tuple[int, int]
    images_per_run: int
    sequence_group_size: int
    warmup_count: int
    timing_brackets_ms: tuple[float, ...]
```

Use 500 images/run, 25-image sequence groups, 10 excluded warmups, and weight SHA-256 in the identity lock. Ignore generated manifests, caches, private CSV/JSON, and plots.

- [ ] **Step 4: Run tests; update `AGENTS.md`; commit**

```powershell
git add .gitignore AGENTS.md aoi_yolo/historical_best_deep_eval
git commit -m "feat: freeze historical best evaluation config"
```

### Task 2: Leakage audit and mutually exclusive manifests

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/dataset.py`
- Create: `aoi_yolo/historical_best_deep_eval/prepare_manifests.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_dataset.py`

**Interfaces:**
- Produces `ImageRecord`, `historical_members(dataset) -> set[str]`, `audit_candidates(...) -> list[ImageRecord]`, and `select_disjoint_runs(...) -> tuple[list[ImageRecord], list[ImageRecord]]`.
- Writes `generated/manifests/run_1.csv`, `run_2.csv`, `audit.json`, and `manifest_lock.json`.

- [ ] **Step 1: Write failing exclusion tests**

```python
def test_runs_are_disjoint_and_exclude_history(records):
    run1, run2 = select_disjoint_runs(records, {"0001"}, (11, 22), 3)
    assert len(run1) == len(run2) == 3
    assert {r.stem for r in run1}.isdisjoint(r.stem for r in run2)
    assert {r.group for r in run1}.isdisjoint(r.group for r in run2)
    assert "0001" not in {r.stem for r in run1 + run2}
```

Also assert selection fails with “1,000 eligible independent images” rather than relaxing exclusions.

- [ ] **Step 2: Verify failure; implement audit/sampling**

Read canonical BMP/JSON pairs; map `inpinhole` to `pinhole`; record hashes, dimensions, groups, source/target labels, object counts, and positive-image labels. Exclude both historical train and validation directories plus SHA duplicates. Select run 1 by seed, remove its groups, then select run 2 by the second seed without class quotas.

- [ ] **Step 3: Test deterministic 500/500 manifests and generate them**

```powershell
.\.venv_yolo\Scripts\python.exe -m pytest aoi_yolo\historical_best_deep_eval\tests\test_dataset.py -q
.\.venv_yolo\Scripts\python.exe -m aoi_yolo.historical_best_deep_eval.prepare_manifests
```

Expected: 500 unique images/run; no shared image/group/hash; no historical overlap.

- [ ] **Step 4: Update `AGENTS.md`; commit**

```powershell
git add AGENTS.md aoi_yolo/historical_best_deep_eval
git commit -m "feat: freeze disjoint historical model holdouts"
```

### Task 3: Exactly-once production timing and prediction cache

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/cache.py`
- Create: `aoi_yolo/historical_best_deep_eval/production_pass.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_cache.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_production_pass.py`

**Interfaces:**
- Produces `PredictionRecord`, `ImagePrediction`, `write_cache`, `read_cache`, `apply_cascade`, and `run_production_pass(run_id, manifest, config, output) -> Path`.
- Cache format: gzip JSONL with polygons, class, confidence, area, kept state, and stage timings.

- [ ] **Step 1: Write failing cache/cascade tests**

```python
def test_cache_round_trip(tmp_path, identity, prediction):
    path = tmp_path / "cache.jsonl.gz"
    write_cache(path, [prediction], identity)
    assert read_cache(path) == (identity, [prediction])

def test_historical_cascade():
    rows = [pred("pinhole", .29, 500), pred("pinhole", .31, 149), pred("scratch", .21, 301)]
    assert [p.label for p in apply_cascade(rows, frozen_rules())] == ["scratch"]
```

- [ ] **Step 2: Write exactly-once test**

```python
def test_each_measured_image_is_inferred_once(fake_model, manifest, tmp_path):
    run_production_pass("run_1", manifest, config_for(fake_model), tmp_path)
    assert fake_model.calls == [row.image_path for row in manifest]
    assert len(read_timing(tmp_path / "run_1" / "timing.csv")) == 500
```

Warmups use 10 historical excluded images, never measured manifest images.

- [ ] **Step 3: Implement the timed loop**

Decode each BMP before timing. Synchronize CUDA before/after prediction. Record Ultralytics preprocess/inference/postprocess, polygon conversion, cascade time, total wall latency, image size, and raw/kept counts. Do not import LabelMe parsing/matching. Write to a temporary directory and atomically finalize only after 500 rows plus an identity/checksum marker. Refuse to overwrite a completed run.

- [ ] **Step 4: Run tests and a two-image smoke; update `AGENTS.md`; commit**

```powershell
git add AGENTS.md aoi_yolo/historical_best_deep_eval
git commit -m "feat: add exactly once production timing pass"
```

### Task 4: Cached-prediction scoring

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/scoring.py`
- Create: `aoi_yolo/historical_best_deep_eval/score_run.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_scoring.py`

**Interfaces:**
- Produces `ground_truth`, `match_one_to_one`, `score_cached_run(manifest, cache, output, hit_iou) -> dict`, and `wilson_interval`.
- Accepts no model or weight argument.

- [ ] **Step 1: Write failing reconciliation/no-model tests**

```python
def test_scoring_reconciles_without_ultralytics(monkeypatch, fixture_run):
    monkeypatch.setitem(sys.modules, "ultralytics", ForbiddenImport())
    summary = score_cached_run(*fixture_run)
    assert summary["ALL"]["tp"] + summary["ALL"]["fn"] == summary["ALL"]["objects"]
    assert summary["ALL"]["slip_rows"] == summary["ALL"]["fn"]
```

Verify Wilson 80/100 is approximately `(0.7112, 0.8666)`.

- [ ] **Step 2: Implement class-aware one-to-one matching**

Load/rasterize LabelMe only here. Write event/slip CSV and per-class JSON containing annotated objects, positive images, TP/FP/FN, recall, precision, F1, Wilson intervals, and matched mask IoU.

- [ ] **Step 3: Run CPU-only scoring tests/smoke; update `AGENTS.md`; commit**

Expected: no Ultralytics import and no GPU initialization.

### Task 5: Per-run timing brackets and defect relationships

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/statistics.py`
- Create: `aoi_yolo/historical_best_deep_eval/plotting.py`
- Create: `aoi_yolo/historical_best_deep_eval/analyze_run.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_statistics.py`

**Interfaces:**
- Produces `assign_bracket`, `summarize_latency`, `analyze_timing_relationships`, `bootstrap_interval`, and `holm_adjust`.

- [ ] **Step 1: Write failing bracket/correlation tests**

```python
@pytest.mark.parametrize(("ms", "label"), [(24.999, "<25"), (25, "25-<50"), (1000, ">=1000")])
def test_bracket_boundaries(ms, label):
    assert assign_bracket(ms, DEFAULT_EDGES).label == label
```

Verify Spearman rho is 1.0 for monotonic prediction count/latency fixtures.

- [ ] **Step 2: Implement summaries and within-run statistics**

Report count, mean, SD, median, p90/p95/p99/max, deterministic bootstrap mean/median CIs, fixed bracket counts/percentages, and pinhole-only/scratch-only/mixed/no-target timing. Calculate Spearman correlations, Kruskal-Wallis, corrected pairwise Mann-Whitney, Cliff’s delta, and Holm p-values; mark under-supported groups.

- [ ] **Step 3: Implement plots**

Create histogram, empirical CDF, bracket counts, category distribution, and latency-versus-prediction-count plots solely from timing/cache/scoring artifacts.

- [ ] **Step 4: Run tests; update `AGENTS.md`; commit**

```powershell
git commit -m "feat: analyze historical model timing distributions"
```

### Task 6: Between-run comparison

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/compare_runs.py`
- Create: `aoi_yolo/historical_best_deep_eval/tests/test_compare_runs.py`

**Interfaces:**
- Produces `compare_runs(run1: RunResults, run2: RunResults, seed: int) -> dict`.

- [ ] **Step 1: Write failing deterministic comparison tests**

```python
def test_identical_runs_have_zero_effect():
    result = compare_runs(run_result([10, 20, 30]), run_result([10, 20, 30]), seed=7)
    assert result["timing"]["cliffs_delta"] == 0
    assert result["timing"]["median_difference_ms"] == 0
```

Also assert differing defect support is reported before performance differences.

- [ ] **Step 2: Implement comparisons**

Compute Mann-Whitney U, two-sample KS, Cliff’s delta, bootstrap mean/median differences, bracket chi-square (or exact/Monte Carlo alternative for sparse cells), composition differences, and per-defect precision/recall differences with CIs. Holm-correct category families.

- [ ] **Step 3: Generate combined CDF, bracket, category, and defect-support/accuracy plots**

- [ ] **Step 4: Run tests; update `AGENTS.md`; commit**

```powershell
git commit -m "feat: compare independent historical model runs"
```

### Task 7: Execute the two exactly-once runs and offline scoring

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/run_experiment.py`
- Modify: `aoi_yolo/historical_best_deep_eval/config.yaml`

**Interfaces:**
- CLI stages: `--stage production --run run_1|run_2`, `--stage score --run ...`, and `--stage compare`.

- [ ] **Step 1: Test orchestration guards**

Scoring refuses missing/incomplete 500-row caches; production refuses a completed run; comparison refuses partial results.

- [ ] **Step 2: Run the complete focused suite**

Run: `.\.venv_yolo\Scripts\python.exe -m pytest aoi_yolo\historical_best_deep_eval\tests -q`

- [ ] **Step 3: Execute production run 1 exactly once**

```powershell
.\.venv_yolo\Scripts\python.exe -m aoi_yolo.historical_best_deep_eval.run_experiment --stage production --run run_1
```

Verify 500 unique timing/cache rows and 500 measured calls.

- [ ] **Step 4: Execute production run 2 exactly once**

```powershell
.\.venv_yolo\Scripts\python.exe -m aoi_yolo.historical_best_deep_eval.run_experiment --stage production --run run_2
```

Verify the same plus zero run overlap.

- [ ] **Step 5: Score cached predictions CPU-only**

```powershell
$env:CUDA_VISIBLE_DEVICES=''
.\.venv_yolo\Scripts\python.exe -m aoi_yolo.historical_best_deep_eval.run_experiment --stage score --run run_1
.\.venv_yolo\Scripts\python.exe -m aoi_yolo.historical_best_deep_eval.run_experiment --stage score --run run_2
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

- [ ] **Step 6: Analyze/compare; update `AGENTS.md`; commit orchestration**

```powershell
.\.venv_yolo\Scripts\python.exe -m aoi_yolo.historical_best_deep_eval.run_experiment --stage compare
git commit -m "feat: run historical best deep evaluation"
```

### Task 8: Final report, changelog #23, and verification

**Files:**
- Create: `aoi_yolo/historical_best_deep_eval/HISTORICAL_BEST_DEEP_EVALUATION_RESULTS.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Reconcile artifacts**

For each run verify 500 manifest rows, 500 unique timing rows, 500 cache records, TP+FN equals target objects, slip count equals FN, Wilson intervals exist, and no historical/cross-run stem/group/hash overlap.

- [ ] **Step 2: Write the report**

Include weight hash, commands, dataset composition, defect-specific TP/FP/FN/recall/precision/F1/CIs, timing stages/brackets, category timing, correlations/tests/effects, between-run differences, plots, limitations, and operational interpretation.

- [ ] **Step 3: Finalize changelog #23**

Change “Direction” to “Completed” with local timestamp and exact seeds, counts, metrics, timing, tests, exactly-once evidence, and production C# non-modification.

- [ ] **Step 4: Verify**

```powershell
.\.venv_yolo\Scripts\python.exe -m pytest aoi_yolo\historical_best_deep_eval\tests -q
git diff --check
```

Visually inspect all plots for labels, units, brackets, legends, and sample counts.

- [ ] **Step 5: Commit and publish only public-safe code/docs**

```powershell
git add AGENTS.md aoi_yolo/historical_best_deep_eval/HISTORICAL_BEST_DEEP_EVALUATION_RESULTS.md
git commit -m "docs: report historical best deep evaluation"
```

Do not publish source images, LabelMe JSON, model weights, prediction caches, or private event/timing data.

