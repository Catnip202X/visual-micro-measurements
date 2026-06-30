# Hybrid Model Comparison

Both options use the same YOLO base model and the same expanded validation split:

```text
base model: aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt
dataset: aoi_yolo/dataset_core2_bg_v2
validation images: 181
ground-truth defects: 730
```

## Option A - Threshold Cascade

Configuration:

```text
pinhole confidence >= 0.30
scratch confidence >= 0.20
pinhole area >= 150 px
scratch area >= 300 px
```

Result:

```text
recall: 83.56%
precision: 83.22%
hits: 610
predictions: 733
false positives: 123
```

Optional edge-mask rejection was tested using the existing `edge` annotations:

```text
edge exclusion 10 px:
recall: 81.64%
precision: 83.59%

edge exclusion 25 px:
recall: 79.32%
precision: 83.67%
```

This suggests that masking near annotated edges can remove some false positives, but it also removes true defects. A stronger result likely requires a true inspectable-face mask, not just an edge-margin mask.

## Option B - Crop Classifier

Best useful measured point from the first crop-only classifier:

```text
classifier threshold: 0.50
recall: 83.01%
precision: 63.52%
hits: 606
predictions: 954
false positives: 348
```

Higher classifier thresholds improve precision only modestly while recall drops sharply:

```text
threshold 0.60: recall 60.00%, precision 76.04%
threshold 0.70: recall 37.40%, precision 80.06%
```

## Recommendation

Use Option A for now. It is simpler, faster, and clearly better on the current validation split.

Option B is still promising as a concept, but the first crop-only classifier is not enough. The next version should include larger context, shape/geometry features, and class-specific hard-negative mining.

The next masking experiment should define the actual inspectable product face/side, either from machine coordinates or from a generated polygon mask per image. That would let the detector ignore dust/background outside the relevant part surface before scoring candidates.
