# Option B - YOLO Candidate Crop Classifier

This option runs YOLO at low confidence, crops each candidate, then uses a small binary CNN to decide whether the candidate is a real defect.

Script:

```text
aoi_yolo/hybrid_models/option_b_crop_classifier/train_and_eval_crop_filter.py
```

Current run:

```powershell
.\.venv_yolo\Scripts\python.exe aoi_yolo\hybrid_models\option_b_crop_classifier\train_and_eval_crop_filter.py --src "test images" --dataset "aoi_yolo\dataset_core2_bg_v2" --weights "aoi_yolo\runs\core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best\weights\best.pt" --out "aoi_yolo\hybrid_models\option_b_crop_classifier" --imgsz 1280 --conf 0.05 --hit-iou 0.05 --epochs 12 --batch 64
```

Outputs:

```text
candidate_filter.pt
train_candidates.csv
val_candidates.csv
val_candidates_with_filter_probs.csv
crops/
```

Corrected one-to-one validation results:

```text
filter threshold 0.20: recall 95.34%, precision 55.72%
filter threshold 0.30: recall 95.07%, precision 56.33%
filter threshold 0.40: recall 93.01%, precision 57.89%
filter threshold 0.50: recall 83.01%, precision 63.52%
filter threshold 0.60: recall 60.00%, precision 76.04%
filter threshold 0.70: recall 37.40%, precision 80.06%
filter threshold 0.80: recall 11.10%, precision 76.42%
```

Conclusion: this first crop-only classifier does not beat Option A. It likely needs stronger inputs, such as larger context, mask/shape features, class-specific classifiers, or hard-negative review.

