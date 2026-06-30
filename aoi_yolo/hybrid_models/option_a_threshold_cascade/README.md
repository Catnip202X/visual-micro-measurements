# Option A - Threshold Cascade

This option uses one YOLO segmentation model with class-specific confidence and area gates.

It does not train a second model. It runs the detector once at a base confidence, then keeps only candidates that pass per-class filters.

Current model:

```text
aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt
```

Current balanced operating point:

```powershell
.\.venv_yolo\Scripts\python.exe aoi_yolo\evaluate_json_polygons.py --src "test images" --dataset "aoi_yolo\dataset_core2_bg_v2" --weights "aoi_yolo\runs\core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best\weights\best.pt" --imgsz 1280 --conf 0.05 --class-conf pinhole=0.30 scratch=0.20 --class-area-min pinhole=150 scratch=300 --hit-iou 0.05 --rename inpinhole=pinhole
```

Last measured result on the expanded validation split:

```text
overall recall: 83.56%
overall precision: 83.22%
```

