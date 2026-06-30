$env:YOLO_CONFIG_DIR = 'C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\ultralytics_config'
$env:MPLCONFIGDIR = 'C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\matplotlib_config'
$env:PYTHONUTF8 = '1'

New-Item -ItemType Directory -Force 'C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\matplotlib_config' | Out-Null
New-Item -ItemType Directory -Force 'C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\runs\logs' | Out-Null
Set-Location 'C:\Users\Catnip\Documents\visual micro measurements'

.\.venv_yolo\Scripts\yolo.exe segment train `
  model=yolo11n-seg.pt `
  data='C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\dataset_defects\data.yaml' `
  epochs=150 `
  imgsz=1024 `
  batch=4 `
  workers=0 `
  device=0 `
  project='C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\runs' `
  name=defects_yolo11n_seg_e150_i1024 `
  exist_ok=True `
  patience=50 `
  plots=True `
  *>&1 | Tee-Object -FilePath 'C:\Users\Catnip\Documents\visual micro measurements\aoi_yolo\runs\logs\train_e150_i1024.combined.log'
