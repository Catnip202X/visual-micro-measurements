# Visual Micro Measurements AOI YOLO Sidecar

Public code and documentation for the offline YOLO-style inspection sidecar used with the visual micro measurements AOI workspace.

This repository is intentionally scoped to sidecar tooling:

- dataset auditing and LabelMe-to-YOLO conversion scripts
- defect evaluation utilities
- edge-policy and product-face mask pseudo-labeling tools
- hard-negative mining experiments
- lightweight training/evaluation notes
- placeholder documentation for local BMP/JSON test batches

## Data Policy

The real inspection batches are not tracked:

- `test images/*.bmp`
- per-image LabelMe JSON metadata
- generated YOLO datasets
- generated product masks and review overlays
- training runs and model weights

Use `test images/README.md` and `test images/metadata_template.json` for the expected local file layout.

## Dependencies

Install dependencies from `requirements.txt`. On the production GPU machine, install CUDA-matched PyTorch and TorchVision wheels first, then install the rest of the requirements.
