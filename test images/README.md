# Test Images Folder

Put local AOI test batches in this folder using the same flat file layout as the development workspace:

```text
test images/
  0001.bmp
  0001.json
  0002.bmp
  0002.json
  ...
```

Rules:

- Each BMP image should have a same-stem LabelMe-style metadata file.
- Example: `0001.bmp` pairs with `0001.json`.
- BMP files are the canonical source images.
- JSON files should store polygon annotations and metadata.
- Keep `imageData` as `null` for repo-friendly metadata handling; do not store embedded image bytes.
- Real BMP files and per-image JSON metadata are intentionally ignored by Git and should not be pushed to GitHub.

Use `metadata_template.json` as a blank starting point for new annotation metadata.
