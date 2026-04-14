# LOSO Prepared Datasets

This folder stores fold-ready datasets for outer-subject LOSO and inner train/val splits.

## File format

Each `.pt` file is a dictionary with:

- `features`: `FloatTensor [N, T, C]`
- `labels`: `LongTensor [N]`
- `subject_ids`: `LongTensor [N]` (`1`=S1, `2`=S2, `3`=S3)

## Subject mapping

- `S1` -> `daniela_group_1`
- `S2` -> `daniela_group_2`
- `S3` -> `daniela_group_3`

## Folds

- `fold1`: train on `S2+S3`, test on `S1`
- `fold2`: train on `S1+S3`, test on `S2`
- `fold3`: train on `S1+S2`, test on `S3`

Inner split is stratified 80/20 over outer-train labels.

See `loso_manifest.json` for per-file sample counts and label distributions.
