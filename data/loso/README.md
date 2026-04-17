# LOSO Prepared Datasets (mixed pseudo-sessions)

Group `concatenated_dataset-daniela-letters-group_*.pt` files were **shuffled and split**
into three pseudo-sessions (`mixed_session1.pt` … `mixed_session3.pt`).
LOSO then treats `S1`/`S2`/`S3` as those sessions (not raw recording groups).

Provenance: `origin_group_ids` in each `.pt` is `1/2/3` for original daniela group.

## File format

Each fold `.pt` dictionary includes:

- `features`, `labels`, `subject_ids`
- `instance_ids` (optional): offset so IDs are unique across concatenated sessions
- `origin_group_ids` (optional): original daniela group index per window

## Folds

- `fold1`: train on `S2+S3`, test on `S1`
- `fold2`: train on `S1+S3`, test on `S2`
- `fold3`: train on `S1+S2`, test on `S3`

Inner split: stratified 80/20 over outer-train **labels**.

See `loso_manifest.json` and `mixed_sessions_manifest.json`.
