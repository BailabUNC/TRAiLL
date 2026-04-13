# TRAiLL Training Workspace

This workspace is focused on the three-stage SOMA training pipeline only:

- Phase I: denoising autoencoder
- Phase II: SimSiam representation refinement
- Phase III: downstream end-task classifier

Runtime folders such as `eval/` and `experiments/` are intentionally excluded from this setup.

## Core Scripts

- `path_config.py` - centralized path defaults and fallback resolution
- `train_phase1_autoencoder.py` - Phase I training
- `train_phase2_simsiam.py` - Phase II training
- `train_phase3_end_task.py` - Phase III training
- `tools/phase3_end_task_lib.py` - shared utilities for Phase III

## Required Directory Layout

```text
TRAiLL/
├── data/
│   ├── .augmented/
│   └── baseline_arrays/
├── checkpoints/
│   ├── phase1/
│   ├── phase2/simsiam/
│   ├── phase3/
│   └── analysis/
├── outputs/
│   ├── phase1/{arrays,plots}/
│   ├── phase2/{arrays,plots}/
│   ├── phase3/{arrays,plots}/
│   ├── paper/
│   └── legacy/
├── archive/
│   └── checkpoints/
├── tools/
│   ├── __init__.py
│   └── phase3_end_task_lib.py
├── path_config.py
├── train_phase1_autoencoder.py
├── train_phase2_simsiam.py
└── train_phase3_end_task.py
```

## Default Data Dependencies

- Phase I default input:
  - `data/.augmented/augmented_dataset_letters_group_1_10_std0.15.pt`
- Phase II default input:
  - `data/.augmented/augmented_dataset_letters_group_1_50_no_translation.pt`
- Phase III default inputs:
  - `data/baseline_arrays/filter_features1.npy`
  - `data/baseline_arrays/filter_labels1.npy`

## Checkpoint Dependencies

- Phase II requires a Phase I checkpoint:
  - `checkpoints/phase1/augmented_phase_1.pth`
- Phase III requires an encoder checkpoint pack (typically containing `"enc"`):
  - `checkpoints/phase3/model_best.pth`

## Smoke Checks

Run from repository root:

```bash
python train_phase1_autoencoder.py --dry-run
python train_phase2_simsiam.py --help
python train_phase3_end_task.py --dry-run
python -m py_compile path_config.py train_phase1_autoencoder.py train_phase2_simsiam.py train_phase3_end_task.py tools/phase3_end_task_lib.py
```