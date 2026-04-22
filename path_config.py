"""Repository path constants and ``first_existing()`` for robust fallbacks.

Import ``ROOT``, ``DATA_DIR``, checkpoint tuples, ``OUTPUTS_PHASE*_``, ``OUTPUTS_PAPER``,
and ``OUTPUTS_LEGACY`` instead of hard-coding strings. Paths are under the directory
containing this file (repo root).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def repo_root_from_here(file: str) -> Path:
    """Walk parents until a directory containing path_config.py (this file's package root)."""
    p = Path(file).resolve()
    for d in [p.parent, *p.parents]:
        if (d / "path_config.py").exists():
            return d
    return p.parent


def first_existing(*candidates: str) -> str:
    for rel in candidates:
        p = ROOT / rel
        if p.exists():
            return str(p)
    return str(ROOT / candidates[0])


DATA_DIR = str(ROOT / "data")
BASELINE_ARRAYS_DIR = "data/baseline_arrays"
PHASE1_DATA_DEFAULT = (
    "data/.augmented/augmented_dataset_letters_group_1_10_std0.15.pt",
    "data/augmented_dataset_letters_group_1_10_std0.15.pt",
)
PHASE2_DATA_DEFAULT = (
    "data/.augmented/augmented_dataset_letters_group_1_50_no_translation.pt",
    "data/phase2_augmented_dataset_letters_group_1_50_no_translation.pt",
)
# Phase-III numpy splits (1 = eval/confusion default; 2 = mixup experiment)
PHASE3_BASELINE_FEATURES1 = (f"{BASELINE_ARRAYS_DIR}/filter_features1.npy",)
PHASE3_BASELINE_LABELS1 = (f"{BASELINE_ARRAYS_DIR}/filter_labels1.npy",)
PHASE3_BASELINE_FEATURES2 = (f"{BASELINE_ARRAYS_DIR}/filter_features2.npy",)
PHASE3_BASELINE_LABELS2 = (f"{BASELINE_ARRAYS_DIR}/filter_labels2.npy",)
PHASE3_MIXUP_ARCHIVE_DIR = "archive/checkpoints/phase3_mixup_filter2"
NOTEBOOK_CACHE_DIR = "data/notebook_cache"

OUTPUTS_DIR = ROOT / "outputs"
# Publication-style exports and legacy panels (merged from former ``figures/paper`` and ``figures/legacy``).
OUTPUTS_PAPER = OUTPUTS_DIR / "paper"
OUTPUTS_LEGACY = OUTPUTS_DIR / "legacy"
OUTPUTS_LOSO = OUTPUTS_DIR / "loso"
OUTPUTS_PHASE1 = OUTPUTS_LOSO / "phase1"
OUTPUTS_PHASE2 = OUTPUTS_LOSO / "phase2"
OUTPUTS_PHASE3 = OUTPUTS_LOSO / "phase3"
OUTPUTS_PHASE1_PLOTS = OUTPUTS_PHASE1 / "plots"
OUTPUTS_PHASE1_ARRAYS = OUTPUTS_PHASE1 / "results"
OUTPUTS_PHASE2_PLOTS = OUTPUTS_PHASE2 / "plots"
OUTPUTS_PHASE2_ARRAYS = OUTPUTS_PHASE2 / "results"
OUTPUTS_PHASE3_PLOTS = OUTPUTS_PHASE3 / "plots"
OUTPUTS_PHASE3_ARRAYS = OUTPUTS_PHASE3 / "results"

# --- Model checkpoints (.pth / weights .pt). Feature tensors (.pt) live under data/ ---
PHASE1_AUTOENCODER_CKPT = (
    "checkpoints/phase1/augmented_phase_1.pth",
)
PHASE2_SIMSIAM_DIR = "checkpoints/phase2/simsiam"
# Historical weights (not manuscript mainline): under ``archive/checkpoints/`` (see ``archive/checkpoints/README.md``)
PHASE2_BYOL_DIR = "archive/checkpoints/phase2_byol"
PHASE2_TRAIN2_DIR = "archive/checkpoints/phase2_train2_contrastive"
# Phase III (manuscript): frozen encoder + MLP classifier checkpoints (was checkpoints/downstream/)
PHASE3_DIR = "checkpoints/phase3"
BASELINES_DIR = "checkpoints/baselines"
ANALYSIS_DIR = "checkpoints/analysis"

PHASE3_ENCODER_BEST = (
    "checkpoints/phase3/model_best.pth",
    "checkpoints/downstream/model_best.pth",
    "runs/model_best.pth",
)
CHANNEL_SCORES_CKPT = (
    "checkpoints/analysis/channel_scores.pt",
    "outputs/loso/phase2/results/channel_scores.pt",
    "outputs/loso/phase2/arrays/channel_scores.pt",
    "outputs/channel_scores.pt",
    "channel_scores.pt",
)
PHASE3_END_TASK_MLP_DROP_F3 = (
    "checkpoints/phase3/end_task_mlp_drop_F3_acc0.868.pth",
    "checkpoints/downstream/end_task_mlp_drop_F3_acc0.868.pth",
    "runs/end_task_mlp_drop_F3_acc0.868.pth",
)

