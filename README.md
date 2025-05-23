# TRAiLL: Tracking & Reconstructing Array of near-infrared LED for body Locomotion

## Overview

**TRAiLL** is a comprehensive Python toolkit for collecting, processing, visualizing, and modeling sensor data from wearable near-infrared LED arrays. It is designed for research and prototyping in gesture recognition, prosthetics, rehabilitation, and human-computer interaction.

---

## Features

- **Real-Time Data Acquisition**: Collects data from wearable NIR sensor arrays via serial communication.
- **Flexible Activity Profiles**: Easily switch between gesture/activity sets using JSON-based profiles.
- **Live Visualization**: Real-time heatmap and activity status panel for intuitive feedback.
- **Data Processing Utilities**: Filtering, augmentation, and concatenation scripts for robust dataset creation.
- **Model Training & Evaluation**: PyTorch-based scripts for training and evaluating gesture recognition models.
- **Extensible & Modular**: Easily adapt to new sensor layouts, activities, or machine learning models.

---

## Quick Start

### 1. Clone the Repository

```
bash
git clone https://github.com/yourusername/TRAiLL.git
cd TRAiLL
```

### 2. Set Up the Environment
Create and activate a virtual environment:

```
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

Install dependencies:

```
pip install -r [requirements.txt](http://_vscodecontentref_/0)
```

### 3. Data Acquisition & Visualization
Connect your device and run:
```
python -m traill_daq.run --port <YOUR_SERIAL_PORT> --profile letters
```

- `--port`: Serial port for your device (e.g., COM31).
- `--profile`: Name of the activity profile (see traill_daq/activity_profiles.json).

### 4. Activity Profiles
Define your activity sets in traill_daq/activity_profiles.json:
```
{
  "letters": { "activities": ["open", "a", "b", "c", ..., "z"] },
  "fingers": { "activities": ["open", "thumb", "index", "middle", "ring", "pinky"] }
}
```

---
## Main Components
- traill_daq/: Data acquisition and real-time visualization.

  - run.py: Main entry point for data collection and visualization.
  - traill_visualizer.py: Visualization logic and GUI.
  - activity_profiles.json: Activity/gesture profile definitions.
  - traill/: Data processing and machine learning.

- traill_dataset.py: Dataset utilities.

  - traill_dataset_concat.py: Concatenate multiple datasets.
  - traill_dataset_augmentation.py: Data augmentation scripts.
  - model.py: Model definitions.
  - train.py: Training script.
  - predict.py: Model evaluation and prediction.

- data/: Raw and processed data files.

- result/: Analysis and visualization scripts (e.g., generate_heatmap.py).

---
## Example Usage
Data Collection
`python -m traill_daq.run --port <YOUR_SERIAL_PORT> --profile letters`

Data Processing
Concatenate datasets:
`python [traill_dataset_concat.py](http://_vscodecontentref_/1) person letters --data-dir data/processed`

Augment datasets:
`python [traill_dataset_augmentation.py](http://_vscodecontentref_/2) --input data/processed/your_dataset.pt --output data/processed/augmented.pt`

Model training:
`python [train.py](http://_vscodecontentref_/3) data/processed/your_dataset.pt --epochs 20 --batch 32 --lr 2e-3 --dropout 0.2`

Model evaluation:
`python [predict.py](http://_vscodecontentref_/4) data/processed/your_dataset.pt best_model.pth --batch-size 32 --num-classes 26`

Heatmap visualization:
`python [generate_heatmap.py](http://_vscodecontentref_/5)`

---
## Folder Structure
```
TRAiLL/
│
├── traill/                # Data processing and model code
├── traill_daq/            # Data acquisition and visualization
│   ├── activity_profiles.json
│   ├── run.py
│   └── traill_visualizer.py
├── data/                  # Raw and processed data
├── result/                # Analysis and visualization scripts
├── [requirements.txt](http://_vscodecontentref_/6)
└── [README.md](http://_vscodecontentref_/7)
```

---
## License
MIT License

---
## Acknowledgements
TRAiLL was developed at the University of North Carolina at Chapel Hill.
Special thanks to all contributors and the open-source community.