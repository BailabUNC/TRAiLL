# TRAiLL Data Folder

This folder contains various `.pt` files and related data files used for training, validation, and augmentation in the TRAiLL project. Below is an explanation of the types of files you might find here:

## 1. Raw Dataset Files
- **Format**: `dataset-{person}-{letter}.pt` or `dataset-{person}-{command}.pt`
- **Description**: These files contain raw data collected from participants performing specific actions or gestures.
- **Examples**:
  - `dataset-john-A.pt`: Data for participant "John" performing the letter "A".
  - `dataset-john-open.pt`: Data for participant "John" performing the "open" command.

## 2. Augmented Dataset Files
- **Format**: `dataset-{person}-{letter}_{augmentation}.pt`
- **Description**: These files contain augmented versions of the raw datasets, where transformations like rotation, translation, or noise have been applied.
- **Examples**:
  - `dataset-john-A_offset.pt`: Augmented data for "John" performing the letter "A" with an offset transformation.
  - `dataset-john-B_rotate.pt`: Augmented data for "John" performing the letter "B" with a rotation transformation.

## 3. Concatenated Dataset Files
- **Format**: `concatenated_dataset-{person}-{pattern}.pt`
- **Description**: These files are created by concatenating multiple datasets (e.g., all letters or commands) for a specific participant.
- **Examples**:
  - `concatenated_dataset-john-letters.pt`: Concatenated dataset for "John" containing all letter data.
  - `concatenated_dataset-john-commands.pt`: Concatenated dataset for "John" containing all command data.

## 4. Processed Dataset Files
- **Format**: `processed_dataset-{person}-{pattern}.pt`
- **Description**: These files contain preprocessed data ready for training or validation, often derived from raw or concatenated datasets.
- **Examples**:
  - `processed_dataset-john-letters.pt`: Preprocessed dataset for "John" containing all letter data.

## 5. Augmentation Metadata
- **Description**: Files containing metadata about the augmentation process, such as parameters used for transformations.
- **Purpose**: These files may accompany augmented datasets to provide additional context.

---

This structure ensures that raw, augmented, and processed datasets are organized and easily accessible for training and evaluation purposes.