# Training and Testing Utilities

This directory contains model, loss/metric, and data-loading definitions used by
the repository-level training and testing entry points.

The modules are not normally executed directly. They are imported by:

```text
../train_slam.py
../test_slam.py
../cross_val.py
../cross_val_single.py
```

## Contents

### `unet_parts.py`

Defines reusable encoder–decoder/U-Net building blocks.

This file is likely derived from or adapted from the upstream RadarHD
implementation and must be included in the file-level provenance audit before a
public release.

### `model.py`

Contains the network architectures and model variants explored during the
project.

The directory's previous documentation notes that several architectures were
tested and that `Unet4` was left as an unfinished or future research direction.
Verify that unused or incomplete classes are clearly marked in the source.

### `dice_score.py`

Defines Dice-based loss or score utilities used during model optimization and
evaluation.

### `dataloader.py`

Provides the data-loading behavior compatible with the original RadarHD-style
processed radar–LiDAR image dataset.

### `dataloader_slam.py`

Contains the data-loading adaptations introduced for the SLAM-RF dataset and
the student group's processed dataset structure.

## Usage

These modules should be used through the top-level scripts:

```bash
python3 train_slam.py
python3 test_slam.py
python3 cross_val.py
python3 cross_val_single.py
```

To inspect the active command-line interface:

```bash
python3 train_slam.py --help
python3 test_slam.py --help
python3 cross_val.py --help
python3 cross_val_single.py --help
```

If a script does not expose CLI options, verify and record the configuration
constants defined inside that script before execution.

## Expected responsibilities

The training/testing utilities collectively define:

- input file discovery;
- radar and LiDAR sample pairing;
- tensor loading and preprocessing;
- dataset indexing;
- model construction;
- forward propagation;
- loss and metric computation.

Changes in this directory may affect checkpoint compatibility. Avoid renaming
classes, changing tensor shapes, or moving modules until existing checkpoints
and imports have been tested.

## Attribution

The model architecture and part of the original training/testing utility
structure are based on the RadarHD research implementation by Akarsh Prabhakara
et al.

The EPFL student group adapted and extended the code for SLAM-RF-specific data,
model experiments, and project workflows:

- M. Atwi
- S. Bernasconi
- A. Dell'Orto

The exact origin of each file must be recorded in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) after comparison with
the upstream repository.
