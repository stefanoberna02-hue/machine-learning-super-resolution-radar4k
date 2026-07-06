# Third-Party Notices

This repository contains or may contain source code adapted from the RadarHD
research implementation and other third-party components.

## RadarHD

- **Project:** RadarHD
- **Paper:** *High Resolution Point Clouds from mmWave Radar*
- **Authors:** Akarsh Prabhakara, Tao Jin, Arnav Das, Gantavya Bhatt,
  Lilly Kumari, Elahe Soltanaghai, Jeff Bilmes, Swarun Kumar, Anthony Rowe
- **Repository:** `https://github.com/akarsh-prabhakara/RadarHD`
- **Paper DOI:** `10.1109/ICRA48891.2023.10161429`
- **Observed repository license:** no explicit root license file identified at
  the time this notice was prepared
- **Implication:** public availability of the upstream source must not be
  interpreted as permission to relicense or redistribute it without applicable
  authorization.

## Provenance table

Complete this table after comparing each local file with the upstream
repository and the original group repository.

| Local path | Provenance | Original source | Local modifications | Public-release status |
|---|---|---|---|---|
| `train_slam.py` | To verify | RadarHD `train_radarhd.py` and/or group code | SLAM-RF training adaptations | Pending |
| `test_slam.py` | To verify | RadarHD `test_radarhd.py` and/or group code | SLAM-RF testing adaptations | Pending |
| `train_test_utils/unet_parts.py` | To verify | RadarHD | Model building blocks | Pending |
| `train_test_utils/model.py` | To verify | RadarHD and group code | Architecture variants | Pending |
| `train_test_utils/dice_score.py` | To verify | RadarHD or another source | Loss/metric adaptations | Pending |
| `train_test_utils/dataloader.py` | To verify | RadarHD | Upstream-compatible loader | Pending |
| `train_test_utils/dataloader_slam.py` | To verify | Group adaptation | SLAM-RF loading logic | Pending |
| `eval/*` | To verify | RadarHD and group code | SLAM-RF evaluation pipeline | Pending |
| `create_dataset/sync_slam_rf.py` | To verify | Group and/or lab-provided code | Synchronization/preprocessing | Pending |
| `cross_val.py` | To verify | Group code | Cross-validation/grid search | Pending |
| `cross_val_single.py` | To verify | Group code | Single-run cross-validation | Pending |
| `cluster/*` | To verify | Group code | EPFL RCP orchestration | Pending |

## Dependencies

Python, PyTorch, NumPy, SciPy, OpenCV, Open3D, and other dependencies retain
their own licenses and notices. The Docker image or dependency manifest should
pin versions used for the project, but this repository does not relicense those
packages.

## No blanket license grant

Nothing in this notice grants rights to:

- the SLAM-RF dataset;
- SENS Lab materials;
- course materials;
- third-party source code;
- images or text copied from research publications;
- jointly authored student work beyond the permissions separately agreed by
  its authors.
