# Third-Party Notices

This repository contains source code adapted from the RadarHD
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

| Local path | Provenance | Original source | Local modifications |
|---|---|---|---|
| `train_slam.py` | Substantially adapted from upstream RadarHD | RadarHD `train_radarhd.py` | Adapted training to the SLAM-RF dataset; added CLI-based experiment configuration, model selection, optimizer and scheduler options, checkpoint reloading, augmentation handling, range-weighted BCE, configurable BCE/Dice weighting, and reusable training functions for cross-validation. |
| `test_slam.py` | Substantially adapted from upstream RadarHD | RadarHD `test_radarhd.py` | Adapted inference to the SLAM-RF dataset and `SLAMDataset`; added CLI configuration, support for the project model variants, checkpoint selection, experiment-specific paths, and SLAM-RF output naming. |
| `train_test_utils/unet_parts.py` | Direct upstream copy | RadarHD `train_test_utils/unet_parts.py`, itself adapted from `milesial/Pytorch-UNet/unet_parts.py` | No project-specific modifications identified; retained as the common U-Net building blocks. |
| `train_test_utils/model.py` | Upstream-derived and substantially extended | RadarHD `train_test_utils/model.py`, itself adapted from `milesial/Pytorch-UNet/unet_model.py` | Retained the original RadarHD U-Net backbone and added multiple experimental architectures for 180-bin SLAM-RF inputs, including additional convolutional blocks, anisotropic or adaptive azimuth reduction, alternative pooling strategies, and deeper azimuth stems. |
| `train_test_utils/dice_score.py` | Direct upstream copy | RadarHD `train_test_utils/dice_score.py` | No project-specific modifications identified; retained for Dice-loss computation. |
| `train_test_utils/dataloader.py` | Direct upstream copy | RadarHD `train_test_utils/dataloader.py` | No project-specific modifications identified; retained for compatibility with the original RadarHD dataset format. |
| `train_test_utils/dataloader_slam.py` | Group-authored adaptation and extension | RadarHD `train_test_utils/dataloader.py` as the conceptual and API basis | Added support for the SLAM-RF directory and filename conventions, radar–LiDAR pairing by filename, day and experiment grouping, temporal histories, augmentation families, and safeguards against mixing incompatible augmented sequences. |
| `eval/pol_to_cart_slam.py` | Adapted from upstream RadarHD | RadarHD `eval/pol_to_cart.py` | Adapted polar-to-Cartesian conversion to the SLAM-RF output dimensions, range configuration, experiment paths, and generated-output directory structure. |
| `eval/image_to_pcd_slam.py` | Adapted from upstream RadarHD | RadarHD `eval/image_to_pcd.py` | Adapted Cartesian-image-to-point-cloud conversion to the SLAM-RF folder structure, dimensions, filenames, and Python-based evaluation workflow. |
| `eval/pc_compare.py` | Group-authored Python reimplementation and extension | RadarHD `eval/pc_compare.m` and `eval/pc_distance.m` | Reimplemented point-cloud comparison in Python; added Chamfer, Hausdorff, and modified Hausdorff metrics, dataset-level aggregation, and support for the project evaluation outputs. |
| `eval/pc_visualize.py` | Group-authored Python reimplementation | RadarHD `eval/pc_vizualize.m` as the functional reference | Replaced the original MATLAB visualization workflow with a Python-based point-cloud visualization utility adapted to the project output structure. |
| `eval/postprocess_slam.py` | Group-authored orchestration utility | No direct counterpart in RadarHD; combines the roles of the upstream `eval/` scripts | Added an end-to-end post-processing workflow that coordinates polar-to-Cartesian conversion, point-cloud generation, metric computation, and result aggregation. |
| `create_dataset/sync_slam_rf.py` | Group-authored SLAM-RF-specific preprocessing pipeline informed by upstream RadarHD | RadarHD `create_dataset/timestamp_check_radar_lidar.py` and `create_dataset/create_dataset_all_radar_lidar.py` as conceptual references | Implemented the SLAM-RF data layout, chirp-to-frame reconstruction, radar–LiDAR synchronization, sensor-file matching, clipping, thresholding, polar-image generation, LiDAR-label generation, and configurable data enhancement. |
| `cross_val.py` | Group-authored | No direct counterpart in RadarHD | Added multi-configuration and multi-fold cross-validation, experiment aggregation, model selection, and grid-search support. |
| `cross_val_single.py` | Group-authored | No direct counterpart in RadarHD | Added execution of a single cross-validation configuration or fold, using the project training and evaluation functions. |
| `cluster/*` | Group-authored infrastructure and orchestration code | No direct counterpart in RadarHD | Added Run:AI and EPFL RCP orchestration for preprocessing, dataset splitting, training, testing, cross-validation, grid search, and post-processing. Public copies contain sanitized infrastructure placeholders. |

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
