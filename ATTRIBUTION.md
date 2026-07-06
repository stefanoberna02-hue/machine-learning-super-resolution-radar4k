# Attribution

## Student project

This repository contains work jointly developed by:

- M. Atwi
- S. Bernasconi
- A. Dell'Orto

The project was completed as part of **CS-433 Machine Learning at EPFL**.

This personal repository is maintained by S. Bernasconi for private archival,
research, and portfolio purposes. Maintenance of this copy does not transfer
exclusive authorship or ownership of the group contributions to the repository
maintainer.


## Laboratory supervision and data

The project was supervised and supported by the **EPFL Laboratory of Sensing
and Networking Systems (SENS Lab)**.

The SENS Lab provided access to the proprietary **SLAM-RF** dataset used by the
student group. The full dataset is not redistributed by this repository.

The complete dataset, derived datasets, checkpoints, visualizations constitute private proprietary material
belonging to the SENS lab and as such they are not included in this repository.

## Upstream RadarHD work

The project builds on the RadarHD research work and public implementation:

> Akarsh Prabhakara, Tao Jin, Arnav Das, Gantavya Bhatt, Lilly Kumari,
> Elahe Soltanaghai, Jeff Bilmes, Swarun Kumar, and Anthony Rowe,
> “High Resolution Point Clouds from mmWave Radar,”
> 2023 IEEE International Conference on Robotics and Automation (ICRA),
> pp. 4135–4142, 2023.
> DOI: `10.1109/ICRA48891.2023.10161429`.

Upstream repository:

```text
https://github.com/akarsh-prabhakara/RadarHD
```

RadarHD supplied the conceptual and implementation foundation for the
encoder–decoder model, training/testing structure, data utilities, and
evaluation pipeline.

## Student-group contributions

The student group adapted and extended the upstream work for the SLAM-RF setting,
including work in the following areas:

- radar–LiDAR synchronization;
- SLAM-RF-specific preprocessing;
- paired dataset generation;
- train/test splitting, including day-based splitting;
- data-loading adaptations;
- architecture experiments and model variants;
- training and testing adaptations;
- cross-validation and grid-search workflows;
- EPFL RCP job orchestration;
- Cartesian conversion, point-cloud generation, visualization, and metrics;
- experiment analysis and course reporting.

This list describes the project-level contribution and is not a substitute for
a file-by-file provenance audit.

