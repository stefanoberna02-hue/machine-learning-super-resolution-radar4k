# Data Access and Handling

## Restricted dataset

The experiments in this repository use the proprietary **SLAM-RF** dataset
provided by the EPFL Laboratory of Sensing and Networking Systems (SENS Lab).

The full dataset is not distributed by this repository. Access to this code does
not grant access to the dataset and does not grant permission to redistribute
the original data or restricted derivatives.

## Local directory layout

Authorized users may place the full local data under:

```text
create_dataset/RadarHD-dataset-1/
├── lidar_pcl/
└── radar/
```

This path is preserved because the preprocessing scripts may rely on the
existing directory structure.

## Representative private samples

The private repository may contain one explicitly authorized, reduced,
structurally representative file for each modality:

```text
create_dataset/RadarHD-dataset-1/
├── README.md
├── lidar_pcl/
│   └── 112_fwd.csv
└── radar/
    └── 112_read.pkl
```

Recommended requirements for these samples:

- use matching session or trajectory identifiers;
- retain only the minimum data required to demonstrate schema and dimensions;
- remove unnecessary metadata, personal identifiers, absolute paths, and
  sensitive annotations;
- document how the sample was produced;
- record the authorization permitting its retention;
- do not treat the sample as a redistributable dataset;
- exclude both samples from the future public repository unless public release
  has been separately authorized.

The original multi-gigabyte files should remain outside ordinary Git tracking.

## Derived data

The following may also be restricted and must be reviewed before publication:

- synchronized radar–LiDAR pairs;
- train/validation/test splits;
- polar and Cartesian images;
- point clouds;
- model predictions;
- maps and trajectory visualizations;
- checkpoints trained on the dataset;
- metrics or plots that reveal sensitive characteristics of the recordings.

  