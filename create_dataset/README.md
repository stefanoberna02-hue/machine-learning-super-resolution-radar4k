# SLAM-RF Dataset Preparation

This directory contains the preprocessing utilities used to transform
authorized raw SLAM-RF radar and LiDAR recordings into paired samples suitable
for the model.

You can ignore this directory when working only with an already prepared
training/test dataset.

## Restricted-data notice

The SLAM-RF data were provided by the EPFL Laboratory of Sensing and Networking
Systems (SENS Lab) under restricted access conditions.

The complete raw recordings and complete derived dataset must not be committed
or redistributed without separate authorization. See
[`../DATA_ACCESS.md`](../DATA_ACCESS.md).

## Local input structure

The existing scripts expect or are designed around a structure similar to:

```text
RadarHD-dataset-1/
├── lidar_pcl/
└── radar/
```

The full authorized data may remain in these paths in the local working copy,
while Git ignores their contents.

The private repository may retain one reduced, explicitly authorized structural
sample per modality:

```text
RadarHD-dataset-1/
├── README.md
├── lidar_pcl/
│   └── 112_fwd.csv
└── radar/
    └── 112_read.pkl
```

The samples document schema and implementation choices only. They do not make
the repository independently reproducible.

## Files

### `sync_slam_rf.py`

Main SLAM-RF synchronization and preprocessing script.

Its responsibilities may include:

- reading radar and LiDAR recordings;
- reconciling timestamps;
- associating radar and LiDAR observations;
- applying radar preprocessing;
- clipping or thresholding sensor representations;
- generating paired model inputs and labels;
- saving the processed dataset.

Run from the repository root:

```bash
python3 create_dataset/sync_slam_rf.py
```

Before execution, inspect:

- input paths;
- output paths;
- timestamp units and offsets;
- trajectory/session selection;
- clipping ranges;
- thresholding method and values;
- output resolution;
- overwrite behavior.

### `TestImagefromarray.py`

Utility used to inspect or validate image/array conversion.

Run:

```bash
python3 create_dataset/TestImagefromarray.py
```

The script may require the input path to be configured in the source.

### `train_test_split.py`

Creates a standard train/test split from the processed paired dataset.

Run:

```bash
python3 create_dataset/train_test_split.py
```

Record the random seed, split ratio, source dataset, and generated manifest.

### `train_test_split_byDays.py`

Creates a split grouped by acquisition day or session.

Run:

```bash
python3 create_dataset/train_test_split_byDays.py
```

This strategy is useful when samples collected close in time are strongly
correlated and should not be divided across training and test sets.

## Preprocessing flexibility

The project pipeline supports or explores choices such as:

- radar data enhancement;
- magnitude-based thresholding;
- alternative threshold values;
- clipping of LiDAR point clouds;
- clipping of polar radar images;
- synchronization and nearest-time association;
- custom train/test partitioning.

Every generated dataset should be accompanied by a short manifest documenting
the settings used.

Recommended manifest:

```text
processed_dataset/
├── README.md
├── preprocessing.json
├── split_manifest.csv
├── train/
└── test/
```

## Safety checks

Before a long preprocessing run:

1. test on one short trajectory;
2. verify radar/LiDAR timestamp ordering;
3. inspect several paired outputs;
4. confirm coordinate orientation and range limits;
5. verify that train/test splitting does not leak sessions;
6. write to a new output directory instead of overwriting source data.

## Attribution

The preprocessing workflow is part of the student group's adaptation of
RadarHD to the SLAM-RF dataset:

- M. Atwi
- S. Bernasconi
- A. Dell'Orto

Any lab-provided code fragments or upstream RadarHD-derived preprocessing must
be identified file by file in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
