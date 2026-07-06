# EPFL RCP Cluster Orchestration

This directory contains shell scripts created by the student group to launch
and parallelize preprocessing, splitting, training, testing, cross-validation,
grid-search, and post-processing jobs on the EPFL RCP cluster.

The scripts were introduced because the full workflow was computationally
expensive and required multiple coordinated job submissions.

## Important disclaimer

> These scripts were developed for internal use by the original student group
> on the EPFL RCP cluster.
>
> They depend on the software environment, scheduler, filesystem layout,
> storage locations, container registry, permissions, and computational
> resources available to the group during the project.
>
> They are retained for transparency and archival purposes only. The authors do
> not guarantee that the scripts will run without modification on another
> system, or that they will reproduce the reported results without the original
> data, dependencies, environment, random seeds, hardware, and cluster
> configuration.

## Public-release note

The scripts in this directory are sanitized templates derived from the
orchestration scripts used by the student group on the EPFL RCP cluster.

Numeric user/group IDs, personal GASPAR account names, the original scratch
mount path, and the persistent-volume claim name have been replaced with
clearly marked placeholders. The overall Run:AI command structure, resource
requests, experiment grids, and container workflow are intentionally retained
to document how the experiments were orchestrated.

These public copies are illustrative archival templates and are not expected to
run without replacing the placeholders and adapting them to a valid cluster
environment. The original operational versions are retained only in the private
archival repository.

## Placeholders used in the public scripts

The sanitized scripts use values such as:

```text
<SCRATCH_MOUNT>
<PVC_CLAIM>
<UID_STE>, <UID_MOH>, <UID_ALEX>
<GID>
<GASPAR_STE>, <GASPAR_MOH>, <GASPAR_ALEX>
```

They deliberately preserve the original control flow while preventing the
public copies from exposing the operational account and storage configuration.

## High-level orchestration scripts

### `automatic_run_preproc_split.sh`

Launches or coordinates preprocessing and dataset-splitting jobs.

```bash
bash cluster/automatic_run_preproc_split.sh
```

### `automatic_run_training_gs.sh`

Launches or coordinates training/grid-search jobs.

```bash
bash cluster/automatic_run_training_gs.sh
```

### `automatic_run_cross_validation_gs.sh`

Launches or coordinates cross-validation/grid-search jobs.

```bash
bash cluster/automatic_run_cross_validation_gs.sh
```

### `automatic_run_train_test_postproc.sh`

Launches or coordinates the end-to-end training, testing, and post-processing
workflow.

```bash
bash cluster/automatic_run_train_test_postproc.sh
```

## Lower-level scripts

### `run_preproc_split.sh`

Runs a preprocessing/splitting task for a selected configuration.

```bash
bash cluster/run_preproc_split.sh
```

### `run_split.sh`

Runs the configured dataset split.

```bash
bash cluster/run_split.sh
```

### `run_training_gs.sh`

Runs one training or grid-search task.

```bash
bash cluster/run_training_gs.sh
```

### `run_cross_validation_single.sh`

Runs one cross-validation configuration or fold.

```bash
bash cluster/run_cross_validation_single.sh
```

## Minimal consistency corrections

Only small corrections were made in addition to sanitization:

- the enhancement mapping is consistently
  `1=none`, `2=low`, `3=high`, `4=vhigh`, `5=vvhigh`, `6=vvvhigh`;
- a duplicated enhancement-map key was corrected;
- the fixed-user work directory now follows the selected user folder;
- inconsistent script names in usage comments were corrected;
- the no-threshold experiment code no longer assigns magnitude and CFAR values.

## Required review before execution

Inspect every script and replace or verify:

- scheduler directives;
- job names;
- placeholder UID/GID and account values;
- Run:AI project identifiers;
- scratch and other absolute paths;
- dataset locations;
- log and checkpoint paths;
- container image and registry names;
- module-loading commands;
- CPU, GPU, memory, and wall-time requests;
- task-array ranges;
- environment variables;
- hard-coded experiment names;
- dependencies on other shell or Python scripts.

Do not commit credentials, access tokens, private registry secrets, or
restricted dataset locations.


## Attribution

These orchestration scripts were developed for the EPFL CS-433 group project by:

- S. Bernasconi
