#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.
# ==============================================================================
# GRID SEARCH — RadarHD / SLAM-RF (training only)
#
# Sweeps:
#   - optimizer
#   - learning rate
#   - adam beta1 / beta2
#   - LR scheduler factor
#   - LR scheduler patience
#   - range-eps (range-weighted BCE)
#
# Fixed:
#   - preprocessing (derived from expt)
#   - model architecture
#   - channels (history)
#   - epochs
#
# Required CLI:
#   --user <ste|moh|alex>
#   --expt <6-digit experiment id>
# ==============================================================================

# ==============================================================================
# GRID SEARCH — RadarHD / SLAM-RF Training (Run:AI)
#
# This script launches a large-scale hyperparameter grid search for the
# RadarHD SLAM-RF training pipeline on the Run:AI cluster.
#
# The script submits one independent Run:AI job per hyperparameter combination,
# each executing `train_slam.py` inside a containerized environment.
#
# Fixed components:
#   - Dataset preprocessing and augmentation are inferred automatically from
#     the experiment ID (--expt) and MUST match the dataset on disk.
#   - Model architecture, input channels (history), and number of epochs are fixed.
#
# Hyperparameters swept in the grid:
#   - Optimizer type (adam, adamw)
#   - Learning rate
#   - Adam beta1 / beta2 coefficients
#   - ReduceLROnPlateau scheduler factor and patience
#   - Range-weighted BCE epsilon (range-eps)
#
# Required arguments:
#   --user <ste|moh|alex>     User identifier (maps to UID/GID and Run:AI project)
#   --expt <XXXXXX>          6-digit experiment ID (defines dataset + preprocessing)
#   --gs   <N>               Grid search identifier (used in job naming)
#
# Example usage:
#   ./run_training_gs.sh --user ste --expt 112005 --gs 01
#
# ============================================================================== 

SCRATCH_MNT="<SCRATCH_MOUNT>"
PVC_CLAIM="<PVC_CLAIM>"
IMAGE_NAME="registry.rcp.epfl.ch/cs-433-group03-radar/radar4k:latest"



# ------------------------------------------------------------------------------
# FIXED TRAINING PARAMETERS
# ------------------------------------------------------------------------------

MODEL_ARCH="unet3"
TRAIN_EPOCHS=100

# EXPT first digit → enhancement
# 1 = none
# 2 = low
# 3 = high
# 4 = vhigh
# 5 = vvhigh
# 6 = vvvhigh

declare -A EXPT_ENHANCEMENT_MAP=(
  ["1"]="none"
  ["2"]="low"
  ["3"]="high"
  ["4"]="vhigh"
  ["5"]="vvhigh"
  ["6"]="vvvhigh"
)



# ------------------------------------------------------------------------------
# GRID DEFINITION (EDIT HERE)
# ------------------------------------------------------------------------------

OPTIMIZERS=("adam" )

LR_LIST=( 1e-4)
BETA1_LIST=( 0.95)
BETA2_LIST=( 0.999)

LRS_FACTOR_LIST=( 0.5)
LRS_PATIENCE_LIST=(5 )

RANGE_EPS_LIST=(0.02 )
# OPTIMIZERS=("adam" "adamw")

# LR_LIST=(1e-3 5e-4 1e-4)
# BETA1_LIST=(0.85 0.9 0.95)
# BETA2_LIST=(0.99 0.999)

# LRS_FACTOR_LIST=(0.1 0.2 0.5)
# LRS_PATIENCE_LIST=(5 10 20)

# RANGE_EPS_LIST=(0.02 0.05 0.1)

# ------------------------------------------------------------------------------
# REQUIRED CLI
# ------------------------------------------------------------------------------

USER=""
EXPT=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --user|-u) USER="$2"; shift 2 ;;
    --expt|-e) EXPT="$2"; shift 2 ;;
    --gs|-g)
            GS_ID="$2"
            shift 2
            ;;
    *)
      echo "Unknown argument $1"
      exit 1
      ;;
  esac
done

[[ -z "$USER" ]] && { echo "ERROR: --user required"; exit 1; }
[[ -z "$EXPT" ]] && { echo "ERROR: --expt required"; exit 1; }
[[ -z "$GS_ID" ]] && { echo "ERROR: --gs is required"; exit 1; }


if [[ ! "$EXPT" =~ ^[0-9]{6}$ ]]; then
  echo "ERROR: --expt must be a 6-digit code"
  exit 1
fi

EXPT_STR="${EXPT}"
EXPT_FIRST_DIGIT="${EXPT_STR:0:1}"
ENHANCEMENT="${EXPT_ENHANCEMENT_MAP[$EXPT_FIRST_DIGIT]}"

if [[ -z "$ENHANCEMENT" ]]; then
  echo "ERROR: No enhancement mapping for experiment ${EXPT}"
  exit 1
fi

echo "[INFO] Running training grid search for user: ${USER}"
echo "[INFO] Grid search ID: ${GS_ID}"
echo "[INFO] Experiment ID: ${EXPT}"
echo "[INFO] Model architecture: ${MODEL_ARCH}"
echo "[INFO] Training epochs: ${TRAIN_EPOCHS}"
echo "[INFO] Enhancement assumed: ${ENHANCEMENT}"

# ------------------------------------------------------------------------------
# USER → UID / PROJECT
# ------------------------------------------------------------------------------

case "$USER" in
  ste)
    CUID="<UID_STE>"; GID="<GID>"; GASPAR="<GASPAR_STE>" ;;
  moh)
    CUID="<UID_MOH>"; GID="<GID>"; GASPAR="<GASPAR_MOH>" ;;
  alex)
    CUID="<UID_ALEX>"; GID="<GID>"; GASPAR="<GASPAR_ALEX>" ;;
  *)
    echo "ERROR: invalid user"
    exit 1 ;;
esac

echo "[INFO] UID=${CUID}"
echo "[INFO] GID=${GID}"
echo "[INFO] GASPAR=${GASPAR}"


WORKDIR="${SCRATCH_MNT}/${USER}/radarhd"
DATASET="dataset_SLAM_${EXPT}"

#used to avoid dots and dashes and unwanted behaviour in job name
normalize() {
  echo "$1" | sed 's/\./p/g; s/-/m/g'
}


# ------------------------------------------------------------------------------
# GRID SEARCH
# ------------------------------------------------------------------------------

for OPT in "${OPTIMIZERS[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for B1 in "${BETA1_LIST[@]}"; do
      for B2 in "${BETA2_LIST[@]}"; do
        for F in "${LRS_FACTOR_LIST[@]}"; do
          for P in "${LRS_PATIENCE_LIST[@]}"; do
            for EPS in "${RANGE_EPS_LIST[@]}"; do
              LR_TAG=$(normalize "${LR}")
              B1_TAG=$(normalize "${B1}")
              B2_TAG=$(normalize "${B2}")
              F_TAG=$(normalize "${F}")
              P_TAG=$(normalize "${P}")
              EPS_TAG=$(normalize "${EPS}")

              MODEL_NAME="${OPT}-lr${LR_TAG}-b1${B1_TAG}-b2${B2_TAG}-f${F_TAG}-p${P_TAG}-eps${EPS_TAG}"
              JOB_NAME="gs-${GS_ID}-${EXPT}-${MODEL_NAME}"

              read -r -d '' INNER_COMMAND << EOF
cd "${WORKDIR}"
python3 -u train_slam.py \
  --folder "${USER}" \
  --model-name "${MODEL_NAME}" \
  --expt "${EXPT}" \
  --source "${DATASET}" \
  --model "${MODEL_ARCH}" \
  --epochs ${TRAIN_EPOCHS} \
  --optimizer "${OPT}" \
  --learning-rate ${LR} \
  --adam-beta1 ${B1} \
  --adam-beta2 ${B2} \
  --lrs-factor ${F} \
  --lrs-patience ${P} \
  --range-eps ${EPS} \
  --enhancement "${ENHANCEMENT}" \
2>&1 | tee "train_${MODEL_NAME}.log"
sleep infinity
EOF

              echo "[INFO] Inner command: ${INNER_COMMAND}"
              echo "[SUBMIT] ${JOB_NAME}"

              runai submit --name "${JOB_NAME}" \
                --image "${IMAGE_NAME}" \
                --node-pools default \
                --gpu 1 \
                --cpu 4 \
                --cpu-limit 8 \
                --memory 256G \
                --memory-limit 300G \
                --large-shm \
                --run-as-uid ${CUID} \
                --run-as-gid ${GID} \
                --project course-cs-433-group03-${GASPAR} \
                --existing-pvc claimname=${PVC_CLAIM},path=${SCRATCH_MNT} \
                --command -- /bin/bash -c "${INNER_COMMAND}"

            done
          done
        done
      done
    done
  done
done





























