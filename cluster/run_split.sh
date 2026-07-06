#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.

# ======================================================================
# First SLAM-RF PIPELINE ( !PREPROCESSING → SPLIT !→ TRAINING)
# Executed as a single Run:AI job
#
# This script performs the following workflow:
#
#   2. train_test_split.py     → dataset construction
#
# All outputs, folder names, and log files dynamically depend on:
#       EXPERIMENT_ID
#       PREPROCESS_MODE
#       TRAIN_EPOCHS
# ======================================================================


# ----------------------------------------------------------------------
# Important Notes (WSL & File Format)
# ----------------------------------------------------------------------
# - The script must be executed inside WSL, or linux not Windows PowerShell.
#   Run:AI CLI and scratch-mounted paths only work correctly inside WSL.
#
# - Ensure the script uses LF line endings.
#   In VS Code: bottom-right corner → select "LF" → save the file again.
#
# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
#
#   chmod +x run_split.sh
#
#    echo "  ./run_split.sh --user <ste|moh|alex> --expt N \"
#    
#
# ======================================================================


SCRATCH_MNT="<SCRATCH_MOUNT>"
PVC_CLAIM="<PVC_CLAIM>"
IMAGE_NAME="registry.rcp.epfl.ch/cs-433-group03-radar/radar4k:latest"

#----------------------------------------------------------------------
# USER-CONFIGURABLE DEFAULT PARAMETERS (can be overridden by CLI)
# ----------------------------------------------------------------------

# [1][2][3][4][5][6]
#  │  │  │   └───┴─── threshold intensity xyz → x.yz
#  │  │  └─ threshold type: 1=none, 2=mag, 3=cfar
#  │  └─ binary intensity: 0/1
#  └─ preprocessing level: 1..6

#experiment ID convention used to distinguish between levels of preprocessing and intensity handling:
#e.g. none + binary + cfar thresh 0.05 = 113005
USER_FOLDER=""               # One of: ste / moh / alex (required)
EXPERIMENT_ID=""              # Numeric experiment identifier (required)

# following parameters will be inferred automatically from experiment ID
# PREPROCESS_MODE=""        # One of: none / low / high / vhigh / vvhigh / vvvhigh
# MAG_THRESHOLD=""             #  threshold value for lidar magnitude thresholding (float > 0)
# CFAR_THRESHOLD=""          # CFAR threshold value (float > 0)
# BINARY_INTENSITY=false     # binary intensity flag (true / false)
# ----------------------------------------------------------------------

# Valid options
VALID_USERS=("ste" "moh" "alex")


# ----------------------------------------------------------------------
# CLI ARGUMENT PARSING
# ----------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --user|-u)
            USER_FOLDER="$2"
            shift 2
            ;;
        --expt|-e)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
         *)
            echo "ERROR: Unknown argument '$1'"
            echo "Usage:"
            echo "  ./run_split.sh --user <ste|moh|alex> --expt N"
            exit 1
            ;;  
    esac
done

# ----------------------------------------------------------------------
# REQUIRED ARGUMENT CHECK (user and experiment ID are required)
# ----------------------------------------------------------------------
if [[ -z "$USER_FOLDER" ]]; then
    echo "ERROR: Missing required argument --user <ste|moh|alex>"
    exit 1
fi

if [[ -z "$EXPERIMENT_ID" ]]; then
    echo "ERROR: Missing required argument --expt <number>"
    exit 1
fi


# ----------------------------------------------------------------------
# VALUE VALIDATION
# ----------------------------------------------------------------------
if [[ ! " ${VALID_USERS[*]} " =~ " ${USER_FOLDER} " ]]; then
    echo "ERROR: Invalid --user '${USER_FOLDER}'"
    exit 1
fi

#check length of experiment name, must be conformal to 6-digit code
if [[ ! "$EXPERIMENT_ID" =~ ^[0-9]{6}$ ]]; then
  echo "ERROR: --expt must be a 6-digit numeric code"
  exit 1
fi

# ----------------------------------------------------------------------

# --- digit extraction ---
PREPROC_CODE=${EXPERIMENT_ID:0:1}
BIN_CODE=${EXPERIMENT_ID:1:1}
THR_CODE=${EXPERIMENT_ID:2:1}
THR_RAW=${EXPERIMENT_ID:3:3}

declare -A PREPROC_MAP=(
  [1]="none"
  [2]="low"
  [3]="high"
  [4]="vhigh"
  [5]="vvhigh"
  [6]="vvvhigh"
)

declare -A THRESHOLD_METHOD_MAP=(
  [1]="none"
  [2]="mag"
  [3]="cfar"
)

THRESHOLDING_METHOD="${THRESHOLD_METHOD_MAP[$THR_CODE]}"

if [[ -z "$THRESHOLDING_METHOD" ]]; then
  echo "Invalid thresholding method code: $THR_CODE" >&2
  exit 1
fi


PREPROCESS_MODE="${PREPROC_MAP[$PREPROC_CODE]}"

if [[ -z "$PREPROCESS_MODE" ]]; then
  echo "Invalid preprocessing code: $PREPROC_CODE" >&2
  exit 1
fi

case "$BIN_CODE" in
  0) BINARY_INTENSITY="false" ;;
  1) BINARY_INTENSITY="true" ;;
  *)
    echo "Invalid binary intensity code: $BIN_CODE" >&2
    exit 1 ;;
esac

MAG_THRESHOLD="none"
CFAR_THRESHOLD="none"

case "$THR_CODE" in
  1)
    # No thresholding: keep both values set to "none".
    ;;
  2)
    MAG_THRESHOLD="$(printf "%d.%02d" ${THR_RAW:0:1} ${THR_RAW:1:2})"
    ;;
  3)
    CFAR_THRESHOLD="$(printf "%d.%02d" ${THR_RAW:0:1} ${THR_RAW:1:2})"
    ;;
  *)
    echo "Invalid threshold type code: $THR_CODE" >&2
    exit 1 ;;
esac


echo "[INFO] Running pipeline for user: ${USER_FOLDER}"
echo "[INFO] Experiment ID: ${EXPERIMENT_ID}"
echo "[INFO] Preprocessing mode: ${PREPROCESS_MODE}"
echo "[INFO] Binary intensity: ${BINARY_INTENSITY}"
echo "[INFO] Magnitude threshold: ${MAG_THRESHOLD}"
echo "[INFO] CFAR threshold: ${CFAR_THRESHOLD}"


JOB_NAME="slam-split-${EXPERIMENT_ID}"

# ----------------------------------------------------------------------
# USER → UID / GID / GASPAR MAPPING (DETERMINISTIC)
# ----------------------------------------------------------------------
case "${USER_FOLDER}" in
  ste)
    CUID="<UID_STE>"
    GID="<GID>"
    GASPAR="<GASPAR_STE>"
    ;;
  moh)
    CUID="<UID_MOH>"
    GID="<GID>"
    GASPAR="<GASPAR_MOH>"
    ;;
  alex)
    CUID="<UID_ALEX>"
    GID="<GID>"
    GASPAR="<GASPAR_ALEX>"
    ;;
  *)
    echo "ERROR: No UID/GID mapping defined for user '${USER_FOLDER}'"
    exit 1
    ;;
esac

echo "[INFO] UID=${CUID}"
echo "[INFO] GID=${GID}"
echo "[INFO] GASPAR=${GASPAR}"


# ----------------------------------------------------------------------
# DERIVED PATHS (DO NOT EDIT)
# ----------------------------------------------------------------------

WORKDIR="${SCRATCH_MNT}/${USER_FOLDER}/radarhd"
CREATEDIR="${WORKDIR}/create_dataset"

# Output folders for the pipeline
PREPROCESSED_DATASET="processed_SLAM_${EXPERIMENT_ID}"
SPLIT_DATASET="dataset_SLAM_${EXPERIMENT_ID}"

# Log file names
LOG_PREPROC="preprocess_${EXPERIMENT_ID}.log"
LOG_SPLIT="split_${EXPERIMENT_ID}.log"
LOG_TRAIN="train_${EXPERIMENT_ID}.log"


# ----------------------------------------------------------------------
# OPTIONAL ARGUMENT CONSTRUCTION
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# DATASET DESCRIPTION STRING (always explicit, even if parameters are empty)
# ----------------------------------------------------------------------

DATASET_DESC="Experiment=${EXPERIMENT_ID} | \
preprocess=${PREPROCESS_MODE:-none} | \
method=${THRESHOLDING_METHOD:-none} | \
mag_threshold=${MAG_THRESHOLD:-none} | \
cfar_threshold=${CFAR_THRESHOLD:-none} | \
binary_intensity=${BINARY_INTENSITY}"

# ----------------------------------------------------------------------
# COMMAND EXECUTED *INSIDE* THE CONTAINER
# ----------------------------------------------------------------------
read -r -d '' INNER_COMMAND << EOF
echo "=== STEP 2: TRAIN/TEST SPLIT ==="
cd "$CREATEDIR"
python3 -u train_test_split.py \
    -fo "$USER_FOLDER" \
    --source "$PREPROCESSED_DATASET" \
    --dest "$SPLIT_DATASET" \
    -c "$DATASET_DESC" \
2>&1 | tee "$LOG_SPLIT"

echo "=== SPLITTING FINISHED. Sleeping forever... ==="
sleep infinity
EOF

echo "======= INNER COMMAND CONTENT ======="
printf "%s\n" "$INNER_COMMAND"
echo "====================================="

# ----------------------------------------------------------------------
# RUN:AI JOB SUBMISSION
# ----------------------------------------------------------------------

runai submit --name ${JOB_NAME} \
  --image ${IMAGE_NAME} \
  --node-pools default \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 10 \
  --cpu-limit 12 \
  --memory 300G \
  --memory-limit 350G \
  --large-shm \
  --run-as-uid ${CUID} \
  --run-as-gid ${GID} \
  --project course-cs-433-group03-${GASPAR} \
  --existing-pvc claimname=${PVC_CLAIM},path=${SCRATCH_MNT} \
  --command -- /bin/bash -c "${INNER_COMMAND}"

