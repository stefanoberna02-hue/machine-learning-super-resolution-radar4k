#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.

# ======================================================================
# This script performs the following workflow:
#   1. sync_slam_rf.py         → preprocessing of raw radar/lidar data
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
#   chmod +x run_preproc_split.sh
#
#    echo "  ./run_preproc_split.sh --user <ste|moh|alex> --expt N \
#    [--preprocess none|low|high|vhigh|vvhigh|vvvhigh] \
#    [--mag-threshold N | --cfar-threshold N] \
#    [--binary-intensity]"
#
# ======================================================================

#example:
#  ./run_preproc_split.sh --user ste --binary-intensity --preprocess none --mag-threshold 0.01 --expt 112001

SCRATCH_MNT="<SCRATCH_MOUNT>"
PVC_CLAIM="<PVC_CLAIM>"
IMAGE_NAME="registry.rcp.epfl.ch/cs-433-group03-radar/radar4k:latest"

#----------------------------------------------------------------------
# USER-CONFIGURABLE DEFAULT PARAMETERS (can be overridden by CLI)
# ----------------------------------------------------------------------

#experiment ID convention used to distinguish between levels of preprocessing and intensity handling:
#naming convention is: preprocessing: 1-6 (none->vvvhigh), intensity handling: 0 or 1 , thresholding method: 1 for none 2 for magnitude thresh and 3 for cfar, followed by the value used for thresholding if applicable
#e.g. none + binary + mag 0.05 = 112005,./
USER_FOLDER=""               # One of: ste / moh / alex (required)
EXPERIMENT_ID=""              # Numeric experiment identifier (required)
PREPROCESS_MODE=""        # One of: none / low / high / vhigh / vvhigh / vvvhigh
MAG_THRESHOLD=""             #  threshold value for lidar magnitude thresholding (float > 0)
CFAR_THRESHOLD=""          # CFAR threshold value (float > 0)
BINARY_INTENSITY=false     # binary intensity flag (true / false)
NO_THRESHOLD=false     # binary flag for no-threshold (true / false), when True overwrites others

# ----------------------------------------------------------------------

# Valid options
VALID_USERS=("ste" "moh" "alex")
VALID_PREPROC=("none" "low" "high" "vhigh" "vvhigh" "vvvhigh")



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
        --preprocess|-p)
            PREPROCESS_MODE="$2"
            shift 2
            ;;
        --mag-threshold|-m)
            MAG_THRESHOLD="$2"
            shift 2
            ;;  
        --cfar-threshold|-c)
            CFAR_THRESHOLD="$2"
            shift 2
            ;;
        --no-threshold|-n)
            NO_THRESHOLD=true
            shift 1
            ;;
        --binary-intensity|-b)
            BINARY_INTENSITY=true
            shift 1
            ;;
         *)
            echo "ERROR: Unknown argument '$1'"
            echo "Usage:"
            echo "  ./run_preproc_split.sh --user <ste|moh|alex> --expt N \
                [--preprocess none|low|high|vhigh|vvhigh|vvvhigh] \
                [--mag-threshold N | --cfar-threshold N] \
                [--binary-intensity]"
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

#validating preprocessing only when passed by user
if [[ -n "$PREPROCESS_MODE" ]] && [[ ! " ${VALID_PREPROC[*]} " =~ " ${PREPROCESS_MODE} " ]]; then
    echo "ERROR: Invalid preprocess mode '${PREPROCESS_MODE}'"
    exit 1
fi


if [[ -n "$MAG_THRESHOLD" && -n "$CFAR_THRESHOLD" ]]; then
    echo "ERROR: --mag-threshold and --cfar-threshold are mutually exclusive"
    exit 1
fi

# MAG_THRESHOLD: float > 0, validated only when passed by user
if [[ -n "$MAG_THRESHOLD" ]]; then
    if ! [[ "$MAG_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]] || \
       awk "BEGIN{exit !($MAG_THRESHOLD > 0)}"; then
        : # ok
    else
        echo "ERROR: --mag-threshold must be a float > 0"
        exit 1
    fi
fi
# CFAR_THRESHOLD: float > 0, validated only when passed by user
if [[ -n "$CFAR_THRESHOLD" ]]; then
    if ! [[ "$CFAR_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]] || \
       awk "BEGIN{exit !($CFAR_THRESHOLD > 0)}"; then
        :
    else
        echo "ERROR: --cfar-threshold must be a float > 0"
        exit 1
    fi
fi
if [[ "$NO_THRESHOLD" == true && ( -n "$MAG_THRESHOLD" || -n "$CFAR_THRESHOLD" ) ]]; then
  echo "ERROR: --no-threshold cannot be combined with --mag-threshold or --cfar-threshold"
  exit 1
fi


echo "[INFO] Running pipeline for user: ${USER_FOLDER}"
echo "[INFO] Experiment ID: ${EXPERIMENT_ID}"
echo "[INFO] Preprocess mode: ${PREPROCESS_MODE}"
echo "[INFO] Magnitude thresholding value: ${MAG_THRESHOLD}"
echo "[INFO] CFAR thresholding value: ${CFAR_THRESHOLD}"
echo "[INFO] NO threshold method: ${NO_THRESHOLD}"
echo "[INFO] Binary intensity enabled: ${BINARY_INTENSITY}"


JOB_NAME="slam-preproc-split-${EXPERIMENT_ID}"

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
PREPROCESS_ARG=""
[[ -n "$PREPROCESS_MODE" ]] && PREPROCESS_ARG="--enhancement $PREPROCESS_MODE"

MAG_THRESHOLD_ARG=""
[[ -n "$MAG_THRESHOLD" ]] && MAG_THRESHOLD_ARG="--mag-threshold $MAG_THRESHOLD"

CFAR_THRESHOLD_ARG=""
[[ -n "$CFAR_THRESHOLD" ]] && CFAR_THRESHOLD_ARG="--cfar-threshold $CFAR_THRESHOLD"

NO_THRESHOLD_ARG=""
[[ "$NO_THRESHOLD" == true ]] && NO_THRESHOLD_ARG="--no-threshold"

BINARY_INTENSITY_ARG=""
[[ "$BINARY_INTENSITY" == true ]] && BINARY_INTENSITY_ARG="--binary-intensity"

# ----------------------------------------------------------------------
# DATASET DESCRIPTION STRING (always explicit, even if parameters are empty)
# ----------------------------------------------------------------------

DATASET_DESC="Experiment=${EXPERIMENT_ID} | \
preprocess=${PREPROCESS_MODE:-none} | \
mag_threshold=${MAG_THRESHOLD:-none} | \
cfar_threshold=${CFAR_THRESHOLD:-none} | \
binary_intensity=${BINARY_INTENSITY}"

# ----------------------------------------------------------------------
# COMMAND EXECUTED *INSIDE* THE CONTAINER
# ----------------------------------------------------------------------
read -r -d '' INNER_COMMAND << EOF
cd "$CREATEDIR"

echo "=== STEP 1: PREPROCESSING ==="
python3 -u sync_slam_rf.py \
    -fo "$USER_FOLDER" \
    --out-folder "$PREPROCESSED_DATASET" \
    $PREPROCESS_ARG \
    $MAG_THRESHOLD_ARG \
    $CFAR_THRESHOLD_ARG \
    $BINARY_INTENSITY_ARG \
    $NO_THRESHOLD_ARG \
2>&1 | tee "$LOG_PREPROC"

echo "=== STEP 2: TRAIN/TEST SPLIT ==="
cd "$CREATEDIR"
python3 -u train_test_split.py \
    -fo "$USER_FOLDER" \
    --source "$PREPROCESSED_DATASET" \
    --dest "$SPLIT_DATASET" \
    -c "$DATASET_DESC" \
2>&1 | tee "$LOG_SPLIT"

echo "=== PREPROCESSING AND SPLITTING FINISHED. Ending job... ==="
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
  --cpu 14 \
  --cpu-limit 16 \
  --memory 300G \
  --memory-limit 350G \
  --large-shm \
  --run-as-uid ${CUID} \
  --run-as-gid ${GID} \
  --project course-cs-433-group03-${GASPAR} \
  --existing-pvc claimname=${PVC_CLAIM},path=${SCRATCH_MNT} \
  --command -- /bin/bash -c "${INNER_COMMAND}"

