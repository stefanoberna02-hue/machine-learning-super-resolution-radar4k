#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.
# ==============================================================================
# CROSS VALIDATION — SINGLE MODEL
#
# Runs:
#   - cross_val_single.py
#
# CONTRACT:
# - CLI matches EXACTLY train_slam.py
# - Only explicitly provided arguments are forwarded
# - Dataset must already exist: dataset_SLAM_<EXPT>
# ==============================================================================


SCRATCH_MNT="<SCRATCH_MOUNT>"
PVC_CLAIM="<PVC_CLAIM>"
IMAGE_NAME="registry.rcp.epfl.ch/cs-433-group03-radar/radar4k:latest"

# ------------------------------------------------------------------------------
# USER-PROVIDED PARAMETERS (empty unless passed via CLI)
# ------------------------------------------------------------------------------

USER_FOLDER=""
MODEL_NAME=""
EXPERIMENT_ID=""

MODEL_ARCHITECTURE=""
ENHANCEMENT=""
TRAIN_EPOCHS=""
SHUFFLE_DATA=true

OPTIMIZER=""
LEARNING_RATE=""
BCEW=""
ADAM_BETA1=""
ADAM_BETA2=""
LRS_FACTOR=""
LRS_PATIENCE=""
RANGE_P=""
RANGE_EPS=""
RANGE_CLIP=""
DEST_FOLDER=""

# ------------------------------------------------------------------------------
# VALID VALUES
# ------------------------------------------------------------------------------

VALID_USERS=("ste" "moh" "alex")
VALID_MODELS=("unet1" "unet2" "unet3" "unet4" "unet5" "unet6")
VALID_ENHANCEMENTS=("none" "low" "high" "vhigh" "vvhigh" "vvvhigh")

# ------------------------------------------------------------------------------
# CLI PARSING — SAME FLAGS AS train_slam.py
# ------------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder|-fo) USER_FOLDER="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --expt|-e) EXPERIMENT_ID="$2"; shift 2 ;;
    --model|-m) MODEL_ARCHITECTURE="$2"; shift 2 ;;
    --enhancement) ENHANCEMENT="$2"; shift 2 ;;
    --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --shuffle-data) SHUFFLE_DATA=true; shift ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --bcew) BCEW="$2"; shift 2 ;;
    --adam-beta1) ADAM_BETA1="$2"; shift 2 ;;
    --adam-beta2) ADAM_BETA2="$2"; shift 2 ;;
    --lrs-factor) LRS_FACTOR="$2"; shift 2 ;;
    --lrs-patience) LRS_PATIENCE="$2"; shift 2 ;;
    --range-p) RANGE_P="$2"; shift 2 ;;
    --range-eps) RANGE_EPS="$2"; shift 2 ;;
    --range-clip) RANGE_CLIP="$2"; shift 2 ;;
    --dest) DEST_FOLDER="$2"; shift 2 ;;
    *)
      echo "ERROR: Unknown argument '$1'"
      exit 1
      ;;
  esac
done

# ------------------------------------------------------------------------------
# REQUIRED ARGUMENTS
# ------------------------------------------------------------------------------

[[ -z "$USER_FOLDER" ]] && { echo "ERROR: --folder is required"; exit 1; }
[[ -z "$EXPERIMENT_ID" ]] && { echo "ERROR: --expt is required"; exit 1; }
[[ -z "$MODEL_NAME" ]] && { echo "ERROR: --model-name is required"; exit 1; }

# ------------------------------------------------------------------------------
# VALUE VALIDATION
# ------------------------------------------------------------------------------

if [[ ! " ${VALID_USERS[*]} " =~ " ${USER_FOLDER} " ]]; then
  echo "ERROR: Invalid folder '${USER_FOLDER}'"
  exit 1
fi

if [[ -n "$MODEL_ARCHITECTURE" ]] && [[ ! " ${VALID_MODELS[*]} " =~ " ${MODEL_ARCHITECTURE} " ]]; then
  echo "ERROR: Invalid model architecture '${MODEL_ARCHITECTURE}'"
  exit 1
fi

if [[ -n "$ENHANCEMENT" ]] && [[ ! " ${VALID_ENHANCEMENTS[*]} " =~ " ${ENHANCEMENT} " ]]; then
  echo "ERROR: Invalid enhancement '${ENHANCEMENT}'"
  exit 1
fi

# ------------------------------------------------------------------------------
# USER → UID / GID / PROJECT MAPPING
# ------------------------------------------------------------------------------

case "$USER_FOLDER" in
  ste)  CUID="<UID_STE>"; GID="<GID>"; GASPAR="<GASPAR_STE>" ;;
  moh)  CUID="<UID_MOH>"; GID="<GID>"; GASPAR="<GASPAR_MOH>" ;;
  alex) CUID="<UID_ALEX>"; GID="<GID>"; GASPAR="<GASPAR_ALEX>" ;;
esac

# ------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------

WORKDIR="${SCRATCH_MNT}/${USER_FOLDER}/radarhd"
DATASET="dataset_SLAM_${EXPERIMENT_ID}"
LOGFILE="crossval_${MODEL_NAME}_${EXPERIMENT_ID}.log"

# ------------------------------------------------------------------------------
# OPTIONAL ARGUMENT CONSTRUCTION (PASS ONLY IF SET)
# ------------------------------------------------------------------------------

ARGS=()

[[ -n "$MODEL_ARCHITECTURE" ]] && ARGS+=(--model "$MODEL_ARCHITECTURE")
[[ -n "$ENHANCEMENT" ]] && ARGS+=(--enhancement "$ENHANCEMENT")
[[ -n "$TRAIN_EPOCHS" ]] && ARGS+=(--epochs "$TRAIN_EPOCHS")
[[ "$SHUFFLE_DATA" == true ]] && ARGS+=(--shuffle-data)
[[ -n "$OPTIMIZER" ]] && ARGS+=(--optimizer "$OPTIMIZER")
[[ -n "$LEARNING_RATE" ]] && ARGS+=(--learning-rate "$LEARNING_RATE")
[[ -n "$BCEW" ]] && ARGS+=(--bcew "$BCEW")
[[ -n "$ADAM_BETA1" ]] && ARGS+=(--adam-beta1 "$ADAM_BETA1")
[[ -n "$ADAM_BETA2" ]] && ARGS+=(--adam-beta2 "$ADAM_BETA2")
[[ -n "$LRS_FACTOR" ]] && ARGS+=(--lrs-factor "$LRS_FACTOR")
[[ -n "$LRS_PATIENCE" ]] && ARGS+=(--lrs-patience "$LRS_PATIENCE")
[[ -n "$RANGE_P" ]] && ARGS+=(--range-p "$RANGE_P")
[[ -n "$RANGE_EPS" ]] && ARGS+=(--range-eps "$RANGE_EPS")
[[ -n "$RANGE_CLIP" ]] && ARGS+=(--range-clip "$RANGE_CLIP")
[[ -n "$DEST_FOLDER" ]] && ARGS+=(--dest "$DEST_FOLDER")

# ------------------------------------------------------------------------------
# INNER COMMAND
# ------------------------------------------------------------------------------

read -r -d '' INNER_COMMAND << EOF
cd "$WORKDIR"
python3 -u cross_val_single.py \
  -fo "$USER_FOLDER" \
  --model-name "$MODEL_NAME" \
  --source "$DATASET" \
  --expt "$EXPERIMENT_ID" \
  ${ARGS[@]} \
2>&1 | tee "$LOGFILE"

sleep infinity
EOF

JOB_NAME="cv-${MODEL_NAME}-${EXPERIMENT_ID}"


############################
#     CHECK
############################
echo "============================================================"
echo "[INFO] CROSS VALIDATION — SINGLE MODEL"
echo "[INFO] User folder      : ${USER_FOLDER}"
echo "[INFO] Model name       : ${MODEL_NAME}"
echo "[INFO] Experiment ID    : ${EXPERIMENT_ID}"
echo "[INFO] Dataset          : ${DATASET}"
echo "[INFO] Model arch       : ${MODEL_ARCHITECTURE:-<default>}"
echo "[INFO] Enhancement      : ${ENHANCEMENT:-<default>}"
echo "[INFO] Epochs           : ${TRAIN_EPOCHS:-<default>}"
echo "[INFO] Shuffle data     : ${SHUFFLE_DATA}"
echo "[INFO] Optimizer        : ${OPTIMIZER:-<default>}"
echo "[INFO] Learning rate    : ${LEARNING_RATE:-<default>}"
echo "[INFO] BCE weight       : ${BCEW:-<default>}"
echo "[INFO] Adam beta1       : ${ADAM_BETA1:-<default>}"
echo "[INFO] Adam beta2       : ${ADAM_BETA2:-<default>}"
echo "[INFO] LRS factor       : ${LRS_FACTOR:-<default>}"
echo "[INFO] LRS patience     : ${LRS_PATIENCE:-<default>}"
echo "[INFO] Range p          : ${RANGE_P:-<default>}"
echo "[INFO] Range eps        : ${RANGE_EPS:-<default>}"
echo "[INFO] Range clip       : ${RANGE_CLIP:-<default>}"
echo "[INFO] Workdir          : ${WORKDIR}"
echo "[INFO] Log file         : ${LOGFILE}"
echo "[INFO] Job name         : ${JOB_NAME}"
echo "============================================================"

# ------------------------------------------------------------------------------
# RUN:AI SUBMISSION
# ------------------------------------------------------------------------------

runai submit \
  --name "$JOB_NAME" \
  --image "$IMAGE_NAME" \
  --node-pools default \
  --gpu 1 \
  --cpu 8 \
  --cpu-limit 10 \
  --memory 64G \
  --memory-limit 100G \
  --large-shm \
  --run-as-uid "$CUID" \
  --run-as-gid "$GID" \
  --project "course-cs-433-group03-${GASPAR}" \
  --existing-pvc claimname="$PVC_CLAIM",path="$SCRATCH_MNT" \
  --command -- /bin/bash -c "$INNER_COMMAND"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELETE_FILE="${SCRIPT_DIR}/delete_jobs_all"

touch "${DELETE_FILE}"

grep -qxF "runai delete job ${JOB_NAME}" "${DELETE_FILE}" || \
  echo "runai delete job ${JOB_NAME}" >> "${DELETE_FILE}"
