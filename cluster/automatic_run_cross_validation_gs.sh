#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.
# ==============================================================================
# CROSS VALIDATION — GRID SEARCH (MULTI-JOB)
#
# INPUT:
#   --folder <ste|moh|alex>
#
# HARD-CODED:
#   - experiment id
#   - optimizer
#   - shuffle
#   - architecture
#   - num_epochs
#
# GRID:
#   - learning rate
#   - bce weight
#   - range p
#   - adam beta1 / beta2
#   - lrs factor
# ==============================================================================

# ------------------------------------------------------------------------------
# HARD-CODED PARAMETERS (EDIT HERE)
# ------------------------------------------------------------------------------

EXPERIMENT_ID=112005
MODEL_ARCHITECTURE="unet3"
OPTIMIZER="adam"
SHUFFLE_DATA=true
NUM_EPOCHS=100


declare -A EXPT_ENHANCEMENT_MAP=(
  ["1"]="none"
  ["2"]="low"
  ["3"]="high"
  ["4"]="vhigh"
  ["5"]="vvhigh"
  ["6"]="vvvhigh"
)
# ------------------------------------------------------------------------------
# INPUT (only user)
# ------------------------------------------------------------------------------

USER_FOLDER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder|-fo)
      USER_FOLDER="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument '$1'"
      exit 1
      ;;
  esac
done

[[ -z "$USER_FOLDER" ]] && { echo "ERROR: --folder is required"; exit 1; }

# ------------------------------------------------------------------------------
# GRID VALUES (EDIT FREELY)
# ------------------------------------------------------------------------------

LEARNING_RATES=(1e-3 1e-4 5e-4)
BCE_WEIGHTS=(0.7 0.8 0.9)
RANGE_PS=(0.0 1.5 1.75)
ADAM_BETA1S=(0.9)
ADAM_BETA2S=(0.999)
LRS_FACTORS=(0.1 0.3)

# ------------------------------------------------------------------------------
# SANITY
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_SINGLE="${SCRIPT_DIR}/run_cross_validation_single.sh"

[[ ! -x "$SCRIPT_SINGLE" ]] && {
  echo "ERROR: ${SCRIPT_SINGLE} not found or not executable"
  exit 1
}

if [[ ! "$EXPERIMENT_ID" =~ ^[0-9]{6}$ ]]; then
  echo "ERROR: --expt must be a 6-digit code"
  exit 1
fi

EXPT_STR="${EXPERIMENT_ID}"
EXPT_FIRST_DIGIT="${EXPT_STR:0:1}"
ENHANCEMENT="${EXPT_ENHANCEMENT_MAP[$EXPT_FIRST_DIGIT]}"

# ------------------------------------------------------------------------------
# GRID LOOP
# ------------------------------------------------------------------------------

TOTAL=0

for LR in "${LEARNING_RATES[@]}"; do
for BCEW in "${BCE_WEIGHTS[@]}"; do
for RP in "${RANGE_PS[@]}"; do
for B1 in "${ADAM_BETA1S[@]}"; do
for B2 in "${ADAM_BETA2S[@]}"; do
for LRS in "${LRS_FACTORS[@]}"; do

  # --------------------------------------------------------------------------
  # MODEL NAME (Run:AI-safe, deterministic, no underscores)
  # --------------------------------------------------------------------------

  LR_TAG=$(echo "$LR" | sed 's/\./p/g' | sed 's/e-/em/g')
  BCE_TAG=$(echo "$BCEW" | sed 's/\./p/g')
  RP_TAG=$(echo "$RP" | sed 's/\./p/g')
  LRS_TAG=$(echo "$LRS" | sed 's/\./p/g')

  CFG_HASH=$(echo "${LR}_${BCEW}_${RP}_${LRS}" | md5sum | cut -c1-6)
  MODEL_NAME="gscv${MODEL_ARCHITECTURE}-${CFG_HASH}"

  echo "============================================================"
  echo "[GRID] Launching automatic CV gs job"
  echo "[GRID] Model name : ${MODEL_NAME}"
  echo "[GRID] lr=${LR} bcew=${BCEW} range_p=${RP} beta1=${B1} beta2=${B2} lrs_factor=${LRS}"
  echo "============================================================"
  if [[ "$RP" != "0.0" && "$RP" != "0" ]]; then
    RANGE_P_ARG="--range-p ${RP}"
  else
    RANGE_P_ARG=""
  fi

  "${SCRIPT_SINGLE}" \
    --folder "${USER_FOLDER}" \
    --model-name "${MODEL_NAME}" \
    --expt "${EXPERIMENT_ID}" \
    --model "${MODEL_ARCHITECTURE}" \
    --enhancement "${ENHANCEMENT}" \
    --epochs "${NUM_EPOCHS}" \
    $( [[ "${SHUFFLE_DATA}" == true ]] && echo "--shuffle-data" ) \
    --optimizer "${OPTIMIZER}" \
    --learning-rate "${LR}" \
    --bcew "${BCEW}" \
    --adam-beta1 "${B1}" \
    --adam-beta2 "${B2}" \
    --lrs-factor "${LRS}" \
    ${RANGE_P_ARG}
  
  sleep 0.2
  TOTAL=$((TOTAL + 1))

done
done
done
done
done
done

echo "============================================================"
echo "[GRID] Submitted ${TOTAL} cross-validation jobs"
echo "============================================================"





