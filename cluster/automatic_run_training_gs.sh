#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.
# ==============================================================================
# SLAM-RF → TRAINING GRID (Run:AI)
#
# Grid over:
#   - EXPT_ID  → enhancement deduced
#   - model architecture
#   - lrs-factor
#
# train_slam.py CLI respected EXACTLY
# ==============================================================================


SCRATCH_MNT="<SCRATCH_MOUNT>"
PVC_CLAIM="<PVC_CLAIM>"
IMAGE_NAME="registry.rcp.epfl.ch/cs-433-group03-radar/radar4k:latest"

# ------------------------------------------------------------------------------
# FIXED USER
# ------------------------------------------------------------------------------

USER_FOLDER="moh"

# ------------------------------------------------------------------------------
# GRID DEFINITION
# ------------------------------------------------------------------------------

EXPT_IDS=(112005 212005 312005 412005 512005 612005)
MODEL_ARCHS=(unet3 unet6)
LRS_FACTOR=(0.1 0.3)

# ------------------------------------------------------------------------------
# FIXED TRAINING PARAMS
# ------------------------------------------------------------------------------

EPOCHS=130

#SHUFFLE_DATA=true hardcoded in submit job

# ------------------------------------------------------------------------------
# USER → UID / GID / PROJECT
# ------------------------------------------------------------------------------

case "$USER_FOLDER" in
  ste)  CUID="<UID_STE>"; GID="<GID>"; GASPAR="<GASPAR_STE>" ;;
  moh)  CUID="<UID_MOH>"; GID="<GID>"; GASPAR="<GASPAR_MOH>" ;;
  alex) CUID="<UID_ALEX>"; GID="<GID>"; GASPAR="<GASPAR_ALEX>" ;;
  *) echo "Invalid user"; exit 1 ;;
esac

WORKDIR="${SCRATCH_MNT}/${USER_FOLDER}/radarhd"

# ------------------------------------------------------------------------------
# FUNCTION: EXPT → ENHANCEMENT
# ------------------------------------------------------------------------------

deduce_enhancement () {
  local expt="$1"
  local first_digit="${expt:0:1}"

  case "$first_digit" in
    1) echo "none" ;;
    2) echo "low" ;;
    3) echo "high" ;;
    4) echo "vhigh" ;;
    5) echo "vvhigh" ;;
    6) echo "vvvhigh" ;;
    *)
      echo "ERROR: Cannot deduce enhancement from EXPT=${expt}" >&2
      exit 1
      ;;
  esac
}

# ------------------------------------------------------------------------------
# GRID LOOP
# ------------------------------------------------------------------------------

for EXPT in "${EXPT_IDS[@]}"; do

  ENHANCEMENT="$(deduce_enhancement "$EXPT")"
  DATASET="dataset_SLAM_${EXPT}"

  for ARCH in "${MODEL_ARCHS[@]}"; do
    for LRS_F in "${LRS_FACTOR[@]}"; do

      LRS_TAG="${LRS_F//./p}"          # 0.1 -> 0p1
      MODEL_NAME="${ARCH}-lrsf${LRS_TAG}-expt${EXPT}"
      JOB_NAME="slam-${MODEL_NAME}"
      LOGFILE="train_${MODEL_NAME}.log"

      echo "============================================================"
      echo "[SUBMIT]"
      echo "  expt        = ${EXPT}"
      echo "  enhancement = ${ENHANCEMENT}"
      echo "  arch        = ${ARCH}"
      echo "  lrs factor     = ${LRS_F}"
      echo "============================================================"

      

      # ------------------------------------------------------------------
      # INNER COMMAND
      # ------------------------------------------------------------------

      read -r -d '' INNER_COMMAND << EOF
cd "$WORKDIR"
python3 -u train_slam.py \
  -fo "$USER_FOLDER" \
  --model-name "$MODEL_NAME" \
  --source "$DATASET" \
  --expt "$EXPT" \
  --model "$ARCH" \
  --epochs "$EPOCHS" \
  --enhancement "$ENHANCEMENT" \
  --shuffle-data \
  --lrs-factor "$LRS_F" \
2>&1 | tee "$LOGFILE"

sleep infinity
EOF

      # ------------------------------------------------------------------
      # RUN:AI SUBMISSION
      # ------------------------------------------------------------------
      OUTPUT=$(runai submit \
    --name "$JOB_NAME" \
    --image "$IMAGE_NAME" \
    --node-pools default \
    --gpu 1 \
    --cpu 4 \
    --cpu-limit 8 \
    --memory 64G \
    --memory-limit 100G \
    --large-shm \
    --run-as-uid "$CUID" \
    --run-as-gid "$GID" \
    --project "course-cs-433-group03-${GASPAR}" \
    --existing-pvc claimname="$PVC_CLAIM",path="$SCRATCH_MNT" \
    --command -- /bin/bash -c "$INNER_COMMAND" \
    2>&1
  )
  RC=$?

  echo "$OUTPUT"

  if [ $RC -eq 0 ]; then
    echo "[OK] ${JOB_NAME}"
  else
    echo "[FAIL] ${JOB_NAME}" >&2
  fi

    done
  done
done
