#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.
# ==============================================================================
# SLAM-RF → TRAINING -> EVAL POSTPROC (Run:AI)
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

EXPT_IDS=(112005 212005 412005 )
MODEL_ARCHS=(unet3 )
LRS_FACTOR=(0.1 0.3)
P_RANGE=(1 1.5 2)
BCEW=(0.7 0.8 0.9)


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
for BCE in "${BCEW[@]}"; do
  for RANGE in "${P_RANGE[@]}"; do
    for EXPT in "${EXPT_IDS[@]}"; do
      ENHANCEMENT="$(deduce_enhancement "$EXPT")"
      DATASET="dataset_SLAM_${EXPT}"

      for ARCH in "${MODEL_ARCHS[@]}"; do
        for LRS_F in "${LRS_FACTOR[@]}"; do

          LRS_TAG="${LRS_F//./p}"          # 0.1 -> 0p1
          RANGE_TAG="${RANGE//./p}"   # 1.5 -> 1p5
          BCE_TAG="${BCE//./p}"        # 0.7 -> 0p7
          MODEL_NAME="fff${ARCH}-l${LRS_TAG}-p${RANGE_TAG}-b${BCE_TAG}-e${EXPT}"
          JOB_NAME="arttp-${MODEL_NAME}"
          LOGFILE="${MODEL_NAME}.log"

      
          echo "------------------------------------------------------------"
          echo "[RUN CONFIG]"
          echo "  user        = ${USER_FOLDER}"
          echo "  workdir     = ${WORKDIR}"
          echo "  dataset     = ${DATASET}"
          echo "  expt        = ${EXPT}"
          echo "  enhancement = ${ENHANCEMENT}"
          echo "  arch        = ${ARCH}"
          echo "  epochs      = ${EPOCHS}"
          echo "  lrs_factor  = ${LRS_F}"
          echo "  range_p     = ${RANGE}"
          echo "  bcew        = ${BCE}"
          echo "  model_name  = ${MODEL_NAME}"
          echo "  job_name    = ${JOB_NAME}"
          echo "------------------------------------------------------------"

          

          # ------------------------------------------------------------------
          # INNER COMMAND
          # ------------------------------------------------------------------

          read -r -d '' INNER_COMMAND_1 << EOF
cd "$WORKDIR"
python3 -u train_slam.py \
  --model-name "$MODEL_NAME" \
  --source "$DATASET" \
  --dt "${EPOCHS}epochs" \
  --range-p "$RANGE" \
  --expt "$EXPT" \
  --model "$ARCH" \
  --bcew "$BCE" \
  --epochs "$EPOCHS" \
  --enhancement "$ENHANCEMENT" \
  --shuffle-data \
  --lrs-factor "$LRS_F" \
2>&1 | tee "train_$LOGFILE"
EOF


          read -r -d '' INNER_COMMAND_2 << EOF
cd "$WORKDIR"
python3 -u test_slam.py \
  --model-name "$MODEL_NAME" \
  --expt "$EXPT" \
  --dt "${EPOCHS}epochs" \
  --source "$DATASET" \
2>&1 | tee "postprocess_$LOGFILE"
EOF

          read -r -d '' INNER_COMMAND_3 << EOF
cd "$WORKDIR/eval"
python3 -u postprocess_slam.py \
  --model-name "$MODEL_NAME" \
  --expt "$EXPT" \
  --dt "${EPOCHS}epochs" \
2>&1 | tee "test_$LOGFILE"
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
      --cpu-limit 10 \
      --memory 64G \
      --memory-limit 100G \
      --large-shm \
      --run-as-uid "$CUID" \
      --run-as-gid "$GID" \
      --project "course-cs-433-group03-${GASPAR}" \
      --existing-pvc claimname="$PVC_CLAIM",path="$SCRATCH_MNT" \
      --command -- /bin/bash -c "$INNER_COMMAND_1 && $INNER_COMMAND_2 && $INNER_COMMAND_3" \
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
  done
done
