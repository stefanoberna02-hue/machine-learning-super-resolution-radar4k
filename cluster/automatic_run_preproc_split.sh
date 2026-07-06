#!/bin/bash
#
# PUBLIC ARCHIVAL VERSION
# Infrastructure-specific identifiers and storage paths were replaced with
# placeholders. This script is retained to document the project workflow
# and is not expected to run without local adaptation.
# ==============================================================================
# AUTOMATED MULTI-EXPERIMENT LAUNCHER FOR run_preproc_split.sh
#
# This script:
#   - takes ONLY --user from CLI
#   - iterates over a predefined list of 6-digit experiment IDs
#   - decodes each experiment ID into preprocessing / threshold flags
#   - calls run_preproc_split.sh passing the correct CLI arguments
#
# ==============================================================================


# ------------------------------------------------------------------------------
# CLI (ONLY USER)
# ------------------------------------------------------------------------------

USER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user|-u)
      USER="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument '$1'"
      echo "Usage: ./automatic_run_preproc_split.sh --user <ste|moh|alex>"
      exit 1
      ;;
  esac
done

if [[ -z "${USER}" ]]; then
  echo "ERROR: --user must be specified"
  exit 1
fi

case "${USER}" in
  ste|moh|alex) ;;
  *)
    echo "ERROR: Invalid user '${USER}'"
    exit 1
    ;;
esac

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_preproc_split.sh"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "ERROR: Script not found or not executable:"
  echo "  ${PIPELINE_SCRIPT}"
  exit 1
fi

# ------------------------------------------------------------------------------
# EXPERIMENT IDS (EDIT ONLY THIS LIST)
# ------------------------------------------------------------------------------

EXPERIMENT_IDS=(
  112005
  212005
  312005
  412005
  512005
  612005
)

# ------------------------------------------------------------------------------
# DECODING FUNCTIONS
# ------------------------------------------------------------------------------

decode_preprocess() {
  case "$1" in
    1) echo "none" ;;
    2) echo "low" ;;
    3) echo "high" ;;
    4) echo "vhigh" ;;
    5) echo "vvhigh" ;;
    6) echo "vvvhigh" ;;
    *)
      echo "ERROR: Invalid preprocessing code '$1'"
      exit 1
      ;;
  esac
}

#we pass a float value from xyz -> x.yz
decode_threshold_value() {
  local xyz="$1"
  awk "BEGIN { printf \"%.2f\", ${xyz} / 100 }"
}

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------

echo "============================================================"
echo "----------- automatic_run_preproc_split.sh ------------"
echo "User: ${USER}"
echo "Experiments: ${EXPERIMENT_IDS[*]}"
echo "============================================================"

for EXPT in "${EXPERIMENT_IDS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Launching experiment ${EXPT}"
  echo "------------------------------------------------------------"

  EXPT_STR="$(printf "%06d" "${EXPT}")"

  A="${EXPT_STR:0:1}"
  B="${EXPT_STR:1:1}"
  C="${EXPT_STR:2:1}"
  DEF="${EXPT_STR:3:3}"

  PREPROCESS_MODE="$(decode_preprocess "${A}")"

  BINARY_INTENSITY=false
  [[ "${B}" == "1" ]] && BINARY_INTENSITY=true

  MAG_THRESHOLD=""
  CFAR_THRESHOLD=""
  NO_THRESHOLD=false

  case "${C}" in
    1)
      NO_THRESHOLD=true
      ;;
    2)
      MAG_THRESHOLD="$(decode_threshold_value "${DEF}")"
      ;;
    3)
      CFAR_THRESHOLD="$(decode_threshold_value "${DEF}")"
      ;;
    *)
      echo "ERROR: Invalid thresholding code '${C}' in experiment ${EXPT}"
      exit 1
      ;;
  esac

  # --------------------------------------------------------------------------
  # BUILD CLI ARGUMENTS
  # --------------------------------------------------------------------------

  ARGS=(--user "${USER}" --expt "${EXPT}" --preprocess "${PREPROCESS_MODE}")

  [[ "${BINARY_INTENSITY}" == true ]] && ARGS+=(--binary-intensity)
  [[ "${NO_THRESHOLD}" == true ]] && ARGS+=(--no-threshold)
  [[ -n "${MAG_THRESHOLD}" ]] && ARGS+=(--mag-threshold "${MAG_THRESHOLD}")
  [[ -n "${CFAR_THRESHOLD}" ]] && ARGS+=(--cfar-threshold "${CFAR_THRESHOLD}")

  echo "[INFO] Decoded parameters:"
  echo "  preprocess        = ${PREPROCESS_MODE}"
  echo "  binary_intensity  = ${BINARY_INTENSITY}"
  echo "  mag_threshold     = ${MAG_THRESHOLD:-none}"
  echo "  cfar_threshold    = ${CFAR_THRESHOLD:-none}"

  # --------------------------------------------------------------------------
  # EXECUTE PIPELINE SCRIPT (REAL CLI CALL)
  # --------------------------------------------------------------------------

  bash "${PIPELINE_SCRIPT}" "${ARGS[@]}"

  echo "Finished launching experiment ${EXPT}"
done

echo
echo "============================================================"
echo "All experiments submitted successfully"
echo "============================================================"
