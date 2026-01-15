#!/usr/bin/env bash
set -euo pipefail

# Runner focused on measuring DER for a given audio file
# using the same deep-learning preprocessing pipeline as run_dl.sh.
# Usage: ./run_dl_der.sh <audio_file>

usage() {
  echo "Usage: $0 <audio_file>" >&2
}

if [ ${#} -lt 1 ]; then
  usage
  exit 1
fi

AUDIO_FILE="$1"
if [ ! -f "$AUDIO_FILE" ]; then
  echo "Audio file not found: $AUDIO_FILE" >&2
  exit 2
fi

# Resolve python
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python is required but not found in PATH." >&2
  exit 3
fi

# Detect device (cuda/cpu)
DEVICE=$($PY - <<'PY'
try:
    import torch
    print('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    print('cpu')
PY
)

EXPERIMENT="$(basename -- "$AUDIO_FILE")"
EXPERIMENT="${EXPERIMENT%.*}_dl"

echo "[run_dl_der] Measuring DER for: $AUDIO_FILE (device: $DEVICE)"

$PY scripts/run_diar_experiment.py \
  --audio-file "$AUDIO_FILE" \
  --experiment "$EXPERIMENT" \
  --device "$DEVICE" \
  --denoise auto

DER_CSV="reports/der_metrics.csv"
if [ -f "$DER_CSV" ]; then
  echo "[run_dl_der] Latest DER entry:"
  tail -n 1 "$DER_CSV"
else
  echo "[run_dl_der] DER metrics CSV not found (expected at $DER_CSV)." >&2
fi

# Helpful pointers for outputs of this run
PRED_RTTM="diarization_output/pred_rttms/${EXPERIMENT}.rttm"
LOG_PATH="logs/neMo_run_${EXPERIMENT}.log"
echo "[run_dl_der] Pred RTTM: $PRED_RTTM"
echo "[run_dl_der] Log: $LOG_PATH"

