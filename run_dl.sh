#!/usr/bin/env bash
set -euo pipefail

# Denoised runner: measure DER for a given audio file
# with DeepFilterNet3 denoising enabled.
# Usage: ./run_dl.sh <audio_file>

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

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python is required but not found in PATH." >&2
  exit 3
fi

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

echo "[run_dl] Measuring DER (denoised) for: $AUDIO_FILE (device: $DEVICE)"

$PY scripts/run_diar_experiment.py \
  --audio-file "$AUDIO_FILE" \
  --experiment "$EXPERIMENT" \
  --device "$DEVICE" \
  --denoise auto

DER_CSV="reports/der_metrics.csv"
if [ -f "$DER_CSV" ]; then
  echo "[run_dl] Latest DER entry:"
  tail -n 1 "$DER_CSV"
else
  echo "[run_dl] DER metrics CSV not found (expected at $DER_CSV)." >&2
fi

echo "[run_dl] Pred RTTM: diarization_output/pred_rttms/2.rttm"
echo "[run_dl] Log: logs/neMo_run_${EXPERIMENT}.log"

