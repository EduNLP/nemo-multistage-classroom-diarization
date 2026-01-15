#!/usr/bin/env bash
set -euo pipefail

# Simple runner: measure DER for a given audio file by
# invoking the project pipeline. Usage:
#   ./run.sh <audio_file>
# Optionally symlink/alias to `run` for: run <audio_file>

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

# Prefer python if available, else python3
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python is required but not found in PATH." >&2
  exit 3
fi

# Auto-detect device for Whisper/NeMo (cuda or cpu)
DEVICE=$($PY - <<'PY'
try:
    import torch
    print('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    print('cpu')
PY
)

EXPERIMENT="$(basename -- "$AUDIO_FILE")"
EXPERIMENT="${EXPERIMENT%.*}"

echo "[run] Measuring DER for: $AUDIO_FILE (device: $DEVICE)"

$PY scripts/run_diar_experiment.py \
  --audio-file "$AUDIO_FILE" \
  --experiment "$EXPERIMENT" \
  --device "$DEVICE" \
  --denoise none

# Summarize results
DER_CSV="reports/der_metrics.csv"
if [ -f "$DER_CSV" ]; then
  echo "[run] Latest DER entry:"
  tail -n 1 "$DER_CSV"
else
  echo "[run] DER metrics CSV not found (expected at $DER_CSV)." >&2
fi

echo "[run] Pred RTTM: diarization_output/pred_rttms/2.rttm"
echo "[run] Log: logs/neMo_run_${EXPERIMENT}.log"
