#!/usr/bin/env bash
set -euo pipefail

# Run diarization for every audio in classbank_audio_data/audio
# twice per file: with DeepFilterNet3 denoising (dl) and without (nodl).
# Appends DER entries to reports/der_metrics.csv via scripts/run_diar_experiment.py.

usage() {
  echo "Usage: $0" >&2
}

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

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
AUDIO_DIR="$ROOT_DIR/classbank_audio_data/audio"
RTTM_DIR="$ROOT_DIR/classbank_audio_data/rttm"

if [ ! -d "$AUDIO_DIR" ]; then
  echo "Audio directory not found: $AUDIO_DIR" >&2
  exit 1
fi

echo "[run_all] Device: $DEVICE"

# Collect audio files (.wav, .flac)
FILES=()
for f in "$AUDIO_DIR"/*.wav "$AUDIO_DIR"/*.flac; do
  [ -f "$f" ] && FILES+=("$f") || true
done

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No audio files found under $AUDIO_DIR" >&2
  exit 0
fi

for AUDIO in "${FILES[@]}"; do
  BASENAME=$(basename -- "$AUDIO")
  BASE="${BASENAME%.*}"
  RTTM="$RTTM_DIR/${BASE}.rttm"

  if [ ! -f "$RTTM" ]; then
    echo "[skip] $BASENAME: missing RTTM -> $RTTM"
    continue
  fi

  echo "[run_all] $BASENAME (dl)"
  "$PY" "$ROOT_DIR/scripts/run_diar_experiment.py" \
    --audio-file "$AUDIO" \
    --experiment "${BASE}_dl" \
    --device "$DEVICE" \
    --denoise auto

  echo "[run_all] $BASENAME (nodl)"
  "$PY" "$ROOT_DIR/scripts/run_diar_experiment.py" \
    --audio-file "$AUDIO" \
    --experiment "${BASE}_nodl" \
    --device "$DEVICE" \
    --denoise none
done

DER_CSV="$ROOT_DIR/reports/der_metrics.csv"
if [ -f "$DER_CSV" ]; then
  echo "[run_all] Completed. Tail of DER CSV:"
  tail -n 20 "$DER_CSV"
else
  echo "[run_all] Completed, but DER CSV not found at $DER_CSV" >&2
fi

