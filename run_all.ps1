Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Run diarization for every audio in classbank_audio_data/audio
# twice per file: with DeepFilterNet3 denoising (dl) and without (nodl).

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AudioDir = Join-Path $Root 'classbank_audio_data\audio'
$RttmDir  = Join-Path $Root 'classbank_audio_data\rttm'

if (-not (Test-Path $AudioDir)) {
  Write-Error "Audio directory not found: $AudioDir"
  exit 1
}

# Resolve python
function Resolve-Python {
  if (Get-Command python -ErrorAction SilentlyContinue) { return 'python' }
  elseif (Get-Command python3 -ErrorAction SilentlyContinue) { return 'python3' }
  else { throw 'Python executable not found in PATH.' }
}

$PY = Resolve-Python

# Detect device (cuda/cpu)
try {
  $dev = & $PY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
  $Device = ($dev | Out-String).Trim()
  if (-not $Device) { $Device = 'cpu' }
} catch { $Device = 'cpu' }

Write-Output "[run_all.ps1] Device: $Device"

# Collect audio files
$files = Get-ChildItem -Path $AudioDir -File | Where-Object { $_.Extension -in '.wav','.flac' }
if (-not $files) {
  Write-Warning "No audio files found under $AudioDir"
  exit 0
}

foreach ($f in $files) {
  $base = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
  $rttm = Join-Path $RttmDir ("$base.rttm")
  if (-not (Test-Path $rttm)) {
    Write-Warning "[skip] $($f.Name): missing RTTM -> $rttm"
    continue
  }

  Write-Output "[run_all.ps1] $($f.Name) (dl)"
  & $PY (Join-Path $Root 'scripts\run_diar_experiment.py') `
    --audio-file $f.FullName `
    --experiment ("${base}_dl") `
    --device $Device `
    --denoise auto

  Write-Output "[run_all.ps1] $($f.Name) (nodl)"
  & $PY (Join-Path $Root 'scripts\run_diar_experiment.py') `
    --audio-file $f.FullName `
    --experiment ("${base}_nodl") `
    --device $Device `
    --denoise none
}

$csv = Join-Path $Root 'reports\der_metrics.csv'
if (Test-Path $csv) {
  Write-Output "[run_all.ps1] Completed. Tail of DER CSV:"
  Get-Content $csv | Select-Object -Last 20
} else {
  Write-Warning "[run_all.ps1] Completed, but DER CSV not found at $csv"
}

