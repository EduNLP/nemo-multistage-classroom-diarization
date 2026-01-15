Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $PY = 'py' } else { throw 'Python not found in PATH.' }
  } else { $PY = 'python' }

  # Upgrade pip tooling (helps with Py3.12 compatibility). Avoid bumping packaging too high.
  & $PY -m pip install --upgrade pip setuptools wheel | Out-Null

  function Test-Import($module) {
    try { & $PY -c "import $module" 2>$null | Out-Null; return $true } catch { return $false }
  }
  function Test-ModuleSpec($module) {
    try {
      $out = & $PY -c "import importlib.util; print(1 if importlib.util.find_spec('$module') else 0)" 2>$null
      return (($out | Out-String).Trim() -eq '1')
    } catch { return $false }
  }

  # Ensure core deps on Windows without pulling platform-incompatible packages from requirements.txt
  $isWindows = $env:OS -eq 'Windows_NT'
  if ($isWindows) {
    $needLibrosa = -not (Test-Import 'librosa')
    $needSF = -not (Test-Import 'soundfile')
    if ($needLibrosa -or $needSF) {
      Write-Output '[run_gui] Installing minimal core deps for Windows (numpy/librosa/soundfile/soxr/resampy) ...'
      try { & $PY -m pip install --only-binary=:all: numpy==1.26.4 | Out-Null } catch {}
      try { & $PY -m pip install resampy==0.4.3 -q | Out-Null } catch {}
      try { & $PY -m pip install soundfile==0.13.1 librosa==0.10.2.post1 -q | Out-Null } catch {}
      # If a curated Windows requirements file exists, install it to consolidate versions
      if (Test-Path 'requirements-windows-minimal.txt') {
        try { & $PY -m pip install -r requirements-windows-minimal.txt -q | Out-Null } catch { Write-Warning '[run_gui] windows-minimal requirements install encountered issues' }
      }
    }
  } else {
    # Non-Windows: install from requirements.txt if core libs missing
    $needsCore = -not (Test-Import 'librosa') -or -not (Test-Import 'soundfile')
    if ($needsCore -and (Test-Path 'requirements.txt')) {
      Write-Output '[run_gui] Installing core project requirements...'
      try { & $PY -m pip install --only-binary=:all: numpy==1.26.4 | Out-Null } catch {}
      try { & $PY -m pip install pdm-backend -q | Out-Null } catch {}
      try { & $PY -m pip install meson-python ninja -q | Out-Null } catch {}
      try { & $PY -m pip install cython -q | Out-Null } catch {}
      & $PY -m pip install -r requirements.txt --no-build-isolation
    }
  }

  # Ensure GUI deps
  if (-not (Test-Import 'streamlit') -or -not (Test-Import 'transformers')) {
    Write-Output '[run_gui] Installing GUI requirements (streamlit/transformers)...'
    & $PY -m pip install -r requirements-gui.txt
  }
  # Ensure packaging stays <24 for deepfilternet compatibility
  try { & $PY -m pip install "packaging<24,>=23" -q | Out-Null } catch { Write-Warning '[run_gui] packaging pin failed' }

  # Ensure DeepFilterNet runtime (df API) for denoising
  $hasDF = (Test-ModuleSpec 'df') -or (Test-ModuleSpec 'deepfilternet')
  if (-not $hasDF) {
    Write-Output '[run_gui] Installing DeepFilterNet (deepfilternet) for denoising...'
    try { & $PY -m pip install deepfilternet -q | Out-Null } catch { Write-Warning '[run_gui] deepfilternet install failed; denoise may be skipped.' }
  }
  # Pin packaging for deepfilternet 0.5.x compatibility
  try { & $PY -m pip install "packaging<24,>=23" -q | Out-Null } catch {}

  # Whisper and torch checks (best-effort on Windows)
  if (-not (Test-Import 'whisper')) { try { & $PY -m pip install openai-whisper -q | Out-Null } catch {} }
  # Avoid false negatives when CUDA torch is installed but DLLs missing: use spec-detection
  $hasTorchSpec = Test-ModuleSpec 'torch'
  if (-not $hasTorchSpec) {
    Write-Warning '[run_gui] torch not detected. Attempting CPU-only install...'
    try { & $PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu -q | Out-Null } catch { Write-Warning '[run_gui] torch install attempt failed; please install manually.' }
  }
  $hasTorchaudioSpec = Test-ModuleSpec 'torchaudio'
  if (-not $hasTorchaudioSpec) {
    Write-Warning '[run_gui] torchaudio not detected. Attempting CPU-only install...'
    try { & $PY -m pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu -q | Out-Null } catch { Write-Warning '[run_gui] torchaudio install attempt failed; please install manually.' }
  }
  # Ensure tqdm unconditionally (used by VAD script)
  if (-not (Test-Import 'tqdm')) { try { & $PY -m pip install tqdm -q | Out-Null } catch {} }
  # Try to ensure NeMo toolkit (for diarization). This may fail on Windows; warn if so.
  if (-not (Test-Import 'nemo')) {
    Write-Warning '[run_gui] nemo-toolkit not detected. Attempting install (may be unsupported on Windows)...'
    try { & $PY -m pip install nemo-toolkit==2.2.0 -q | Out-Null } catch { Write-Warning '[run_gui] nemo-toolkit install failed. See NeMo requirements; diarization step may fail.' }
  }
  # lhotse is used by some NeMo ASR data modules
  if (-not (Test-Import 'lhotse')) { try { & $PY -m pip install lhotse==1.31.1 -q | Out-Null } catch { Write-Warning '[run_gui] lhotse install failed' } }
  # Lightning / Hydra / OmegaConf for NeMo entry script
  if (-not (Test-Import 'lightning')) { try { & $PY -m pip install lightning==2.4.0 -q | Out-Null } catch { Write-Warning '[run_gui] lightning install failed' } }
  if (-not (Test-Import 'omegaconf')) { try { & $PY -m pip install omegaconf==2.3.0 -q | Out-Null } catch { Write-Warning '[run_gui] omegaconf install failed' } }
  if (-not (Test-Import 'hydra')) { try { & $PY -m pip install hydra-core==1.3.2 -q | Out-Null } catch { Write-Warning '[run_gui] hydra-core install failed' } }
  # Ensure VAD runtime deps
  if (-not (Test-Import 'scipy')) { try { & $PY -m pip install "scipy>=1.11,<1.16" -q | Out-Null } catch { Write-Warning '[run_gui] scipy install failed' } }
  if (-not (Test-Import 'sklearn')) { try { & $PY -m pip install scikit-learn==1.6.1 -q | Out-Null } catch { Write-Warning '[run_gui] scikit-learn install failed' } }
  if (-not (Test-Import 'speechbrain')) { try { & $PY -m pip install speechbrain==1.0.2 -q | Out-Null } catch { Write-Warning '[run_gui] speechbrain install failed' } }
  # Ensure cffi for soundfile backend (_cffi_backend)
  if (-not (Test-Import 'cffi')) { try { & $PY -m pip install cffi -q | Out-Null } catch { Write-Warning '[run_gui] cffi install failed (soundfile may not import).' } }
  # Optional: moviepy for video file support (audio extraction)
  if (-not (Test-Import 'moviepy')) {
    try { & $PY -m pip install moviepy imageio-ffmpeg -q | Out-Null } catch { Write-Warning '[run_gui] moviepy install failed; video inputs will require ffmpeg CLI.' }
  }
  # Pin/ensure a few problematic wheels on Windows (transformers/whisper stack)
  try { & $PY -m pip install regex==2024.11.6 -q | Out-Null } catch { Write-Warning '[run_gui] regex pin failed' }
  try { & $PY -m pip install tiktoken==0.7.0 -q | Out-Null } catch { Write-Warning '[run_gui] tiktoken install failed' }
  try { & $PY -m pip install sentencepiece==0.1.99 -q | Out-Null } catch { Write-Warning '[run_gui] sentencepiece install failed' }
  # Ensure pandas compatible with numpy 1.26.x
  if (-not (Test-Import 'pandas')) { try { & $PY -m pip install pandas==2.2.3 -q | Out-Null } catch { Write-Warning '[run_gui] pandas install failed' } }

  # NeMo extra deps
  if (-not (Test-Import 'einops')) { try { & $PY -m pip install einops==0.8.1 -q | Out-Null } catch { Write-Warning '[run_gui] einops install failed' } }
  if (-not (Test-Import 'lhotse')) { try { & $PY -m pip install lhotse==1.31.1 -q | Out-Null } catch { Write-Warning '[run_gui] lhotse install failed' } }
  if (-not (Test-Import 'webdataset')) { try { & $PY -m pip install webdataset==0.2.111 -q | Out-Null } catch { Write-Warning '[run_gui] webdataset install failed' } }
  if (-not (Test-Import 'datasets')) { try { & $PY -m pip install datasets==3.4.1 -q | Out-Null } catch { Write-Warning '[run_gui] datasets install failed' } }
  if (-not (Test-Import 'pyarrow')) { try { & $PY -m pip install pyarrow==19.0.1 -q | Out-Null } catch { Write-Warning '[run_gui] pyarrow install failed' } }
  if (-not (Test-Import 'jiwer')) { try { & $PY -m pip install jiwer==3.1.0 -q | Out-Null } catch { Write-Warning '[run_gui] jiwer install failed' } }
  # pyannote metrics stack (already in minimal requirements, but enforce quietly)
  try { & $PY -m pip install pyannote.core==5.0.0 pyannote.database==5.1.3 pyannote.metrics==3.2.1 -q | Out-Null } catch { Write-Warning '[run_gui] pyannote stack install had issues' }
  # onnx (NeMo sometimes imports ONNX)
  try { & $PY -m pip install onnx==1.17.0 -q | Out-Null } catch { Write-Warning '[run_gui] onnx install failed (may not be required on Windows)' }

  # Optional: prewarm df to avoid first-run overhead
  try {
    $code = @'
from importlib.util import find_spec
if find_spec("df") is not None:
    try:
        from df.enhance import init_df
        init_df()
        print("[run_gui] DeepFilterNet df prewarm OK")
    except Exception as e:
        print("[run_gui] DeepFilterNet df prewarm failed:", type(e).__name__)
else:
    print("[run_gui] DeepFilterNet df not found; skipping prewarm")
'@
    & $PY -c $code | Out-Null
  } catch {}

  # Workaround: if soxr import fails due to DLL load error, uninstall soxr and force librosa to fallback to resampy
  try {
    $pycode = @'
import sys
try:
    import soxr
    print("OK")
except Exception as e:
    print("FAIL:" + type(e).__name__)
'@
    $soxr_probe = & $PY -c $pycode 2>$null
    $soxr_probe = ($soxr_probe | Out-String).Trim()
    if ($soxr_probe -like 'FAIL*') {
      Write-Warning '[run_gui] soxr import failed; uninstalling soxr and forcing librosa to use resampy.'
      try { & $PY -m pip uninstall -y soxr -q | Out-Null } catch {}
      $env:LIBROSA_RESAMPLER = 'resampy'
    }
  } catch {}

  & $PY -m streamlit run app_streamlit.py
}
finally {
  Pop-Location
}
