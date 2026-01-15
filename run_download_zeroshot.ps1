Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

param(
  [Parameter(Mandatory=$true)] [string]$ModelId,
  [string]$OutDir
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $PY = 'py' } else { throw 'Python not found in PATH.' }
  } else { $PY = 'python' }

  # Ensure hub client is available
  try { & $PY -c "import huggingface_hub" 2>$null | Out-Null } catch { & $PY -m pip install -q huggingface_hub }

  $argsList = @('scripts/download_hf_model.py', $ModelId)
  if ($OutDir) { $argsList += ,$OutDir }
  & $PY @argsList
}
finally {
  Pop-Location
}

