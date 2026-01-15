param(
  [ValidateSet('cpu','cu121','cu124')]
  [string]$Cuda = 'cpu',
  [string]$Py311Path = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
  function Resolve-Py311Path {
    param([string]$Hint)
    if ($Hint -and (Test-Path $Hint)) { return (Resolve-Path $Hint).Path }
    # Try python launcher to obtain actual 3.11 executable path
    if (Get-Command py -ErrorAction SilentlyContinue) {
      try {
        $exe = & py -3.11 -c "import sys;print(sys.executable)" 2>$null
        $exe = ($exe | Out-String).Trim()
        if ($exe -and (Test-Path $exe)) { return $exe }
      } catch {}
    }
    # Common installation locations
    $candidates = @(
      (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
      (Join-Path $env:ProgramFiles 'Python311\python.exe'),
      (Join-Path $env:ProgramFiles 'Python\Python311\python.exe'),
      (Join-Path ${env:ProgramFiles(x86)} 'Python311\python.exe')
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    # Fallback: search under user local programs
    $base = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    if (Test-Path $base) {
      $found = Get-ChildItem -Path $base -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*Python311*' } | Select-Object -First 1
      if ($found) { return $found.FullName }
    }
    throw 'Python 3.11 executable not found. Provide -Py311Path or ensure "py -3.11" works.'
  }

  $PyExe = Resolve-Py311Path -Hint $Py311Path

  Write-Output '[setup_venv_311] Creating .venv with Python 3.11...'
  & $PyExe -m venv .venv
  if (-not (Test-Path '.venv\Scripts\python.exe')) { Write-Error 'Failed to create venv with Python 3.11.'; exit 2 }

  $Vpy = Join-Path $Root '.venv\Scripts\python.exe'

  Write-Output '[setup_venv_311] Upgrading pip tooling...'
  & $Vpy -m pip install --upgrade pip setuptools wheel packaging

  # Install PyTorch/torchaudio first
  if ($Cuda -eq 'cpu') {
    Write-Output '[setup_venv_311] Installing CPU PyTorch/torchaudio...'
    & $Vpy -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  } elseif ($Cuda -eq 'cu124') {
    Write-Output '[setup_venv_311] Installing CUDA 12.4 PyTorch/torchaudio...'
    & $Vpy -m pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
  } elseif ($Cuda -eq 'cu121') {
    Write-Output '[setup_venv_311] Installing CUDA 12.1 PyTorch/torchaudio...'
    & $Vpy -m pip install torch==2.6.0+cu121 torchaudio==2.6.0+cu121 --index-url https://download.pytorch.org/whl/cu121
  }

  # Core minimal Windows deps (audio/whisper/etc.)
  if (Test-Path 'requirements-windows-minimal.txt') {
    Write-Output '[setup_venv_311] Installing requirements-windows-minimal.txt...'
    & $Vpy -m pip install -r requirements-windows-minimal.txt
  }

  # GUI deps
  if (Test-Path 'requirements-gui.txt') {
    Write-Output '[setup_venv_311] Installing requirements-gui.txt...'
    & $Vpy -m pip install -r requirements-gui.txt
  }

  Write-Output ''
  Write-Output '[setup_venv_311] Done.'
  Write-Output 'Next steps:'
  Write-Output '  1) Activate: .\.venv\Scripts\Activate.ps1'
  Write-Output '  2) Run GUI:  .\run_gui.ps1'
}
finally {
  Pop-Location
}
