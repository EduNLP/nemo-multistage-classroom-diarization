param(
    [Parameter(Mandatory=$true)][string]$AudioFile,
    [string]$Experiment = $(Get-Date -Format 'yyyyMMdd_HHmmss'),
    [switch]$KeepNewAudio
)

$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $projectRoot

function Run-Step {
    param([string]$Title,[scriptblock]$Action)
    Write-Host "[RUN] $Title" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Title' failed with exit code $LASTEXITCODE"
    }
}

function Run-PythonScript {
    param([string]$Title,[string]$ScriptContent)
    $pyFile = [System.IO.Path]::GetTempFileName() + '.py'
    Set-Content -LiteralPath $pyFile -Value $ScriptContent
    try {
        Run-Step $Title { python $pyFile }
    } finally {
        Remove-Item $pyFile -Force -ErrorAction SilentlyContinue
    }
}

$audioDir = Join-Path $projectRoot 'classbank_audio_data/audio'
$currentAudio = Join-Path $audioDir '2.wav'
$backupAudio = Join-Path $audioDir '2_backup_autorun.wav'
$newAudio = Resolve-Path $AudioFile

if (Test-Path $currentAudio) {
    Copy-Item $currentAudio $backupAudio -Force
}

$prepScript = """
from pathlib import Path
import librosa
import soundfile as sf

root = Path(r'$projectRoot')
source = Path(r'$newAudio')
out_path = root / 'classbank_audio_data/audio/2.wav'
audio, sr = librosa.load(source.as_posix(), sr=16000, mono=True)
sf.write(out_path.as_posix(), audio, 16000)
print(f"Wrote {out_path} (len={len(audio)})")
"""
Run-PythonScript 'Prepare 16k mono audio' $prepScript

Run-Step 'Clean previous outputs' { Remove-Item vad_output_frames, whisper_output_frames, diarization_output -Recurse -Force -ErrorAction SilentlyContinue; Remove-Item vad_outs.json, vad_outs_abs.json -Force -ErrorAction SilentlyContinue }

Run-Step 'Generate VAD frames' { python generate_w2v2_speech_labels/run_vad.py --manifest_file manifests/test_2s.json --checkpoint_path checkpoints/w2v2.ckpt --vad_manifest_path null.json --frames_output_path vad_output_frames }

Run-Step 'Generate Whisper labels' { python generate_whisper_speech_labels/whisper_transcribe.py --manifest_file manifests/test_2s.json --output_dir whisper_output_frames --model base --device cuda }

Run-Step 'Combine VAD/ASR (relative manifest)' { python run_diarization/tune_vad_params.py --manifest_file manifests/test_2s.json --frame_dir vad_output_frames --asr_dir whisper_output_frames --alpha 0.6 --onset 0.55 --offset 0.10 --out_dir vad_outs.json }

Run-Step 'Combine VAD/ASR (absolute manifest)' { python run_diarization/tune_vad_params.py --manifest_file manifests/test_2s_abs.json --frame_dir vad_output_frames --asr_dir whisper_output_frames --alpha 0.6 --onset 0.55 --offset 0.10 --out_dir vad_outs_abs.json }

$logDir = Join-Path $projectRoot 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logPath = Join-Path $logDir "neMo_run_${Experiment}.log"

Run-Step 'Run NeMo diarizer' { python NeMo/offline_diar_infer.py diarizer.manifest_filepath=manifests/test_2s_abs.json diarizer.out_dir=diarization_output diarizer.vad.model_path=null diarizer.vad.external_vad_manifest=vad_outs_abs.json diarizer.speaker_embeddings.parameters.save_embeddings=False diarizer.speaker_embeddings.model_path=titanet_large diarizer.clustering.parameters.oracle_num_speakers=True num_workers=0 hydra.job.chdir=false *> $logPath }

$metricsScript = """
import re
import pathlib
root = pathlib.Path(r'$projectRoot')
log_path = pathlib.Path(r'$logPath')
text = log_path.read_text()
match = re.findall(r"\\| FA: ([^\\|]+) \\| MISS: ([^\\|]+) \\| CER: ([^\\|]+) \\| DER: ([^\\|]+) \\| Spk. Count Acc. ([^\\n]+)", text)
if not match:
    raise SystemExit('No DER metrics found in log')
fa, miss, cer, der, sca = match[-1]
metrics = root / 'reports/der_metrics.csv'
metrics.parent.mkdir(parents=True, exist_ok=True)
if not metrics.exists():
    metrics.write_text('source,alpha,onset,offset,FA,MISS,CER,DER,SpkCountAcc\n')
with metrics.open('a') as fp:
    fp.write(f"nemo_gpu_run_$Experiment,0.6,0.55,0.10,{fa},{miss},{cer},{der},{sca}\n")
print('Appended DER metrics')
"""
Run-PythonScript 'Append DER metrics' $metricsScript

$labelScript = """
from pathlib import Path
import soundfile as sf

root = Path(r'$projectRoot')
audio_path = root / 'classbank_audio_data/audio/2.wav'
data, sr = sf.read(audio_path)
if data.ndim > 1:
    data = data.mean(axis=1)
duration = len(data) / sr

def parse_rttm(path: Path):
    if not path.exists():
        return False, 0, 0, 0.0
    segments = 0
    speakers = set()
    total = 0.0
    with path.open('r', encoding='utf-8') as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            segments += 1
            total += float(parts[4])
            speakers.add(parts[7])
    return True, segments, len(speakers), total

def rel(path: Path):
    return str(path.relative_to(root)).replace('\\\\', '/')

experiment = '$Experiment'
audio_rel = rel(audio_path)

gt_path = root / 'classbank_audio_data/rttm/2.rttm'
gt_exists, gt_segments, gt_unique, gt_total = parse_rttm(gt_path)

gt_rel = rel(gt_path) if gt_exists else ''

pred_path = root / 'diarization_output/pred_rttms/2.rttm'
pred_exists, pred_segments, pred_unique, pred_total = parse_rttm(pred_path)

pred_rel = rel(pred_path) if pred_exists else ''

row = [
    experiment,
    audio_rel,
    'True',
    f"{duration:.5f}",
    gt_rel,
    'True' if gt_exists else 'False',
    str(gt_segments if gt_exists else 0),
    str(gt_unique if gt_exists else 0),
    f"{gt_total:.2f}" if gt_exists else '0.0',
    pred_rel,
    'True' if pred_exists else 'False',
    str(pred_segments if pred_exists else 0),
    str(pred_unique if pred_exists else 0),
    f"{pred_total:.2f}" if pred_exists else '0.0'
]

labels = root / 'reports/label_summary.csv'
labels.parent.mkdir(parents=True, exist_ok=True)
if not labels.exists():
    labels.write_text('file_name,audio_path,audio_exists,duration_sec,gt_rttm_path,gt_exists,gt_segments,gt_unique_speakers,gt_total_speech_sec,pred_rttm_path,pred_exists,pred_segments,pred_unique_speakers,pred_total_speech_sec\n')
with labels.open('a') as fp:
    fp.write(','.join(row) + '\n')
print('Appended label summary row')
"""
Run-PythonScript 'Append label summary row' $labelScript

if (-not $KeepNewAudio -and (Test-Path $backupAudio)) {
    Copy-Item $backupAudio $currentAudio -Force
    Remove-Item $backupAudio -Force
}

Write-Host "Experiment completed: $Experiment" -ForegroundColor Yellow
