# Docker Execution Guide

This image packages the diarization pipeline end to end (VAD, Whisper, NeMo, and CSV reporting). Build once, then supply whatever audio and tuning flags you need at runtime.

## 1. Build the image

```bash
# from the project root
docker build -t diarization-pipeline .
```

> - Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
> - Install the NVIDIA Container Toolkit if you plan to run with `--gpus all`

## 2. Run the pipeline

Mount the repository so outputs land back on the host. The container entrypoint is already `scripts/run_diar_experiment.py`, so you only pass that script's flags.

```bash
# GPU run (recommended)
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  diarization-pipeline \
  --audio-file classbank_audio_data/audio/demucs_output_16k_mono.wav \
  --experiment demucs_in_container \
  --alpha 0.45 --onset 0.45 --offset 0.05 \
  --device cuda

# CPU-only (slower)
docker run --rm \
  -v $(pwd):/workspace \
  diarization-pipeline \
  --audio-file classbank_audio_data/audio/demucs_output_16k_mono.wav \
  --experiment demucs_cpu \
  --device cpu
```

Notes:
- On PowerShell or cmd.exe, replace `$(pwd)` with `${PWD}` or `%cd%`.
- Outputs such as `logs/`, `reports/`, and `diarization_output/` are written into the mounted project directory.
- Because the workdir inside the image is `/workspace`, binding the project directory keeps the container and host in sync without rebuilding.
- All existing CLI flags from `scripts/run_diar_experiment.py` remain available (`--alpha`, `--onset`, `--offset`, `--keep-new-audio`, `--device`, etc.).
- Debug shell example:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  --entrypoint bash \
  diarization-pipeline
```

## 3. System packages baked into the image

The Dockerfile installs Boost, SoX, libsndfile, FFmpeg, and general build tooling so `pip install -r requirements.txt` (including editable KenLM) succeeds.

## 4. Troubleshooting

- **Missing CUDA:** Verify the host driver + toolkit and pass `--gpus all`. Otherwise, fall back to `--device cpu`.
- **Root-owned outputs:** The container writes as UID 0. Adjust with `sudo chown -R $USER:$USER reports logs diarization_output` if needed.
- **Large image size:** The requirements list is heavy (NeMo, Whisper, Torch, KenLM), so expect a multi-GB image.
