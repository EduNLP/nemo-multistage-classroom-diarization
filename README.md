# Multistage Speaker Diarization in Noisy Classrooms

A specialized system for classroom speech diarization (identifying who is speaking when) using NVIDIA's NeMo framework, optimized for educational environments.

## Overview

This project implements a speaker diarization pipeline specifically designed for classroom audio recordings. It combines:

- Voice Activity Detection (VAD) using a fine-tuned wav2vec2.0 model
- Whisper-based transcription to improve speech detection accuracy
- Speaker clustering using TitaNet embeddings

The system is optimized for the teacher-student interaction patterns typically found in classroom settings.

## Quick Start

1. **Clone and setup the repository:**
   ```bash
   git clone https://github.com/EduNLP/nemo-multistage-classroom-diarization.git
   cd nemo-multistage-classroom-diarization
   rm -rf NeMo
   # Clone the NVIDIA NeMo repository
   git clone https://github.com/NVIDIA/NeMo.git
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Download the pretrained checkpoint:**
   ```bash
   mkdir -p checkpoints/w2v2-robust-large-ckpt
   gdown --fuzzy 'https://drive.google.com/file/d/1f9mMqzpGaLA2RB0m7dcesxo4deOB_GDq/view?usp=sharing' -O ckpt.pt
   mv ckpt.pt checkpoints/w2v2-robust-large-ckpt/
   ```

3. **Prepare audio and manifest files** (see details below). If you are looking to replicate results from our paper, you have the download the respective files listed in `manifests/classbank_test_filenames.txt` from [ClassBank](https://class.talkbank.org/).

4. **Adjust paths in `run.sh`** to match your environment

5. **Run the diarization pipeline:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

## Data Preparation

### Manifest File Format

The pipeline requires manifest files in JSON format with one entry per line:

```json
{"audio_filepath": "data/audio/classroom1.wav", "offset": 0, "duration": 719.72, "label": "infer", "text": "-", "num_speakers": 2, "rttm_filepath": "data/rttm/classroom1.rttm", "uem_filepath": null, "ctm_filepath": null}
```

Each entry must include:
- `audio_filepath`: Path to your audio file
- `num_speakers`: Number of speakers (typically 2 for teacher/student scenarios)
- `rttm_filepath`: Path to corresponding RTTM file for evaluation
- `offset`: Start time (usually 0 to process the entire file)
- `duration`: Length of your audio in seconds

Create your manifest file in `manifests/test.json` or adjust the path in `run.sh`.

### RTTM File Format

Reference RTTM files should follow this format:

```
SPEAKER session_id 1 start_time duration <NA> <NA> speaker_label <NA> <NA>
```

Example:
```
SPEAKER Classroom/1 1 0.000 1.281 <NA> <NA> STU <NA> <NA>
SPEAKER Classroom/1 1 1.281 4.868 <NA> <NA> TEA <NA> <NA>
```

Where:
- `start_time` and `duration` are in seconds
- `speaker_label` is typically TEA (teacher) or STU (student) for two-speaker scenarios
- For multi-speaker experiments, use unique speaker labels or actual speaker names

## Pipeline Components

The diarization system consists of these main components:

1. **VAD Frame Generation**: Uses a wav2vec2.0 robust-large model to detect speech frames
2. **Whisper Transcription**: Generates additional speech frames using Whisper-Large-v3 ASR
3. **VAD Parameter Tuning**: Combines VAD and Whisper frames with tunable parameters
4. **Speaker Diarization**: Uses TitaNet with clustering to identify different speakers

### Changing Whisper Model

To use a different Whisper model variant:

1. Open `generate_whisper_speech_labels/whisper_transcribe.py`
2. Locate the line `MODEL = "large-v3"`
3. Change to any available Whisper model such as:
   - `"base"`
   - `"small"`
   - `"medium"`
   - `"large"`
   - `"large-v2"`
   - `"large-v3"`

Smaller models will run faster but may have lower accuracy, while larger models provide better results but require more computational resources.

## Configuration

The main parameters that can be tuned in `run.sh`:
- `alpha`: Weight for combining VAD and Whisper frames (default: 0.8)
- `onset`: Speech onset threshold (default: 0.3)
- `offset`: Speech offset threshold (default: 0.2)

### Parameter Tuning

To experiment with different parameter values, modify the loops in `run.sh`:

```bash
for alpha in 0.8 0.7 0.6
do
  for offset in 0.2 0.1 0.3
  do
    for onset in 0.3 0.4 0.5
    do
        if (( $(echo "$onset > $offset" | bc -l) )); then
            # Pipeline execution code...
        fi
    done
  done
done
```

Simply add the values you want to test to each loop. For example:
- `for alpha in 0.8 0.7 0.6 0.5` to test alpha values of 0.8, 0.7, 0.6, and 0.5
- `for offset in 0.2 0.1 0.15 0.25` to test offset values of 0.2, 0.1, 0.15, and 0.25
- `for onset in 0.3 0.35 0.4 0.45` to test onset values of 0.3, 0.35, 0.4, and 0.45

Note that the condition `onset > offset` ensures that the onset threshold is always greater than the offset threshold, which is typically required for stable VAD behavior.

## Output

Results from the diarization pipeline are stored in:
- `DER_results.txt`: Contains Diarization Error Rate (DER) metrics for different parameter combinations
- `diarization_output/`: Contains the diarization predictions including RTTM files

## Directory Structure

```
.
├── checkpoints/                    # Pre-trained model checkpoints
├── generate_w2v2_speech_labels/    # VAD frame generation scripts
├── generate_whisper_speech_labels/ # Whisper transcription scripts
├── run_diarization/                # Main diarization pipeline
├── NeMo/                           # NeMo framework submodule
├── manifests/                      # Test and training manifests
└── run.sh                          # Main execution script
```

## Requirements

- NVIDIA GPU (recommended)
- Dependencies listed in `requirements.txt`

## Citation

If you use this system in your research, please cite:

```
@inproceedings{khan2025a,
  title = {Multi-Stage Speaker Diarization for Noisy Classrooms},
  year = {2025},
  doi = {10.5281/zenodo.15870278},
  author = {Khan, Ali Sartaz and Ogunremi, Tolulope and Attia, Ahmed Adel and Demszky, Dorottya},
  booktitle = {Proceedings of the 18th International Conference on Educational Data Mining}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [askhan1@stanford.edu].
