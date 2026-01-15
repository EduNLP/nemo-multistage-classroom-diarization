Models directory layout and GUI selection

Overview
- You can pre-download models and select them in the Streamlit GUI.
- Place files under the `models/` folder with the structure below.

Layout
- `models/whisper/`
  - Put Whisper checkpoints here: either `.pt` files or unpacked model directories.
  - Examples: `models/whisper/base.pt`, `models/whisper/large-v2/`
- `models/nemo/msdd/`
  - NeMo MSDD diarization models (`.nemo`).
  - Example: `models/nemo/msdd/msdd_model.nemo`
- `models/nemo/embedder/`
  - NeMo speaker embedding models (`.nemo`).
  - Example: `models/nemo/embedder/titanet_large.nemo`
- `models/hf/zero-shot/`
  - Hugging Face zero-shot classification models as directories (config + weights).
  - Example: `models/hf/zero-shot/facebook-bart-large-mnli/`

GUI usage
- In the sidebar:
  - Whisper: choose a model name or a local path from `models/whisper/`. Optional cache dir can be set.
  - NeMo MSDD: choose from local `.nemo` (optional) or leave empty to disable.
  - Speaker embedder: choose a default name or a local `.nemo` path.
  - Zero-shot roles: select a local HF model path or leave empty to auto/fallback.

Notes
- If no local models are found, defaults will be used (and may attempt to download on first run).
- In offline/restricted environments, ensure files are placed beforehand into the paths above.

