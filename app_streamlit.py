#!/usr/bin/env python
import io
import os
import sys
import json
import time
import uuid
import subprocess
from pathlib import Path

import streamlit as st

# Local imports
from tools.speaker_text_align import align, guess_roles
from tools.speaker_role_infer import infer_roles_zero_shot, infer_roles_keywords, ROLE_KEYWORDS


def _list_files(root: Path, exts: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    if not root.exists():
        return out
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p))
    return sorted(out)


def discover_local_models(base: Path) -> dict:
    whisper_dir = base / 'whisper'
    whisper_paths = _list_files(whisper_dir, ('.pt',))
    whisper_paths += [str(p) for p in whisper_dir.glob('*') if p.is_dir()]
    models = {
        'whisper': sorted(set(whisper_paths)),
        'nemo_msdd': _list_files(base / 'nemo' / 'msdd', ('.nemo',)),
        'nemo_embedder': _list_files(base / 'nemo' / 'embedder', ('.nemo',)),
        'hf_zeroshot': [str(p) for p in (base / 'hf' / 'zero-shot').glob('*') if p.is_dir()],
    }
    return models


PROJECT_ROOT = Path(__file__).resolve().parent


def detect_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_pipeline(audio_path: Path, experiment: str, device: str, denoise: str,
                 whisper_model: str, msdd_model: str | None, spk_embedder: str,
                 whisper_model_path: str | None = None, whisper_cache_dir: str | None = None) -> tuple[Path, Path]:
    """Run the existing pipeline and return (pred_rttm, whisper_asr_json)."""
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / 'ui_run.log'
    try:
        if log_path.exists():
            log_path.unlink()
    except Exception:
        pass

    cmd = [
        sys.executable, '-u',
        'scripts/run_diar_experiment.py',
        '--audio-file', str(audio_path),
        '--experiment', experiment,
        '--device', device,
        '--denoise', denoise,
        '--whisper-model', whisper_model,
        '--spk-embedder', spk_embedder,
    ]
    if whisper_model_path:
        cmd += ['--whisper-model-path', whisper_model_path]
    if whisper_cache_dir:
        cmd += ['--whisper-cache-dir', whisper_cache_dir]
    
    with open(log_path, 'w', encoding='utf-8') as lf:
        st.info('Running diarization pipeline... this can take time.')
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed (see log {log_path}).")

    # Paths produced by the pipeline
    tmp_audio = PROJECT_ROOT / 'classbank_audio_data' / 'audio_tmp' / f'{experiment}.wav'
    if not tmp_audio.exists():
        df_tmp = tmp_audio.with_name(f"{tmp_audio.stem}_df{tmp_audio.suffix}")
        if df_tmp.exists():
            tmp_audio = df_tmp
        else:
            alt = sorted(tmp_audio.parent.glob(f"{tmp_audio.stem}*.wav"))
            if alt:
                tmp_audio = alt[-1]

    def _resolve_generated(base: Path) -> Path:
        if base.exists():
            return base
        pattern = f"{tmp_audio.stem}*{base.suffix}"
        matches = sorted(base.parent.glob(pattern))
        return matches[-1] if matches else base

    pred_rttm = _resolve_generated(PROJECT_ROOT / 'diarization_output' / 'pred_rttms' / f'{tmp_audio.stem}.rttm')
    asr_json = _resolve_generated(PROJECT_ROOT / 'whisper_output_frames' / f'{tmp_audio.stem}.asr.json')
    # Show run log for user visibility
    try:
        with st.expander('Run Log', expanded=False):
            txt = log_path.read_text(encoding='utf-8', errors='ignore')
            st.code(txt)
    except Exception:
        pass

    return pred_rttm, asr_json


def main():
    st.set_page_config(page_title='Speaker Diarization Enhancement Using Denoising DL-Model', layout='wide')
    st.title('Speaker Diarization Enhancement Using Denoising DL-Model')

    with st.sidebar:
        st.header('Settings')
        device_default = detect_device()
        device = st.selectbox('Device', options=['cuda', 'cpu'], index=0 if device_default == 'cuda' else 1)
        denoise = st.selectbox('Denoise', options=['auto', 'dfnet3', 'none'], index=0)
        # Local models discovery
        models_dir = PROJECT_ROOT / 'models'
        try:
            models_dir_rel = models_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            models_dir_rel = models_dir
        local = discover_local_models(models_dir)
        st.caption(f'Local models dir: {models_dir_rel}')

        whisper_presets = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
        whisper_options: list[tuple[str, str | None, str | None]] = [
            (f"Preset: {name}", name, None) for name in whisper_presets
        ]
        whisper_options += [
            (f"Local: {Path(path).name}", None, path) for path in local['whisper']
        ]
        whisper_labels = [label for label, _, _ in whisper_options] or ['Preset: base']
        default_idx = next((i for i, (_, name, _) in enumerate(whisper_options) if name == 'base'), 0)
        selected_whisper = st.selectbox('Whisper model', options=whisper_labels, index=default_idx if whisper_labels else 0)
        sel_label_idx = whisper_labels.index(selected_whisper)
        _, preset_name, preset_path = whisper_options[sel_label_idx]
        if preset_name:
            whisper_model = preset_name
            whisper_model_path = None
        else:
            whisper_model = Path(preset_path).stem if preset_path else 'base'
            whisper_model_path = preset_path
        whisper_cache_dir = st.text_input('Whisper cache dir (optional)', value='')

        embedder_presets = ['titanet_large', 'ecapa_tdnn', 'speakerverification_speakernet']
        embedder_options: list[tuple[str, str | None]] = [(f"Preset: {name}", name) for name in embedder_presets]
        embedder_options += [(f"Local: {Path(path).name}", path) for path in local['nemo_embedder']]
        embedder_labels = [label for label, _ in embedder_options] or ['Preset: titanet_large']
        embedder_default = next((i for i, (_, val) in enumerate(embedder_options) if val == 'titanet_large'), 0)
        selected_embedder = st.selectbox('Speaker embedder', options=embedder_labels, index=embedder_default if embedder_labels else 0)
        spk_embedder = embedder_options[embedder_labels.index(selected_embedder)][1] or 'titanet_large'

        msdd_model = None
        max_gap = st.slider('Utterance merge gap (sec)', min_value=0.2, max_value=2.0, value=0.8, step=0.1)
        st.divider()
        st.subheader('Role Inference')
        role_mode = st.selectbox('Mode', options=['Text-based (zero-shot/keywords)', 'Heuristic (duration)'], index=0)
        role_scenario = st.selectbox('Scenario', options=list(ROLE_KEYWORDS.keys()), index=0)
        zero_shot_presets = [
            'facebook/bart-large-mnli',
            'facebook/bart-large-xsum',
            'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
        ]
        zero_shot_options: list[tuple[str, str | None]] = [(f"Preset: {name}", name) for name in zero_shot_presets]
        zero_shot_options += [(f"Local: {Path(path).name}", path) for path in local['hf_zeroshot']]
        zero_shot_options.append(('None', None))
        zero_shot_labels = [label for label, _ in zero_shot_options]
        zero_default = next((i for i, (_, val) in enumerate(zero_shot_options) if val == 'facebook/bart-large-mnli'), len(zero_shot_options) - 1)
        selected_zero_shot = st.selectbox('HF zero-shot model', options=zero_shot_labels, index=zero_default if zero_shot_labels else 0)
        hf_zero_shot_model = zero_shot_options[zero_shot_labels.index(selected_zero_shot)][1]

    uploaded = st.file_uploader('Upload audio/video', type=['wav', 'flac', 'mp3', 'm4a', 'mp4', 'mkv', 'mov', 'avi', 'webm'])
    run_clicked = st.button('Run diarization + ASR')

    if uploaded is not None:
        st.audio(uploaded, format='audio/wav')

    if run_clicked:
        if uploaded is None:
            st.warning('Please upload an audio file first.')
            st.stop()

        # Save upload to a temp location inside the project
        in_dir = PROJECT_ROOT / 'ui_uploads'
        in_dir.mkdir(exist_ok=True)
        in_path = in_dir / uploaded.name
        with open(in_path, 'wb') as f:
            f.write(uploaded.getbuffer())

        # If video, extract audio to 16k mono wav
        def is_video(p: Path) -> bool:
            return p.suffix.lower() in {'.mp4', '.mkv', '.mov', '.avi', '.webm'}

        def _resolve_ffmpeg_bin() -> str | None:
            import shutil
            bin_path = shutil.which('ffmpeg')
            if bin_path:
                return bin_path
            try:
                import imageio_ffmpeg  # type: ignore
                return imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                return None

        def extract_audio_ffmpeg(src: Path, dst: Path) -> bool:
            import subprocess
            ffm = _resolve_ffmpeg_bin()
            if not ffm:
                return False
            cmd = [ffm, '-y', '-i', str(src), '-vn', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', str(dst)]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0
            except Exception:
                return False

        def extract_audio_moviepy(src: Path, dst: Path) -> bool:
            try:
                import moviepy.editor as mpe  # type: ignore
                clip = mpe.VideoFileClip(str(src))
                audio = clip.audio
                if audio is None:
                    return False
                # moviepy write_audiofile lets us set fps and codec
                audio.write_audiofile(str(dst), fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                try:
                    clip.close()
                except Exception:
                    pass
                return dst.exists() and dst.stat().st_size > 0
            except Exception:
                return False

        src_for_pipeline = in_path
        if is_video(in_path):
            st.info('Extracting audio from video...')
            wav_out = in_dir / (in_path.stem + '.wav')
            ok = extract_audio_ffmpeg(in_path, wav_out)
            if not ok:
                ok = extract_audio_moviepy(in_path, wav_out)
            if not ok:
                st.error('?????⑦떍?????ㅻ쿋驪???????곌숯: ffmpeg ?????moviepy 嚥????β뼯援η뙴??醫딆쓧? ????썹땟???嶺뚮ㅎ???? ffmpeg??????紐???????춦??pip install moviepy imageio-ffmpeg ??????⑤베鍮???嶺뚮㉡????嶺뚮슣堉???')
                st.stop()
            src_for_pipeline = wav_out

        experiment = f"ui_{Path(uploaded.name).stem}_{uuid.uuid4().hex[:6]}"

        try:
            # Map denoise selection to values
            pred_rttm, asr_json = run_pipeline(
                audio_path=src_for_pipeline,
                experiment=experiment,
                device=device,
                denoise=denoise,
                whisper_model=whisper_model,
                msdd_model=(msdd_model or None),
                spk_embedder=spk_embedder,
                whisper_model_path=(whisper_model_path or None),
                whisper_cache_dir=(whisper_cache_dir or None),
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

        cols = st.columns(3)
        cols[0].metric('Pred RTTM exists', str(pred_rttm.exists()))
        cols[1].metric('ASR JSON exists', str(asr_json.exists()))
        cols[2].metric('Experiment', experiment)

        if not pred_rttm.exists() or not asr_json.exists():
            st.error('Outputs not found; check logs in ./logs')
            st.stop()

        result = align(asr_json, pred_rttm, max_gap=max_gap)
        st.subheader('Speakers')
        st.write(', '.join(result['speakers']))

        st.subheader('Speaker-Attributed Transcript')
        # Role inference per user preference
        if role_mode.startswith('Text-based'):
            chosen = hf_zero_shot_model or None
            dev_idx = (0 if device == 'cuda' else -1)
            if chosen:
                role_map, used = infer_roles_zero_shot(result['utterances'], scenario=role_scenario, model_name_or_path=chosen, device=dev_idx)
            else:
                role_map, used = infer_roles_zero_shot(result['utterances'], scenario=role_scenario, model_name_or_path=None, device=dev_idx)
            if used != 'zero-shot':
                st.caption('Zero-shot model unavailable; used keyword-based inference.')
        else:
            role_map = guess_roles(result['utterances'])

        with st.expander('Rename speakers (inferred roles applied)', expanded=False):
            edited_map: dict[str, str] = {}
            for spk in result['speakers']:
                role = role_map.get(spk, spk)
                edited_map[spk] = st.text_input(f'Name for {spk}', value=role, key=f'name_{spk}')

        for utt in result['utterances']:
            spk = utt['speaker']
            name = edited_map.get(spk) or role_map.get(spk) or spk
            start = utt['start']
            end = utt['end']
            text = utt['text']
            st.markdown(f"**{name}** [{start:.2f}–{end:.2f}]: {text}")

        with st.expander('Debug: word-level assignments'):
            st.dataframe(result['words'])


if __name__ == '__main__':
    main()

