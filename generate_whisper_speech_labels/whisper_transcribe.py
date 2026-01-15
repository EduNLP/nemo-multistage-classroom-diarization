import os
import json
import math
import argparse
from typing import List, Dict

import librosa
import numpy as np
import whisper


def words_from_transcript(transcript: Dict) -> List[Dict]:
    """Normalize Whisper transcript to a flat list of word dicts: {text, start, end}."""
    words: List[Dict] = []
    for seg in transcript.get("segments", []):
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                text = w.get("word") or w.get("text") or w.get("token") or ""
                try:
                    start = float(w.get("start", seg.get("start", 0.0)))
                except Exception:
                    start = float(seg.get("start", 0.0))
                try:
                    end = float(w.get("end", seg.get("end", start)))
                except Exception:
                    end = float(seg.get("end", start))
                words.append({"text": text.strip(), "start": start, "end": end})
        else:
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                continue
            tokens = seg_text.split()
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + 0.01))
            duration = max(1e-3, seg_end - seg_start)
            per_token = duration / max(1, len(tokens))
            for i, tok in enumerate(tokens):
                token_start = seg_start + i * per_token
                token_end = token_start + per_token
                words.append({"text": tok, "start": token_start, "end": token_end})
    return words


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio from a manifest using Whisper and save word-level ASR JSON + sample-level speech labels (.npy)."
    )
    parser.add_argument("--manifest_file", type=str, required=True, help="Path to the manifest file (JSONL, with audio_filepath).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs (.asr.json and .npy).")
    parser.add_argument("--model", type=str, default="large-v2", help="Whisper model name (default: large-v2).")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a local Whisper model (.pt or dir).")
    parser.add_argument("--download_root", type=str, default=None, help="Directory to cache/download Whisper models.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Whisper (cuda or cpu).")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files where both outputs already exist.")
    args = parser.parse_args()

    manifest_file = args.manifest_file
    output_dir = args.output_dir
    MODEL = args.model
    MODEL_PATH = args.model_path
    DL_ROOT = args.download_root
    DEVICE = args.device

    os.makedirs(output_dir, exist_ok=True)

    name_or_path = MODEL_PATH if MODEL_PATH else MODEL
    print(f"Loading Whisper model '{name_or_path}' on device '{DEVICE}' ...")
    try:
        model = whisper.load_model(name_or_path, device=DEVICE, download_root=DL_ROOT)
    except TypeError:
        # older whisper versions may not support download_root
        model = whisper.load_model(name_or_path, device=DEVICE)
    fp16 = DEVICE != "cpu"

    with open(manifest_file, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as e:
                print(f"Skipping malformed manifest line: {e}")
                continue

            audio_path = entry.get("audio_filepath") or entry.get("audio") or entry.get("path")
            if not audio_path:
                print("No audio filepath found in manifest entry, skipping.")
                continue

            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            asr_json_path = os.path.join(output_dir, base_name + ".asr.json")
            npy_path = os.path.join(output_dir, base_name + ".npy")

            if args.skip_existing and os.path.exists(asr_json_path) and os.path.exists(npy_path):
                print(f"Skipping {audio_path}, outputs already exist.")
                continue

            print("Processing:", audio_path)
            sr = 16000
            try:
                audio_np, _ = librosa.load(audio_path, sr=sr)
            except Exception as e:
                print(f"ERROR loading audio {audio_path}: {e}")
                continue

            try:
                transcript = model.transcribe(audio_np, word_timestamps=True, fp16=fp16)
            except TypeError:
                transcript = model.transcribe(audio_np, fp16=fp16)
            except Exception as e:
                print(f"ERROR transcribing {audio_path}: {e}")
                continue

            words = words_from_transcript(transcript)

            try:
                with open(asr_json_path, "w", encoding="utf-8") as jf:
                    json.dump({"segments": transcript.get("segments", []), "words": words}, jf, ensure_ascii=False, indent=2)
                print(f"Saved ASR JSON to {asr_json_path}")
            except Exception as e:
                print(f"ERROR saving ASR JSON for {audio_path}: {e}")

            num_samples = len(audio_np)
            labels = np.zeros(num_samples, dtype=np.int8)
            for w in words:
                try:
                    start_sec = float(w.get("start", 0.0))
                    end_sec = float(w.get("end", start_sec))
                except Exception:
                    continue
                start_sample = int(math.floor(start_sec * sr))
                end_sample = int(math.ceil(end_sec * sr))
                start_sample = max(0, start_sample)
                end_sample = min(num_samples, end_sample)
                if end_sample > start_sample:
                    labels[start_sample:end_sample] = 1

            try:
                np.save(npy_path, labels)
                print(f"Saved speech labels to {npy_path}")
            except Exception as e:
                print(f"ERROR saving .npy for {audio_path}: {e}")


if __name__ == "__main__":
    main()
