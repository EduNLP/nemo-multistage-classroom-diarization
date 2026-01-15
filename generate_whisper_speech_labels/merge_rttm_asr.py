import os
import json
import argparse
from typing import List, Dict


def read_rttm(rttm_path: str) -> List[Dict]:
    """Read an RTTM file and return a list of segments: {start, end, speaker}"""
    segs = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            # RTTM format: SPEAKER <file> <channel> <start> <duration> <ortho> <stype> <name> <confidence>
            try:
                start = float(parts[3])
                duration = float(parts[4])
                end = start + duration
            except Exception:
                continue
            speaker = parts[7] if len(parts) > 7 else "speaker"
            segs.append({"start": start, "end": end, "speaker": speaker})
    return segs


def read_asr_json(asr_json_path: str) -> List[Dict]:
    """Read ASR JSON saved by whisper_transcribe.py; returns list of words with start/end/text."""
    with open(asr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("words", [])


def assign_words_to_speakers(rttm_segs: List[Dict], words: List[Dict]) -> List[Dict]:
    """
    Return list of speaker utterances: {speaker, start, end, text}
    Strategy: iterate words in time order and assign each word to the RTTM segment that contains its midpoint.
    If no segment contains it, skip the word.
    Consecutive words for same speaker are concatenated into utterances.
    """
    rttm_segs = sorted(rttm_segs, key=lambda x: x["start"])
    words = sorted(words, key=lambda x: x["start"])

    utterances = []

    def find_speaker_at_time(t: float):
        # simple linear scan (RTTM files are typically small)
        for seg in rttm_segs:
            if seg["start"] - 1e-6 <= t <= seg["end"] + 1e-6:
                return seg["speaker"]
        return None

    current = None
    for w in words:
        try:
            mid = (float(w["start"]) + float(w["end"])) / 2.0
        except Exception:
            continue
        spk = find_speaker_at_time(mid)
        if spk is None:
            # skip words not inside any RTTM speaker region
            continue
        text = (w.get("text") or "").strip()
        if not text:
            continue
        if current is None:
            current = {"speaker": spk, "start": float(w["start"]), "end": float(w["end"]), "text": text}
        elif current["speaker"] == spk and float(w["start"]) <= current["end"] + 1.0:
            # extend current utterance (1s gap tolerance)
            current["end"] = float(w["end"])
            current["text"] = current["text"] + " " + text
        else:
            utterances.append(current)
            current = {"speaker": spk, "start": float(w["start"]), "end": float(w["end"]), "text": text}

    if current is not None:
        utterances.append(current)

    return utterances


def main():
    parser = argparse.ArgumentParser(description="Merge RTTM speaker segments with ASR words to produce per-speaker transcripts.")
    parser.add_argument("--rttm", type=str, required=True, help="Path to the RTTM file (predicted).")
    parser.add_argument("--asr_json", type=str, required=True, help="Path to the ASR JSON produced by whisper_transcribe.py")
    parser.add_argument("--out", type=str, required=True, help="Output file (JSONL) where each line is {speaker,start,end,text}")
    args = parser.parse_args()

    if not os.path.isfile(args.rttm):
        print("RTTM not found:", args.rttm)
        return
    if not os.path.isfile(args.asr_json):
        print("ASR JSON not found:", args.asr_json)
        return

    rttm_segs = read_rttm(args.rttm)
    words = read_asr_json(args.asr_json)

    utterances = assign_words_to_speakers(rttm_segs, words)

    # write JSONL
    with open(args.out, "w", encoding="utf-8") as fout:
        for u in utterances:
            out = {
                "speaker": u["speaker"],
                "start": round(float(u["start"]), 3),
                "end": round(float(u["end"]), 3),
                "text": u["text"].strip(),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(utterances)} speaker utterances to {args.out}")


if __name__ == "__main__":
    main()