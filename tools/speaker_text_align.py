from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import json


@dataclass
class SpkSeg:
    start: float
    end: float
    speaker: str


def parse_rttm(path: Path) -> List[SpkSeg]:
    segs: List[SpkSeg] = []
    if not path.exists():
        return segs
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            # RTTM layout: TYPE FILE CHAN START DURATION ... SPEAKER ...
            try:
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
            except Exception:
                continue
            segs.append(SpkSeg(start=start, end=start + dur, speaker=spk))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def load_asr_words(asr_json: Path) -> List[Dict]:
    if not asr_json.exists():
        return []
    obj = json.loads(asr_json.read_text(encoding="utf-8", errors="ignore"))
    words: List[Dict] = obj.get("words") or []
    norm: List[Dict] = []
    for w in words:
        try:
            s = float(w.get("start", 0.0))
            e = float(w.get("end", s))
            t = (w.get("text") or "").strip()
        except Exception:
            continue
        if not t:
            continue
        if e <= s:
            continue
        norm.append({"start": s, "end": e, "text": t})
    return norm


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def assign_speakers(words: List[Dict], segs: List[SpkSeg]) -> List[Dict]:
    # Two-pointer sweep to find best-overlap speaker per word
    out: List[Dict] = []
    i = 0
    n = len(segs)
    for w in words:
        ws, we = w["start"], w["end"]
        best_spk = None
        best_ov = 0.0
        # advance i to first segment that might overlap
        while i < n and segs[i].end <= ws:
            i += 1
        j = max(0, i - 2)
        while j < n and segs[j].start < we + 1e-6:
            ov = _overlap((ws, we), (segs[j].start, segs[j].end))
            if ov > best_ov:
                best_ov = ov
                best_spk = segs[j].speaker
            if segs[j].start > we:
                break
            j += 1
        out.append({**w, "speaker": best_spk or "UNK", "overlap": best_ov})
    return out


def group_utterances(words_with_spk: List[Dict], max_gap: float = 0.8) -> List[Dict]:
    # Merge consecutive words by same speaker, small temporal gaps
    utts: List[Dict] = []
    cur = None
    for w in words_with_spk:
        spk = w.get("speaker", "UNK")
        if cur is None:
            cur = {"speaker": spk, "start": w["start"], "end": w["end"], "text": w["text"]}
            continue
        if spk == cur["speaker"] and w["start"] - cur["end"] <= max_gap:
            cur["end"] = max(cur["end"], w["end"])
            cur["text"] = (cur["text"] + " " + w["text"]).strip()
        else:
            utts.append(cur)
            cur = {"speaker": spk, "start": w["start"], "end": w["end"], "text": w["text"]}
    if cur is not None:
        utts.append(cur)
    return utts


def align(asr_json: Path, rttm: Path, max_gap: float = 0.8) -> Dict:
    segs = parse_rttm(rttm)
    words = load_asr_words(asr_json)
    words_spk = assign_speakers(words, segs)
    utts = group_utterances(words_spk, max_gap=max_gap)
    speakers = sorted({u["speaker"] for u in utts})
    return {"speakers": speakers, "utterances": utts, "words": words_spk}


def summarize_speakers(utterances: List[Dict]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    for u in utterances:
        spk = u.get("speaker", "UNK")
        dur = float(u.get("end", 0.0)) - float(u.get("start", 0.0))
        s = stats.setdefault(spk, {"duration": 0.0, "utts": 0, "first_start": None})
        s["duration"] += max(0.0, dur)
        s["utts"] += 1
        st_val = float(u.get("start", 0.0))
        s["first_start"] = st_val if s["first_start"] is None else min(s["first_start"], st_val)
    return stats


def guess_roles(utterances: List[Dict]) -> Dict[str, str]:
    """Heuristic role guessing: label the longest speaker as 'Teacher', others as 'Student 1..N'."""
    stats = summarize_speakers(utterances)
    if not stats:
        return {}
    order = sorted(stats.items(), key=lambda kv: (-kv[1]["duration"], kv[1]["first_start"] or 1e9))
    mapping: Dict[str, str] = {}
    if order:
        mapping[order[0][0]] = "Teacher"
    for idx, (spk, _) in enumerate(order[1:], start=1):
        mapping[spk] = f"Student {idx}"
    return mapping
