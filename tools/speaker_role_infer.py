from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import re

from .speaker_text_align import summarize_speakers


# Basic keyword dictionaries for several common scenarios.
# These are used when no ML model is available.
ROLE_KEYWORDS: dict[str, dict[str, List[str]]] = {
    "Generic": {
        "Host": ["welcome", "introduce", "agenda", "start", "question"],
        "Guest": ["thanks", "experience", "answer"],
    },
    "Classroom": {
        "Teacher": ["homework", "assignment", "quiz", "lecture", "chapter", "학생", "숙제", "과제", "수업", "강의"],
        "Student": ["question", "answer", "understand", "시험", "질문", "과제", "선생님"],
    },
    "Call Center": {
        "Agent": ["support", "ticket", "account", "policy", "verify", "help", "문의", "계정", "정책", "도와"],
        "Customer": ["problem", "issue", "refund", "cancel", "불만", "환불", "취소", "문제"],
    },
    "Medical": {
        "Doctor": ["symptom", "diagnosis", "prescribe", "treatment", "약", "증상", "진단", "처방", "치료"],
        "Patient": ["pain", "feel", "hurt", "아파", "통증", "호소", "느껴"],
    },
    "Interview": {
        "Interviewer": ["tell me", "walk me", "resume", "experience", "project", "소개", "경험", "질문"],
        "Interviewee": ["i worked", "i built", "i led", "했습니다", "했습니다.", "경험", "프로젝트"],
    },
    "Meeting": {
        "Facilitator": ["agenda", "action items", "follow up", "minutes", "정리", "안건", "회의록"],
        "Participant": ["agree", "disagree", "update", "의견", "참석", "업데이트"],
    },
    "Sales": {
        "Seller": ["price", "discount", "offer", "quote", "재고", "가격", "할인", "제안", "견적"],
        "Buyer": ["budget", "purchase", "buy", "주문", "구매", "문의"],
    },
    "Legal": {
        "Lawyer": ["case", "evidence", "testimony", "contract", "소송", "증거", "증언", "계약"],
        "Client": ["situation", "concern", "i need", "상황", "걱정", "필요"],
    },
}


def _count_keywords(text: str, keywords: List[str]) -> int:
    t = text.lower()
    count = 0
    for kw in keywords:
        # simple word/phrase presence; count occurrences
        c = len(re.findall(re.escape(kw.lower()), t))
        count += c
    return count


def aggregate_speaker_text(utterances: List[dict], max_chars: int = 4000) -> Dict[str, str]:
    agg: Dict[str, str] = {}
    for u in utterances:
        spk = u.get("speaker", "UNK")
        txt = (u.get("text") or "").strip()
        if not txt:
            continue
        prev = agg.get(spk, "")
        new = (prev + (" " if prev else "") + txt).strip()
        # limit size per speaker to keep it light
        if len(new) > max_chars:
            new = new[:max_chars]
        agg[spk] = new
    return agg


def infer_roles_keywords(utterances: List[dict], scenario: str = "Generic") -> Dict[str, str]:
    agg = aggregate_speaker_text(utterances)
    stats = summarize_speakers(utterances)
    role_dict = ROLE_KEYWORDS.get(scenario, ROLE_KEYWORDS["Generic"])  # type: ignore[index]
    roles = list(role_dict.keys())

    # Score speakers for each role
    scores: Dict[str, Dict[str, float]] = {spk: {} for spk in agg.keys()}
    for role, kws in role_dict.items():
        for spk, text in agg.items():
            scores[spk][role] = float(_count_keywords(text, kws))

    # Assign roles: greedy by highest score; tie-breaker by duration
    remaining_roles = roles.copy()
    mapping: Dict[str, str] = {}
    # sort speakers by total duration desc
    order = sorted(stats.items(), key=lambda kv: -kv[1]["duration"])
    for spk, _ in order:
        if not remaining_roles:
            break
        # best role for this speaker
        best = max(remaining_roles, key=lambda r: scores.get(spk, {}).get(r, 0.0))
        mapping[spk] = best
        remaining_roles.remove(best)

    # Any leftover speakers -> assign remaining generic label by index
    for spk in agg.keys():
        if spk not in mapping:
            base = roles[-1] if roles else "Speaker"
            # disambiguate
            idx = sum(1 for v in mapping.values() if v.startswith(base)) + 1
            mapping[spk] = f"{base} {idx}"
    return mapping


def infer_roles_zero_shot(utterances: List[dict], scenario: str = "Generic", model_name_or_path: str | None = None, device: int | None = None) -> Tuple[Dict[str, str], str]:
    """Try to use a local zero-shot model (if available). Falls back to keywords.

    Returns (mapping, mode_used).
    """
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return infer_roles_keywords(utterances, scenario), "keywords"

    labels = list(ROLE_KEYWORDS.get(scenario, ROLE_KEYWORDS["Generic"]).keys())  # type: ignore[index]
    agg = aggregate_speaker_text(utterances)
    stats = summarize_speakers(utterances)

    try:
        if model_name_or_path:
            clf = pipeline("zero-shot-classification", model=model_name_or_path, device=device if device is not None else None)
        else:
            clf = pipeline("zero-shot-classification")
    except Exception:
        return infer_roles_keywords(utterances, scenario), "keywords"

    scores: Dict[str, Dict[str, float]] = {}
    for spk, text in agg.items():
        try:
            out = clf(text, candidate_labels=labels, multi_label=False)
            # normalize to dict[label] -> score
            spk_scores = {lab: 0.0 for lab in labels}
            for lab, sc in zip(out.get("labels", []), out.get("scores", [])):
                spk_scores[str(lab)] = float(sc)
            scores[spk] = spk_scores
        except Exception:
            # fall back per speaker
            scores[spk] = {lab: 0.0 for lab in labels}

    # Greedy assignment by best score per speaker; tie-break by duration
    remaining = labels.copy()
    mapping: Dict[str, str] = {}
    order = sorted(stats.items(), key=lambda kv: -kv[1]["duration"])
    for spk, _ in order:
        if not remaining:
            break
        best = max(remaining, key=lambda r: scores.get(spk, {}).get(r, 0.0))
        mapping[spk] = best
        remaining.remove(best)

    # Leftover speakers -> keyword fallback generic labels
    for spk in agg.keys():
        if spk not in mapping:
            mapping.update(infer_roles_keywords(utterances, scenario))
            break

    return mapping, "zero-shot"
