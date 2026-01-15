import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import soundfile as sf


@dataclass
class SegmentStats:
    segment_count: int
    unique_speakers: int
    total_speech_sec: float


RTTM_PATTERN = re.compile(r"^SPEAKER\s+\S+\s+\S+\s+(?P<start>\d+(?:\.\d+)?)\s+(?P<duration>\d+(?:\.\d+)?)\s+\S+\s+\S+\s+(?P<speaker>\S+)")
DER_PATTERN = re.compile(
    r"alpha=(?P<alpha>[-+]?\d*\.\d+|\d+)\s*,\s*"
    r"onset=(?P<onset>[-+]?\d*\.\d+|\d+)\s*,\s*"
    r"offset=(?P<offset>[-+]?\d*\.\d+|\d+)\s*,\s*\|\s*"
    r"FA:\s*(?P<FA>[-+]?\d*\.\d+|\d+)\s*\|\s*"
    r"MISS:\s*(?P<MISS>[-+]?\d*\.\d+|\d+)\s*\|\s*"
    r"CER:\s*(?P<CER>[-+]?\d*\.\d+|\d+)\s*\|\s*"
    r"DER:\s*(?P<DER>[-+]?\d*\.\d+|\d+)\s*\|\s*"
    r"Spk\.\s*Count\s*Acc\.\s*(?P<SpkCountAcc>[-+]?\d*\.\d+|\d+)"
)


def read_file_list(file_list_path: Path) -> List[str]:
    entries: List[str] = []
    with file_list_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            name = line.strip()
            if name:
                entries.append(name)
    return entries


def ensure_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_audio_duration(path: Path) -> Optional[float]:
    try:
        with sf.SoundFile(path) as audio:
            return len(audio) / float(audio.samplerate)
    except Exception as exc:  # pragma: no cover - informational
        print(f"[WARN] Could not determine audio duration: {path} ({exc})")
        return None


def parse_rttm(path: Path) -> SegmentStats:
    segment_count = 0
    speakers = set()
    speech_total = 0.0

    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                match = RTTM_PATTERN.match(line.strip())
                if not match:
                    continue
                segment_count += 1
                speakers.add(match.group("speaker"))
                duration = float(match.group("duration"))
                speech_total += duration
    except FileNotFoundError:
        return SegmentStats(0, 0, 0.0)

    return SegmentStats(segment_count, len(speakers), speech_total)


def build_manifest_entry(
    audio_relative: str,
    audio_exists: bool,
    duration: Optional[float],
    gt_stats: SegmentStats,
    gt_rttm_relative: Optional[str]
) -> Optional[dict]:
    if not audio_exists:
        return None

    entry = {
        "audio_filepath": audio_relative.replace("\\", "/"),
        "offset": 0,
        "duration": round(duration, 6) if duration is not None else None,
        "label": "infer",
        "text": "-",
        "num_speakers": gt_stats.unique_speakers if gt_stats.unique_speakers > 0 else None,
        "rttm_filepath": gt_rttm_relative.replace("\\", "/") if gt_rttm_relative else None,
        "uem_filepath": None,
        "ctm_filepath": None,
    }
    return entry


def write_manifest(entries: Iterable[dict], manifest_path: Path) -> None:
    data = [entry for entry in entries if entry is not None]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fp:
        for entry in data:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_label_csv(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_der_file(path: Path, base: Path) -> List[dict]:
    parsed_rows: List[dict] = []
    if not path.exists():
        return parsed_rows

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            match = DER_PATTERN.search(line)
            if not match:
                continue
            row = {"source": ensure_relative(path, base).replace("\\", "/")}
            row.update({key: float(value) for key, value in match.groupdict().items()})
            parsed_rows.append(row)
    return parsed_rows


def write_der_csv(csv_path: Path, sources: List[Path], base: Path) -> None:
    rows: List[dict] = []
    for source in sources:
        rows.extend(parse_der_file(source, base))

    if not rows:
        return

    fieldnames = ["source", "alpha", "onset", "offset", "FA", "MISS", "CER", "DER", "SpkCountAcc"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLASSBANK data curation and reporting")
    parser.add_argument("--file-list", default="manifests/classbank_test_filenames.txt", help="List of filenames to process")
    parser.add_argument("--audio-dir", default="classbank_audio_data/audio", help="Audio directory")
    parser.add_argument("--rttm-dir", default="classbank_audio_data/rttm", help="Ground-truth RTTM directory")
    parser.add_argument("--pred-rttm-dir", default="diarization_output/pred_rttms", help="Predicted RTTM directory")
    parser.add_argument("--manifest-out", default="manifests/test_all_speaker.json", help="Manifest output path")
    parser.add_argument("--label-csv", default="reports/label_summary.csv", help="Label summary CSV path")
    parser.add_argument("--der-csv", default="reports/der_metrics.csv", help="DER metrics CSV path")
    parser.add_argument(
        "--der-sources",
        nargs="*",
        default=[
            "run_diarization/test_results.txt",
            "DER_results_all_speaker.txt",
            "DER_results_2s.txt",
        ],
        help="List of DER text sources"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    file_list_path = project_root / args.file_list
    audio_dir = project_root / args.audio_dir
    rttm_dir = project_root / args.rttm_dir
    pred_rttm_dir = project_root / args.pred_rttm_dir
    manifest_path = project_root / args.manifest_out
    label_csv_path = project_root / args.label_csv
    der_csv_path = project_root / args.der_csv
    der_sources = [project_root / Path(src) for src in args.der_sources]

    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_path}")

    file_names = read_file_list(file_list_path)
    label_rows: List[dict] = []
    manifest_entries: List[dict] = []

    for filename in file_names:
        audio_path = audio_dir / filename
        stem = Path(filename).stem
        gt_rttm_path = rttm_dir / f"{stem}.rttm"
        pred_rttm_path = pred_rttm_dir / f"{stem}.rttm"

        audio_exists = audio_path.exists()
        gt_exists = gt_rttm_path.exists()
        pred_exists = pred_rttm_path.exists()

        duration = load_audio_duration(audio_path) if audio_exists else None
        gt_stats = parse_rttm(gt_rttm_path) if gt_exists else SegmentStats(0, 0, 0.0)
        pred_stats = parse_rttm(pred_rttm_path) if pred_exists else SegmentStats(0, 0, 0.0)

        audio_rel = ensure_relative(audio_path, project_root)
        gt_rttm_rel = ensure_relative(gt_rttm_path, project_root) if gt_exists else None
        pred_rttm_rel = ensure_relative(pred_rttm_path, project_root) if pred_exists else None

        manifest_entry = build_manifest_entry(
            audio_relative=audio_rel,
            audio_exists=audio_exists,
            duration=duration,
            gt_stats=gt_stats,
            gt_rttm_relative=gt_rttm_rel,
        )
        if manifest_entry:
            manifest_entries.append(manifest_entry)

        label_rows.append({
            "file_name": stem,
            "audio_path": audio_rel,
            "audio_exists": audio_exists,
            "duration_sec": round(duration, 6) if duration is not None else None,
            "gt_rttm_path": gt_rttm_rel,
            "gt_exists": gt_exists,
            "gt_segments": gt_stats.segment_count,
            "gt_unique_speakers": gt_stats.unique_speakers,
            "gt_total_speech_sec": round(gt_stats.total_speech_sec, 6),
            "pred_rttm_path": pred_rttm_rel,
            "pred_exists": pred_exists,
            "pred_segments": pred_stats.segment_count,
            "pred_unique_speakers": pred_stats.unique_speakers,
            "pred_total_speech_sec": round(pred_stats.total_speech_sec, 6),
        })

    write_manifest(manifest_entries, manifest_path)
    write_label_csv(label_rows, label_csv_path)
    write_der_csv(der_csv_path, der_sources, project_root)

    print(f"[INFO] Manifest updated: {manifest_path}")
    print(f"[INFO] Label summary CSV created: {label_csv_path}")
    if der_csv_path.exists():
        print(f"[INFO] DER CSV created: {der_csv_path}")
    else:
        print("[INFO] No DER CSV generated because no matching source files were found.")


if __name__ == "__main__":
    main()
