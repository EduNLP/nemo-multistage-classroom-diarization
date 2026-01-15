#!/usr/bin/env python
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise SystemExit("huggingface_hub is required. Install via: pip install huggingface_hub")


def main():
    ap = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to a local directory.")
    ap.add_argument("model_id", help="Model ID, e.g., MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    ap.add_argument("out_dir", nargs="?", default=None, help="Target directory (defaults to models/hf/zero-shot/<model_id_basename>)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    if args.out_dir:
        out = Path(args.out_dir)
    else:
        name = args.model_id.split("/")[-1]
        out = base / "models" / "hf" / "zero-shot" / name
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.model_id}' to '{out}' ...")
    snapshot_download(repo_id=args.model_id, local_dir=str(out), local_dir_use_symlinks=False)
    print("Done.")
    print(str(out))


if __name__ == "__main__":
    main()

