from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force CPU-only before any heavy import
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DF_DEVICE'] = 'cpu'
os.environ['DEVICE'] = 'cpu'

import soundfile as sf  # type: ignore

# Prefer resampy to avoid soxr DLL issues on Windows
os.environ.setdefault('LIBROSA_RESAMPLER', 'resampy')
try:
    import librosa  # type: ignore
except Exception:
    import types as _types, importlib  # type: ignore
    sys.modules.setdefault('soxr', _types.SimpleNamespace(__version__='0'))
    librosa = importlib.import_module('librosa')  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()

    # Always run on CPU for stability
    args.device = 'cpu'

    # Lazy import DeepFilterNet
    try:
        from df.enhance import enhance, init_df  # type: ignore
        import torch  # type: ignore
        # Disable cuDNN for safety (both CPU/CUDA)
        try:
            torch.backends.cudnn.enabled = False
        except Exception:
            pass
    except Exception as e:
        print(f"[df_denoise] missing deepfilternet/torch: {e}", file=sys.stderr)
        return 2

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype('float32', copy=False)

    # Resample to DF sample rate (48 kHz)
    target_sr = 48000
    if sr != target_sr:
        x48 = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype('float32')
    else:
        x48 = x

    # Initialize model/state on CPU
    model, state, _ = init_df()
    dev = torch.device('cpu')
    model = model.to(dev)
    
    def _move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        if hasattr(obj, 'to') and callable(getattr(obj, 'to')):
            try:
                return obj.to(dev)
            except Exception:
                return obj
        if isinstance(obj, dict):
            return {k: _move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_move_to_device(v) for v in obj)
        return obj

    state = _move_to_device(state)

    xt = torch.as_tensor(x48, dtype=torch.float32, device=dev).contiguous()
    if xt.dim() == 1:
        xt = xt.unsqueeze(0)
    try:
        with torch.no_grad():
            yt = enhance(model, state, xt)
    except Exception as e:
        # Fallback to CPU if CUDA path fails
        if dev.type == 'cuda':
            model = model.cpu()
            state = _move_to_device(state)
            xt = xt.cpu().contiguous()
            with torch.no_grad():
                yt = enhance(model, state, xt)
        else:
            raise
    if yt.dim() > 1:
        yt = yt.squeeze(0)
    y48 = yt.detach().cpu().to(dtype=torch.float32).contiguous().numpy()

    # Back to original sr
    if target_sr != sr:
        y = librosa.resample(y48, orig_sr=target_sr, target_sr=sr).astype('float32')
    else:
        y = y48
    # Trim/pad to match input length
    if len(y) > len(x):
        y = y[: len(x)]
    elif len(y) < len(x):
        import numpy as np
        y = np.pad(y, (0, len(x) - len(y)), mode='constant')

    sf.write(out, y, sr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
