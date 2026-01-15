from __future__ import annotations

from typing import Optional, Tuple


def _match_length(signal, target_len: int):
    import numpy as np
    if len(signal) == target_len:
        return signal
    if len(signal) > target_len:
        return signal[:target_len]
    pad = target_len - len(signal)
    return np.pad(signal, (0, pad), mode='constant')


def denoise_dfnet3(y, sr: int, enable: bool = True, device: str = 'cpu') -> Tuple[Optional[object], str]:
    """
    DeepFilterNet3 denoise (pre-process):
    - Uses df API (init_df/enhance), resamples to 48k and back, preserves length.
    - Runs on CPU by default to avoid mixed-device issues.
    """
    if not enable:
        return None, 'disabled'

    # Robust numpy/librosa import (handle Windows soxr issues)
    try:
        import os, sys
        os.environ.setdefault('LIBROSA_RESAMPLER', 'resampy')
        try:
            import numpy as np  # noqa: F401
            import librosa
        except Exception:
            # Provide a benign stub for 'soxr' if librosa tries to import it
            import types as _types, importlib
            sys.modules.setdefault('soxr', _types.SimpleNamespace(__version__='0'))
            import numpy as np  # noqa: F401
            librosa = importlib.import_module('librosa')
    except Exception as e:
        return None, f'missing numpy/librosa ({e})'

    # 1) Prefer `df` package API (shipped by DeepFilterNet pip)
    try:
        # Hint device selection for DeepFilterNet before import/init
        try:
            import os as _os
            _os.environ.setdefault('DF_DEVICE', 'cuda' if ('cuda' in device.lower()) else 'cpu')
        except Exception:
            pass
        from df.enhance import enhance, init_df
        try:
            import torch  # type: ignore
        except Exception as e:
            return None, f'missing torch ({e})'

        def _move_to_device(obj, device):
            try:
                import torch as _torch
            except Exception:
                return obj
            if isinstance(obj, _torch.Tensor):
                return obj.to(device)
            if hasattr(obj, 'to') and callable(getattr(obj, 'to')):
                try:
                    return obj.to(device)
                except Exception:
                    pass
            if isinstance(obj, dict):
                return {k: _move_to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                t = type(obj)
                return t(_move_to_device(v, device) for v in obj)
            if hasattr(obj, '__dict__'):
                for k, v in list(obj.__dict__.items()):
                    try:
                        setattr(obj, k, _move_to_device(v, device))
                    except Exception:
                        pass
            return obj

        try:
            model, df_state, _ = init_df()
        except Exception as e:
            return None, f'df init failed: {type(e).__name__}: {e}'

        try:
            target_sr = 48000
            x48 = y.astype('float32')
            if sr != target_sr:
                x48 = librosa.resample(x48, orig_sr=sr, target_sr=target_sr).astype('float32')
            # Force a single device strictly from requested flag (default cpu)
            dev = torch.device('cuda') if (isinstance(device, str) and device.lower().startswith('cuda') and torch.cuda.is_available()) else torch.device('cpu')
            try:
                model = model.to(dev)
            except Exception:
                pass
            try:
                df_state = _move_to_device(df_state, dev)
            except Exception:
                pass
            x48_t = torch.as_tensor(x48, dtype=torch.float32, device=dev)
            if x48_t.dim() == 1:
                x48_t = x48_t.unsqueeze(0)
            with torch.no_grad():
                out48_t = enhance(model, df_state, x48_t)
            if out48_t.dim() > 1:
                out48_t = out48_t.squeeze(0)
            out48 = out48_t.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
            out16 = out48
            if target_sr != sr:
                out16 = librosa.resample(out16, orig_sr=target_sr, target_sr=sr).astype('float32')
            out16 = _match_length(out16.astype('float32'), len(y))
            dev_str = dev.type if hasattr(dev, 'type') else str(dev)
            return out16, f'df package api (dev={dev_str})'
        except Exception as e:
            return None, f'df enhance failed: {type(e).__name__}: {e}'
    except Exception:
        pass

    # 2) Try official deepfilternet module (may not be importable)
    try:
        import deepfilternet as dfn

        try:
            model = dfn.DeepFilterNet.load_pretrained('deepfilternet3')
        except Exception:
            model = dfn.DeepFilterNet.from_pretrained('deepfilternet3')

        target_sr = getattr(model, 'sample_rate', 48000)
        x = y.astype('float32')
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        out = model.enhance(x, sr=target_sr)
        if target_sr != sr:
            out = librosa.resample(out.astype('float32'), orig_sr=target_sr, target_sr=sr)
        out = _match_length(out.astype('float32'), len(y))
        return out, 'deepfilternet api'
    except Exception as e:
        return None, f'deepfilternet failed: {type(e).__name__}: {e}'

    return None, 'DeepFilterNet3 not available'
