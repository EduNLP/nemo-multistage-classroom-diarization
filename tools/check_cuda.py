#!/usr/bin/env python
import os
import sys
import subprocess


def main() -> int:
    print('[CUDA Check] Python:', sys.version.split()[0])

    try:
        import torch
    except Exception as e:
        print('[CUDA Check] torch import FAILED:', e)
        return 1

    print('[CUDA Check] torch:', getattr(torch, '__version__', 'unknown'))
    print('[CUDA Check] torch.version.cuda:', getattr(torch.version, 'cuda', None))
    print('[CUDA Check] cuda.is_available:', torch.cuda.is_available())
    print('[CUDA Check] cudnn.enabled:', getattr(torch.backends, 'cudnn', None) and torch.backends.cudnn.enabled)

    if torch.cuda.is_available():
        try:
            num = torch.cuda.device_count()
            print('[CUDA Check] device_count:', num)
            for i in range(num):
                props = torch.cuda.get_device_properties(i)
                gb = props.total_memory / (1024**3)
                print(f'  - cuda:{i} -> {props.name} (CC {props.major}.{props.minor}, {gb:.2f} GB)')
        except Exception as e:
            print('[CUDA Check] device query error:', e)
    else:
        print('[CUDA Check] CUDA not available in torch. DLL/runtime or build mismatch is likely.')

    # Try nvidia-smi
    try:
        proc = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            lines = (proc.stdout or '').strip().splitlines()
            print('[CUDA Check] nvidia-smi -L:')
            for ln in lines[:5]:
                print('  ', ln)
        else:
            print('[CUDA Check] nvidia-smi not available or failed.')
    except Exception:
        print('[CUDA Check] nvidia-smi not found in PATH.')

    # Helpful envs
    for k in ('CUDA_PATH', 'CUDA_HOME'): 
        if k in os.environ:
            print(f'[CUDA Check] env {k}:', os.environ[k])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

