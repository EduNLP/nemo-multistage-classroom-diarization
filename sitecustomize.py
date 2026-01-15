"""
Windows compatibility tweaks loaded automatically by Python if present on sys.path.

Ensures libraries that reference POSIX-only signal names (e.g., SIGKILL) can
import successfully on Windows by mapping them to the closest available signal.
"""
import signal

# Some libraries (e.g., NeMo exp_manager) reference SIGKILL which is not defined on Windows.
if not hasattr(signal, "SIGKILL") and hasattr(signal, "SIGTERM"):
    try:
        signal.SIGKILL = signal.SIGTERM  # type: ignore[attr-defined]
    except Exception:
        pass

