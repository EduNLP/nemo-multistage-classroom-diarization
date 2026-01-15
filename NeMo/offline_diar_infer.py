import signal
if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM

import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
This script demonstrates how to use run speaker diarization.
Usage:
  python offline_diar_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.vad.model_path='vad_marblenet' \
    diarizer.speaker_embeddings.parameters.save_embeddings=False

Check out whole parameters in ./conf/offline_diarization.yaml and their meanings.
For details, have a look at <NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
"""

seed_everything(42)


@hydra_runner(config_path="conf/inference", config_name="diar_infer_meeting.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device is None:
        cfg.device = auto_device
    elif str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        cfg.device = "cpu"

    logging.info(f"Using device: {cfg.device}")

    dry_run = cfg.get("dry_run", False)
    if dry_run:
        logging.info("Dry-run mode enabled; skipping diarization pipeline.")
        OmegaConf.set_struct(cfg, True)
        return

    OmegaConf.set_struct(cfg, True)

    sd_model = ClusteringDiarizer(cfg=cfg).to(cfg.device)
    sd_model.diarize()


if __name__ == '__main__':
    main()
