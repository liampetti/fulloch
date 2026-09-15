"""Orukeet ASR backend using its pinned NeMo checkpoint artifact.

Orukeet is Parakeet-derived but is a distinct Hugging Face model: it ships a
named `.nemo` file, so it must be restored from Fulloch's setup-downloaded hub
cache rather than passed to NeMo's generic ``from_pretrained`` loader.
"""

import logging
from pathlib import Path
from typing import Optional

from .asr import stream_generator  # noqa: F401 - backend module contract
from .asr_parakeet import ParakeetASRPipelineWrapper

logger = logging.getLogger(__name__)

ORUKEET_REPO = "oruk/orukeet"
ORUKEET_FILE = "orukeet-v0.1.0.nemo"
ORUKEET_REVISION = "555136b50265a132d4cea0d35560c26fc4f657ab"


def load_asr_model(model_name: Optional[str] = None, language: Optional[str] = None, **opts):
    """Restore the setup-cached Orukeet checkpoint on CUDA without network access."""
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel

    if not torch.cuda.is_available():
        raise RuntimeError("Orukeet requires CUDA in Fulloch's GPU image")
    model_name = model_name or ORUKEET_REPO
    checkpoint = Path(model_name)
    if not checkpoint.is_file():
        checkpoint = Path(
            hf_hub_download(
                repo_id=model_name,
                filename=opts.get("checkpoint_file", ORUKEET_FILE),
                revision=opts.get("revision", ORUKEET_REVISION),
                local_files_only=True,
            )
        )
    logger.info("Restoring cached Orukeet checkpoint %s on CUDA", checkpoint)
    model = ASRModel.restore_from(str(checkpoint))
    model.eval().freeze()
    model.cuda()
    return ParakeetASRPipelineWrapper(model)
