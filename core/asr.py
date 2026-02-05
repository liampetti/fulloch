import logging
import torch
import numpy as np
from typing import Generator, Optional, Union
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)

# Model configuration
ASR_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
SAMPLE_RATE = 16000 # Qwen3-ASR standard sample rate

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


class QwenASRPipelineWrapper:
    """
    Wrapper for Qwen3-ASR to mimic basic HF pipeline behavior for streaming.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, audio_input: Union[np.ndarray, Generator], batch_size: int = 1, generate_kwargs: Optional[dict] = None, **kwargs):
        logger.info(f"[Batch size {batch_size}] and {generate_kwargs} are not being used")

        if isinstance(audio_input, Generator):
            for chunk in audio_input:
                if chunk is not None:
                    if isinstance(chunk, np.ndarray):
                        audio_tuple = (chunk, SAMPLE_RATE)
                    elif isinstance(chunk, torch.Tensor):
                        audio_tuple = (chunk.cpu().numpy(), SAMPLE_RATE)
                    else:
                        audio_tuple = (np.array(chunk), SAMPLE_RATE)

                    try:
                        results = self.model.transcribe(
                            audio=[audio_tuple], 
                            return_time_stamps=False
                        )
                    except TypeError as e:
                        logger.warning(f"Transcribe argument error: {e}. Retrying.")
                        results = self.model.transcribe(
                            audio=[audio_tuple], 
                            return_time_stamps=False
                        )
                        
                    if isinstance(results, list):
                        for res in results:
                            yield {"text": getattr(res, 'text', str(res))}
                    else:
                        yield {"text": getattr(results, 'text', str(results))}

        else:
            if isinstance(audio_input, np.ndarray):
                audio_tuple = (audio_input, SAMPLE_RATE)
            else:
                 audio_tuple = (np.array(audio_input), SAMPLE_RATE)

            results = self.model.transcribe(audio=[audio_tuple])
            
            out_text = []
            if isinstance(results, list):
                out_text = [{"text": getattr(r, 'text', str(r))} for r in results]
            else:
                out_text = [{"text": getattr(results, 'text', str(results))}]
            return out_text


def load_asr_model():
    logger.info(f"Loading {ASR_MODEL_NAME} on {DEVICE}...")

    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_NAME,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="flash_attention_2",
    )

    return QwenASRPipelineWrapper(model)


def stream_generator(queue) -> Generator:
    """
    Generator that yields audio from a queue.
    """
    while True:
        audio = queue.get()
        if audio is None:
            break
        yield audio
