from unittest.mock import MagicMock, patch

import numpy as np

from core import asr_tiny


def test_moonshine_transcription_uses_inference_watchdog():
    pipeline = MagicMock(return_value={"text": "hello"})
    wrapper = asr_tiny.MoonshineASRPipelineWrapper(pipeline)
    watchdog = MagicMock()
    watchdog.__enter__.return_value = watchdog
    watchdog.__exit__.return_value = False

    with patch.object(asr_tiny, "InferenceWatchdog", return_value=watchdog) as watchdog_ctor:
        assert wrapper._transcribe(np.zeros(160, dtype=np.float32), {}) == "hello"

    watchdog_ctor.assert_called_once_with("Moonshine ASR transcription")
