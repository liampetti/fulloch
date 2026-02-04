"""
Main assistant orchestration module.

Handles wakeword detection, intent processing, and response generation.
"""

import json
import logging
import random
import threading
from typing import Optional

from .audio import AudioCapture
from .tts import speak_stream, remove_emoji
from .slm import load_slm, generate_slm

from utils.system_prompts import getIntentSystemPrompt, getChatSystemPrompt
from utils.intent_catch import catchAll
import utils.intents as intents

logger = logging.getLogger(__name__)


class Assistant:
    """
    Main voice assistant class that orchestrates all components.

    Attributes:
        wakeword: The activation phrase
        audio_capture: Audio capture instance
        asr_pipe: Speech recognition pipeline
        slm_model: Language model (optional)
        grammar: JSON grammar for structured output
    """

    def __init__(self, wakeword: str, use_ai: bool, use_tiny_asr: bool = False):
        """
        Initialize the assistant.

        Args:
            wakeword: Activation phrase to listen for
            use_ai: Whether to use the SLM for intent detection
            use_tiny_asr: Whether to use Moonshine Tiny ASR instead of Qwen ASR
        """
        self.wakeword = wakeword.lower()
        self.use_ai = use_ai
        self.use_tiny_asr = use_tiny_asr
        self.audio_capture = AudioCapture()

        # Models loaded lazily in transcriber thread
        self.asr_pipe = None
        self.asr_stream_generator = None
        self.slm_model = None
        self.grammar = None
        self.intent_prompt = None
        self.chat_prompt = None

    def _load_models(self):
        """Load ASR and optionally SLM models."""
        if self.use_tiny_asr:
            from .asr_tiny import load_asr_model, stream_generator
            logger.info("Using Moonshine Tiny ASR")
        else:
            from .asr import load_asr_model, stream_generator
            logger.info("Using Qwen ASR")

        self.asr_pipe = load_asr_model()
        self.asr_stream_generator = stream_generator

        if self.use_ai:
            self.grammar, self.slm_model = load_slm()
            self.intent_prompt = getIntentSystemPrompt()
            self.chat_prompt = getChatSystemPrompt()

    def _handle_wakeword(self, user_prompt: str) -> str:
        """
        Process user input after wakeword detection.

        Args:
            user_prompt: Text spoken after the wakeword

        Returns:
            Response text to speak
        """
        answer = ""
        run_chat = False

        logger.info(f"WAKEWORD detected: {user_prompt}")

        # Try regex intent catch first (fast path)
        caught = catchAll(user_prompt)

        if isinstance(caught, dict):
            # Regex matched an intent
            logger.info(f"Regex caught intent: {caught}")
            answer = intents.handle_intent(caught)
            logger.info(f"Intent answer: {answer}")

        elif self.slm_model:
            # Use AI for intent detection
            logger.info(f"AI intent query: {user_prompt}")

            answer = generate_slm(
                self.slm_model,
                user_prompt=user_prompt,
                grammar=self.grammar,
                system_prompt=self.intent_prompt,
            )

            logger.info(f"AI answer: {answer}")

            if answer.strip('"'):
                try:
                    parsed = json.loads(answer)
                    logger.info(f"AI generated intent: {parsed}")
                    answer = intents.handle_intent(parsed)
                    logger.info(f"Intent answer: {answer}")

                    if "User question:" in answer:
                        user_prompt = answer
                        run_chat = True
                except Exception as e:
                    logger.error(f"Unable to load intent from {answer}: {e}")
                    answer = ""
            else:
                run_chat = True

            if run_chat:
                # Provide feedback while processing
                speak_stream(random.choice([
                    "Okay, let me think about that.",
                    "Just a second.",
                    "Got it, let me think.",
                    "Let's see."
                ]))

                logger.info(f"AI chat query: {user_prompt}")
                answer = generate_slm(
                    self.slm_model,
                    user_prompt=user_prompt,
                    system_prompt=self.chat_prompt,
                )
                logger.info(f"AI chat answer: {answer}")

        # Fallback if no answer
        if not answer.strip('"'):
            answer = random.choice([
                "Sorry, can you repeat that",
                "I don't understand",
                "Sorry, I didn't hear you properly",
                "Can you say that again?"
            ])

        return answer

    def _transcriber_thread(self):
        """
        Main transcription thread that processes audio and responds.
        """
        self._load_models()
        logger.info("Transcriber started")

        for result in self.asr_pipe(
            self.asr_stream_generator(self.audio_capture.audio_queue),
            batch_size=1,
            generate_kwargs={"max_new_tokens": 256}
        ):
            try:
                text = result.get("text", "").strip()
                if not text:
                    continue

                logger.debug(f"Transcribed: {text}")

                if self.wakeword not in text.lower():
                    continue

                # Extract text after wakeword
                user_prompt = text.lower().split(self.wakeword)[1].strip(",. ").replace('"', '')

                if not user_prompt:
                    logger.debug("Nothing after wakeword")
                    continue

                # Pause transcription while processing
                self.audio_capture.transcribing = False

                # Process and respond
                answer = self._handle_wakeword(user_prompt)
                cleaned = remove_emoji(answer.replace('"', '').replace('*', ''))
                speak_stream(cleaned)

                # Resume transcription
                self.audio_capture.transcribing = True

            except Exception as e:
                logger.error(f"Transcription error: {e}")

    def run(self):
        """
        Start the assistant and run until interrupted.
        """
        rec_thread = threading.Thread(
            target=self.audio_capture.recorder_thread,
            daemon=True
        )
        trans_thread = threading.Thread(
            target=self._transcriber_thread,
            daemon=True
        )

        rec_thread.start()
        trans_thread.start()

        logger.info("Press Ctrl+C to stop.")
        try:
            while True:
                import time
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.audio_capture.stop()
            rec_thread.join(timeout=2)
            trans_thread.join(timeout=2)
