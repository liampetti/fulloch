# Model Sources and Licenses

Fulloch downloads model assets during setup. The selected backend determines which sources are used; gated assets are marked below.

## Speech Recognition

- Qwen3-ASR PyTorch: [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).
- Qwen3-ASR ONNX: [1.7B](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) and [0.6B](https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU).
- Parakeet TDT v3 ONNX: [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) (CC-BY-4.0; multilingual).
- Qwen3-ASR GGUF: [1.7B](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) and [0.6B](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF).
- Moonshine: [Base](https://huggingface.co/UsefulSensors/moonshine-base) and [Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny).

## Wakeword

- Hey Atticus: repository-provided custom openWakeWord ONNX classifiers named
  `data/models/wakeword/hey_atticus_v*.onnx`. Docker images seed every bundled
  version into the persistent data volume on first run, so configuration can
  select a model filename without a model download. Fulloch uses it only as an
  acoustic candidate gate. ASR verification is on by default except for
  Moonshine Base/Tiny and Parakeet ONNX, which cannot reliably recover unusual
  wakeword names. Custom activation remains available in config.yml only. The openWakeWord runtime is
  [Apache-2.0](https://github.com/dscripka/openWakeWord/blob/main/LICENSE).
- Alexa and Hey Jarvis: official unmodified `alexa_v0.1.onnx` and
  `hey_jarvis_v0.1.onnx` classifiers from the openWakeWord v0.5.1 release below,
  bundled in `third_party/openwakeword-models/` for native and Docker use.
  **CC BY-NC-SA 4.0 (non-commercial)**, separate from the application license.
  Selectable alongside Hey Atticus in the wizard and Settings; no runtime
  model download or cloud voice-service connection is needed.
- Feature extractors: Fulloch images and this repository include the unmodified
  `embedding_model` and `melspectrogram` ONNX/TFLite files from
  [openWakeWord v0.5.1](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1).
  They are pre-trained models licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), not
  Fulloch's MIT license, and may be used only subject to that license. Their
  attribution, source, and SHA-256 checksums are in
  [`third_party/openwakeword-models/NOTICE.md`](third_party/openwakeword-models/NOTICE.md).

## Text to Speech

- Audio8 TTS Realistic Air: [Edge0/Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air), an experimental CUDA voice-cloning backend (Apache-2.0). Its pinned Transformers implementation emits complete sentence fragments rather than native live audio frames.
- Audio8 TTS Preview 0.1B: [Edge0/Audio8-TTS-Preview-0.1b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b), an experimental CUDA voice-cloning backend. Its [Audio8 Community License v1.0](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b/blob/8c837020ab800498834038363dea06757872afda/LICENSE) permits non-commercial use and commercial use by entities with annual revenue below US$2 million; entities at or above that threshold need a separate written commercial license. Downloading or using the model is subject to its license, including consent and attribution requirements. English and Chinese are the primary languages; other languages are experimental. Like Realistic Air, this Transformers backend emits complete sentence fragments, not live audio frames.
- Qwen3-TTS PyTorch: [1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) and [0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
- Qwen3-TTS GGUF: [1.7B](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF), [0.6B](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF), and [tokenizer](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF).
- Kokoro: [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).
- Pocket TTS ONNX: [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx), using the English INT8 bundle for CPU one-shot voice cloning (CC-BY-4.0 model; Apache-2.0 wrapper code).
- Pocket TTS PyTorch: Kyutai's CPU/GPU voice-cloning runtime with native PCM streaming, using the [ungated pinned lookbe collection](https://huggingface.co/lookbe/pocket-tts-6l). Premium's default; no Hugging Face account or token required. See provenance and consent below.

- Pocket TTS GGUF: [cstr/pocket-tts-GGUF](https://huggingface.co/cstr/pocket-tts-GGUF), using the English Q8_0 voice-cloning build for GPU use through CrispASR (CC-BY-4.0).
- Higgs TTS 3: [Fulloch mirror](https://huggingface.co/liampetti/HiggsTTS3.gguf), using `higgs-v3-tts-q4_k.gguf` and `higgs_tts_v3_tokenizer.json`. The mirror preserves provenance from the [original GGUF conversion](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) and is derived from [Boson AI's Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b).
- Breeze TTS 2: [cstr/breeze-tts-2-GGUF](https://huggingface.co/cstr/breeze-tts-2-GGUF), using `breeze-tts-2-q4_k.gguf` with the [Qwen3 TTS 12 Hz tokenizer](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF). It runs through the GPU CrispASR `bt2-tts` backend and supports Breeze's documented English audio tags.

## Language Models

- Ornith 1.5 9B GGUF: [ornith-ai/Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF), using `Ornith-1.5-9B-Q4_K_M.gguf`. Default for Pro, Premium and Economy.
- Local Qwen 9B GGUF: [unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF), using the `UD-Q4_K_XL` quant. MTP speculative decoding is an optional runtime setting.
- Optional Gemma 4 GGUF: [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF), using the `UD-Q4_K_XL` quant.
- Qwen3.8 27B GGUF: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), using `Qwen3.8-27B-UD-IQ4_XS.gguf` (the Max tier's "God-tier" agent model; the full Qwen 1.7B ASR/TTS stack targets a 32 GB GPU; no MTP draft head). A custom Orukeet + Pocket speech stack can target 24 GB instead.

Max defaults to Qwen3.8 27B; Lite uses Laya and regex rather than a generative language model. See the preset table in [README.md](README.md#model-options) for the default speech backends.

## Semantic Command Routing

- Laya: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), a CPU-only local semantic command router. Fulloch runs its regex fast-path first, then Laya may select from supported action schemas after a miss; it does not provide free-form conversation or generate arbitrary tool arguments.

## Higgs TTS 3 License

Higgs TTS 3 is source-available under Boson's Research and Non-Commercial License, not the Fulloch MIT license. Fulloch offers it as an optional, self-hosted TTS backend for permitted research and non-commercial use. Commercial products, hosted TTS, APIs, and services require a separate license from Boson AI.

The Higgs GGUF mirror must distribute Boson's full license and upstream `NOTICE` file. Fulloch documentation and public-facing Higgs voice output identify it as: `Built with Higgs TTS 3 licensed from Boson AI USA, Inc.` The backend/model documentation is: `Derived from Higgs TTS 3, licensed from Boson AI USA, Inc.`

Only provide a reference recording when you have that person's explicit, verifiable consent. Users remain responsible for applicable synthetic-audio disclosure and consent obligations.

## Breeze TTS 2 License

Breeze TTS 2 weights and derivative GGUFs are licensed under the BreezeBlue Research and Non-Commercial License Agreement, not the Fulloch MIT license. They may be used only for permitted research and non-commercial self-hosted use. Only provide a voice reference when you have that person's explicit, verifiable consent.

## Pocket TTS PyTorch provenance and consent

The `pocket-tts-pytorch` backend runs the [Kyutai Labs Pocket TTS runtime](https://github.com/kyutai-labs/pocket-tts) with multilingual voice-cloning weights. Kyutai's original weights are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); credit Kyutai Labs and link to the license and [original model](https://huggingface.co/kyutai/pocket-tts). The runtime code has its own license. Fulloch obtains weights and tokenizers from [lookbe/pocket-tts-6l](https://huggingface.co/lookbe/pocket-tts-6l), pinned to revision `eda74b8dfa99876b660e95088fbf30f8af6e1f44`; its English weights match Kyutai's `english_2026-09` and its French weights match Kyutai's six-layer French checkpoint. Other third-party files have not all been independently verified. No external model code is executed; mirror availability is not guaranteed. Some language folders contain third-party fine-tunes with their own terms; check their provenance and license before use. Users may configure local compatible files instead.

The collection identifies its Wolof weights as derived from [abdazizsall/wolof-pocket-tts](https://huggingface.co/abdazizsall/wolof-pocket-tts) under CC BY-SA 4.0; check the original model's terms. Fulloch does not include that model's Wolof-specific text normalization or number expansion. The Hindi and Egyptian fine-tunes' full provenance and terms have not been independently verified; check before deployment.

**Consent and responsible use:** only clone your own voice or a voice for which you have the person's explicit, lawful consent. Do not impersonate people deceptively or use generated audio for fraud, misinformation, unlawful, harmful, or privacy-invasive purposes. Respect applicable laws, disclosure requirements and each reference recording's license. A model license does not grant rights to a person's voice or to individual reference recordings. The ONNX and GGUF Pocket backends are separate public conversions with their own source documentation.

The generic Hugging Face denial handler remains available if another selected asset requires access; in that case the setup screen may request a read token after the user accepts that repository's terms. The Premium Pocket PyTorch model itself does not require one.

## Higgs Creator Use Grant

Higgs TTS 3's Creator Use Grant permits digital creators to make and monetize podcasts, videos, and social posts at no charge when they credit Boson AI's Higgs Audio in the audio or accompanying text.

Suggested credit: `This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio`

This grant concerns creator output. Before distributing model weights, the Higgs runtime, or a container containing either, review the applicable model, code, and GGML licenses independently.
