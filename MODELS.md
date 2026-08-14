# Model Sources and Licenses

Fulloch downloads model assets during setup. The selected backend determines which sources are used. All listed Hugging Face repositories are public at the time of writing except where explicitly marked gated below.

## Speech Recognition

- Qwen3-ASR PyTorch: [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).
- Qwen3-ASR ONNX: [1.7B](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) and [0.6B](https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU).
- Qwen3-ASR GGUF: [1.7B](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) and [0.6B](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF).
- Moonshine: [Base](https://huggingface.co/UsefulSensors/moonshine-base) and [Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny).

## Wakeword

- Hey Atticus v0.3: repository-provided custom openWakeWord ONNX classifier at
  `data/models/wakeword/hey_atticus_v0.3.onnx`. Docker images seed it into the
  persistent data volume on first run, so the default wizard preset works without
  a model download. Fulloch uses it only as an optional candidate gate and
  verifies each candidate with ASR. The openWakeWord runtime is
  [Apache-2.0](https://github.com/dscripka/openWakeWord/blob/main/LICENSE).
- Feature extractors: Fulloch images and this repository include the unmodified
  `embedding_model` and `melspectrogram` ONNX/TFLite files from
  [openWakeWord v0.5.1](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1).
  They are pre-trained models licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), not
  Fulloch's MIT license, and may be used only subject to that license. Their
  attribution, source, and SHA-256 checksums are in
  [`third_party/openwakeword-models/NOTICE.md`](third_party/openwakeword-models/NOTICE.md).

## Text to Speech

- Qwen3-TTS PyTorch: [1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) and [0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
- Qwen3-TTS GGUF: [1.7B](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF), [0.6B](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF), and [tokenizer](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF).
- Kokoro: [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).
- Pocket TTS ONNX: [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx), using the English INT8 bundle for experimental CPU one-shot voice cloning (CC-BY-4.0 model; Apache-2.0 wrapper code).
- Pocket TTS PyTorch: [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts), using the official English 2026-04 PyTorch model for experimental CUDA voice cloning with native PCM streaming (CC-BY-4.0). **Gated:** accept the model terms on Hugging Face before downloading. Fulloch downloads only the revision-pinned English weights and tokenizer.
- Pocket TTS GGUF: [cstr/pocket-tts-GGUF](https://huggingface.co/cstr/pocket-tts-GGUF), using the English Q8_0 voice-cloning build for experimental GPU use through CrispASR (CC-BY-4.0).
- Higgs TTS 3: [Fulloch mirror](https://huggingface.co/liampetti/HiggsTTS3.gguf), using `higgs-v3-tts-q4_k.gguf` and `higgs_tts_v3_tokenizer.json`. The mirror preserves provenance from the [original GGUF conversion](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) and is derived from [Boson AI's Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b).

## Language Models

- Local Qwen 9B GGUF: [unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF), using the `UD-Q4_K_XL` quant. MTP speculative decoding is an optional runtime setting.
- Optional Gemma 4 GGUF: [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF), using the `UD-Q4_K_XL` quant.

## Higgs TTS 3 License

Higgs TTS 3 is source-available under Boson's Research and Non-Commercial License, not the Fulloch MIT license. Fulloch offers it as an optional, self-hosted TTS backend for permitted research and non-commercial use. Commercial products, hosted TTS, APIs, and services require a separate license from Boson AI.

The Higgs GGUF mirror must distribute Boson's full license and upstream `NOTICE` file. Fulloch documentation and public-facing Higgs voice output identify it as: `Built with Higgs TTS 3 licensed from Boson AI USA, Inc.` The backend/model documentation is: `Derived from Higgs TTS 3, licensed from Boson AI USA, Inc.`

Only provide a reference recording when you have that person's explicit, verifiable consent. Users remain responsible for applicable synthetic-audio disclosure and consent obligations.

## Hugging Face Access

The experimental `pocket-tts-pytorch` backend uses Kyutai's official PyTorch implementation on CUDA and streams PCM while it generates. It is intended for benchmarking low time-to-first-audio on supported NVIDIA GPUs. Its `kyutai/pocket-tts` cloning weights are the only currently configured gated model asset; the tokenizer comes from the separate public `kyutai/pocket-tts-without-voice-cloning` repository. The GGUF and ONNX Pocket options use independent public conversions and do not require this gated download.

Before downloading Pocket TTS PyTorch, accept the [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts) terms while signed in to Hugging Face. If Hugging Face returns an authentication, authorization, or gated-access denial during setup or reconfiguration, Fulloch displays a token field on the failed download screen. Create a read token at [Hugging Face settings](https://huggingface.co/settings/tokens), paste it into that field, and retry. Fulloch stores it as `hf_token` in `data/credentials.json` and provides it to Hugging Face as `HF_TOKEN`.

For headless use, pass `-e HF_TOKEN=hf_...` to Docker. An explicit environment variable takes precedence over the saved credential. The generic denial handling also prompts for a token if a future selected Hugging Face asset becomes gated; accepting that repository's terms is still required.

## Higgs Creator Use Grant

Higgs TTS 3's Creator Use Grant permits digital creators to make and monetize podcasts, videos, and social posts at no charge when they credit Boson AI's Higgs Audio in the audio or accompanying text.

Suggested credit: `This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio`

This grant concerns creator output. Before distributing model weights, the Higgs runtime, or a container containing either, review the applicable model, code, and GGML licenses independently.
