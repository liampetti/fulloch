# Model Sources and Licenses

Fulloch downloads model assets during setup. The selected backend determines which sources are used.

## Speech Recognition

- Qwen3-ASR PyTorch: [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).
- Qwen3-ASR ONNX: [1.7B](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) and [0.6B](https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU).
- Qwen3-ASR GGUF: [1.7B](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) and [0.6B](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF).
- Moonshine: [Base](https://huggingface.co/UsefulSensors/moonshine-base) and [Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny).

## Text to Speech

- Qwen3-TTS PyTorch: [1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) and [0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
- Qwen3-TTS GGUF: [1.7B](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF), [0.6B](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF), and [tokenizer](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF).
- Kokoro: [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).
- Pocket TTS ONNX: [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx), using the English INT8 bundle for experimental CPU one-shot voice cloning (CC-BY-4.0 model; Apache-2.0 wrapper code).
- Pocket TTS GGUF: [cstr/pocket-tts-GGUF](https://huggingface.co/cstr/pocket-tts-GGUF), using the English Q8_0 voice-cloning build for experimental GPU use through CrispASR (CC-BY-4.0).
- Higgs TTS 3: [Fulloch mirror](https://huggingface.co/liampetti/HiggsTTS3.gguf), using `higgs-v3-tts-q4_k.gguf` and `higgs_tts_v3_tokenizer.json`. The mirror preserves provenance from the [original GGUF conversion](https://huggingface.co/NeemaShioSe/HiggsTTS3.gguf) and is derived from [Boson AI's Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b).

## Language Models

- Local Qwen 9B GGUF: [unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF), using the `UD-Q4_K_XL` quant with native MTP speculative decoding.
- Optional Gemma 4 GGUF: [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF), using the `UD-Q4_K_XL` quant.

## Higgs TTS 3 License

Higgs TTS 3 is source-available under Boson's Research and Non-Commercial License, not the Fulloch MIT license. Fulloch offers it as an optional, self-hosted TTS backend for permitted research and non-commercial use. Commercial products, hosted TTS, APIs, and services require a separate license from Boson AI.

The Higgs GGUF mirror must distribute Boson's full license and upstream `NOTICE` file. Fulloch documentation and public-facing Higgs voice output identify it as: `Built with Higgs TTS 3 licensed from Boson AI USA, Inc.` The backend/model documentation is: `Derived from Higgs TTS 3, licensed from Boson AI USA, Inc.`

Only provide a reference recording when you have that person's explicit, verifiable consent. Users remain responsible for applicable synthetic-audio disclosure and consent obligations.

## Higgs Creator Use Grant

Higgs TTS 3's Creator Use Grant permits digital creators to make and monetize podcasts, videos, and social posts at no charge when they credit Boson AI's Higgs Audio in the audio or accompanying text.

Suggested credit: `This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio`

This grant concerns creator output. Before distributing model weights, the Higgs runtime, or a container containing either, review the applicable model, code, and GGML licenses independently.
