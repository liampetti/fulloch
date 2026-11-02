# openWakeWord Models

This directory contains unmodified feature extractor artifacts from
[openWakeWord v0.5.1](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1):

- `embedding_model.tflite`
- `melspectrogram.tflite`
- `embedding_model.onnx`
- `melspectrogram.onnx`
- `alexa_v0.1.onnx` — Alexa wakeword classifier
- `hey_jarvis_v0.1.onnx` — Hey Jarvis wakeword classifier

Classifiers are from the same v0.5.1 release linked above, downloaded unmodified
from `https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/<filename>`.
They run locally; choosing Alexa does not connect Fulloch to Amazon.

The openWakeWord project states that its included pre-trained models are
licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
These files are not covered by Fulloch's MIT license. They are included only
for non-commercial use under CC BY-NC-SA 4.0. The openWakeWord runtime code is
separately licensed under Apache-2.0.

Source: David Scripka and openWakeWord contributors,
<https://github.com/dscripka/openWakeWord>.

SHA-256:

```text
c0aea21eb84a4ce90a08c870da41b7a7173b45269e6a3207c71d67c40f3a59d8  embedding_model.tflite
96fa0adccb6e8cf95cb14465409a1a2898ee4a96a85bb9ed3c7eb0e68bf163e8  melspectrogram.tflite
70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f  embedding_model.onnx
ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f  melspectrogram.onnx
6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e  alexa_v0.1.onnx
94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb  hey_jarvis_v0.1.onnx
```
