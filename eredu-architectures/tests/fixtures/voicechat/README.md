# VoiceChat header fixtures

`official-header.json` is the exact JSON header obtained with byte range
`8-216023` from the pinned official `model.safetensors`. The eight-byte length
prefix is 216016. No model payload is included. `provenance.json` identifies the
immutable source revision and header digest.

The seven `gguf-*-tensors.json` fixtures contain every tensor name, logical
row-major shape (reversed from GGUF dimension order), and numeric GGML type in
the corresponding published directory. They are extracted from bounded 16 MiB
prefix downloads; the largest directory ends at byte 7883801. Metadata and
weight payloads are omitted. The exact source paths, revisions and fixture
digests appear in `provenance.json`.

The SafeTensors fixture covers 1632 tensors including 6 I64 control/lookup
buffers. Every complete GGUF variant preserves the same 1632 identities. The
four split files contain 640 perception, 341 backbone, 1 function-head and 559
TTS/codec tensors. Two TTS character tables are derived from the tokenizer;
15 RNNT tensors, 76 codec encoder tensors and two scalar padding buffers are
absent from those split files.

Canonical split recipes reverse the parameter operations documented by the
community converter sources at `sansamour/llama-voicechat.cpp` revision
`f45001fc3d8013c72beb6753d3eb0b976b6a9fff`: `convert_voicechat_to_nemotron_h.py`,
`convert_voicechat_perception_to_mmproj.py`, and
`convert_voicechat_tts_to_gguf.py`. The published artifact revision does not
identify its converter commit. These sources substantiate the transformations;
header fixtures independently establish the published names and geometry.

The `gguf-*-metadata.json` fixtures retain published scalar metadata,
per-layer arrays and complete embedded NeMo configuration. The large embedded
backbone tokenizer string arrays and chat template are omitted here; source
preparation compares them with the retained tokenizer assets. These fixtures
exercise conflicting geometry, numerical defaults and special-token IDs.

Header admission does not validate tensor values or establish numerical
execution parity. In particular, quantized codebook representations can differ
from separately retained dense codebook buffers.
