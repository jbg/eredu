# VoiceChat admission evidence

`official-inference-config.json` retains inference-relevant fields from NVIDIA's
[pinned configuration](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/blob/443794ea956ef0065f001967ffd00e77f519cb39/config.json).
Defaults follow [NVIDIA's reference implementation](https://github.com/NVIDIA-NeMo/Speech/tree/097dfe9e2f55baf653b83035868bdc89849f1b47)
and its README's Transformers 4.56.0 pin. NeMo class targets are declarative data.

Auxiliary assets use the official revision above and base model
`nvidia/NVIDIA-Nemotron-Nano-9B-v2` revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`.
`assets::VOICECHAT_ASSET_REQUESTS` records every filename, size and SHA-256.
The base configuration's EOS ID is 12; VoiceChat text-channel EOS is 2. The
published TTS `special_flags` buffer independently retains its EOS flag at 12.

Integer values are checked from bounded payload ranges, beyond tensor headers.
The following external fixtures identify exact HTTP byte ranges, inclusive:

| Fixture | Pinned source | Range | Bytes | SHA-256 |
|---|---|---|---:|---|
| `official-integer-buffers.bin` | Official `model.safetensors`, revision above | `216024-2313471` | 2097448 | `f03ba91519dacc89a28ab5dd08b0777afd6511dc07a2f84d7b6b1394662ca45b` |
| `split-character-buffers.bin` | Community `llamacpp/voicechat-tts-Q4_0.gguf`, revision `89883a05a031557729771f94abb9998e4facdd45` | `338055168-342144035` | 4088868 | `d5cf743b48e053462c4c1c1b1a997367a43a7f5e54df6de42e00d4b2f3b6fa91` |

`controls::VOICECHAT_INTEGER_BUFFERS` records per-tensor canonical I64 digests.
I32 split GGUF storage is sign-extended before comparison. The character tables
match an independent reconstruction from the retained tokenizer: single-character
tokens sorted by token ID define the character vocabulary; each token contributes
its matching characters in order. `char_ids` has 891139 values and `char_offsets`
has 131073. The two tables occupy byte offsets 0 and 3564576 in their external
fixture; their canonical I64 digests are recorded in
`controls::VOICECHAT_SPLIT_CHARACTER_BUFFERS`.

The split backbone's complete token list, ordered merges, token types and chat
template match the retained base assets with VoiceChat control-token overrides.
Array identity checks use length-delimited UTF-8 values and exact integer values.
These checks establish artifact admission, not numerical inference parity.
Floating preprocessing buffers, speaker conditioning, and duplicated RVQ values
are learned payloads; shape admission alone does not establish their numerical
equivalence across quantized representations.

The opt-in tests use downloaded fixtures outside the source tree. `official-header.json`
is the official SafeTensors JSON header, and `gguf-2-header.json` is the extracted
split-backbone metadata JSON with a `metadata` object. The official and base asset
directories retain their original relative filenames.

```sh
VOICECHAT_REFERENCE=/private/tmp/eredu-voicechat-reference \
VOICECHAT_OFFICIAL_ASSETS=/private/tmp/eredu-voicechat-reference/assets \
VOICECHAT_BASE_ASSETS=/private/tmp/eredu-voicechat-reference/base-assets \
cargo test -p eredu-architectures --lib voicechat::assets::tests -- --include-ignored
```

The fixture test checks the actual released integer and tokenizer values and
rejects corrupted integer values and reordered merges. The default tests cover
exact-length asset corruption, resolver error preservation, control-code order,
retained base-EOS flags, and bounded exact-range reads.
