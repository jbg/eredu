# Chat-template fixtures

Every Jinja file in this directory is referenced by a rendering or behavioral
protocol test. Whitespace is significant, and filenames identify source
snapshots where applicable.

`qwen3-0.6b-older-c945a4a8.jinja` is the older Qwen3 template used to verify
that behavioral recognition survives converter refactors. Its suffix is the
leading portion of the template's SHA-256 digest.

Runtime capability is determined from rendered behavior and tokenizer special
tokens, not filenames or body hashes. Tests cover message framing, reasoning,
tool-definition rendering, call parsing, and stop behavior. Ambiguous or
unsupported templates remain fail-closed.

Fixture changes must include rendering tests for the claimed behavior.

`qwen3.8-flash-next-de4b8e4d.jinja` and
the corresponding `-tokenizer-config.json` and `-generation-config.json` files are the unmodified
[`chat_template.jinja`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/chat_template.jinja)
and [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer_config.json),
and [`generation_config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json)
from `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. Unlike several older fixture names,
their suffix identifies the checkpoint revision. SHA-256:

- Template (8,952 bytes): `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- Tokenizer configuration (17,928 bytes): `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`.
- Generation configuration (202 bytes): `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.

API rendering tests cover default/enabled/disabled thinking, all three supported
effort settings, preserved historical and current tool-call reasoning, tagged
mapping arguments, grouped tool-result replay, image/video placeholder order,
the metadata EOS token and generation EOS aliases, and rejection of ambiguous tagged history. Small synthetic
tokenizers exercise behavioral profile selection independently of architecture names;
the full checkpoint tokenizer and model payload are not implied by these fixtures.

`lfm2.5-1.2b-instruct-ba551d58.jinja` is the unmodified `chat_template.jinja`
from `LiquidAI/LFM2.5-1.2B-Instruct` revision
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`. Its suffix identifies the template's
SHA-256 digest. It covers native requests with Python keyword argument names
and rendering parsed arguments back into conversation history.

`deepseek-r1-distill-qwen-1.5b-ad9f0ae0.jinja` is the exact `chat_template`
string extracted from
[`tokenizer_config.json`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/tokenizer_config.json)
at revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`. It has no final newline;
SHA-256 is `56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f`.

`qwq-32b-976055f8.jinja` is the exact `chat_template` string extracted from
[`tokenizer_config.json`](https://huggingface.co/Qwen/QwQ-32B/resolve/976055f8c83f394f35dbd3ab09a285a984907bd0/tokenizer_config.json)
at revision `976055f8c83f394f35dbd3ab09a285a984907bd0`.
SHA-256 is `16488ad69d85833e5f66c50174f39834b688343337e9337f018deaa21b354117`.
Its metadata declares `<think>` and `</think>` as ordinary added tokens
(`special=false`, IDs 151667 and 151668).

These fixtures establish reasoning admission from the generation suffix even
when reasoning history is discarded or native tools cannot be established.
Small tokenizers cover ordinary multi-token delimiters, ordinary added tokens,
and special added tokens. Neutral backend tests cover shared generation,
controlled and speculative delivery, snapshot restoration during a closing
marker, and retained inspection/loading. They do not claim numerical checkpoint
or native-device validation.
