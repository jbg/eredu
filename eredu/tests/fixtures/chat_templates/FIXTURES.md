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

`lfm2.5-1.2b-instruct-ba551d58.jinja` is the unmodified `chat_template.jinja`
from `LiquidAI/LFM2.5-1.2B-Instruct` revision
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`. Its suffix identifies the template's
SHA-256 digest. It covers native requests with Python keyword argument names
and rendering parsed arguments back into conversation history.
