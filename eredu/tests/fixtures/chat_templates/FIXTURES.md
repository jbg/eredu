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
