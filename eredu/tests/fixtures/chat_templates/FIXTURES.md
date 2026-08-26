# Chat-template fixtures

Every Jinja file in this directory is referenced by a rendering or behavioral
protocol test. Whitespace is significant, and filenames identify source
snapshots where applicable.

Runtime capability is determined from rendered behavior and tokenizer special
tokens, not filenames or body hashes. Tests cover message framing, reasoning,
tool-definition rendering, call parsing, and stop behavior. Ambiguous or
unsupported templates remain fail-closed.

Fixture changes must include rendering tests for the claimed behavior.
