# Chat-template fixtures

These Jinja files test rendering and behavioral protocol recognition across the
supported chat-template families. Whitespace is significant, and filenames
identify source snapshots where applicable.

Runtime capability is determined from rendered behavior and tokenizer special
tokens, not filenames or body hashes. Tests cover message framing, reasoning,
tool-definition rendering, call parsing, and stop behavior. Ambiguous or
unsupported templates remain fail-closed.

Fixture changes must include rendering tests for the claimed behavior.
