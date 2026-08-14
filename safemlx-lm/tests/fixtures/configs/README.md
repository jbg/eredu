# Checkpoint configuration fixtures

These immutable configuration bodies are copied from pinned official
checkpoint revisions. The Qwen3.6 and Qwen3.8 fixtures verify that both
releases retain the `qwen3_5` / `qwen3_5_text` architecture contract together
with vision metadata and one embedded MTP layer. Fixture files add one
repository line terminator; signature tests remove it before comparing the
pinned source body.
