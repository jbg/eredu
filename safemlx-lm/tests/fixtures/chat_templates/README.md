# Chat-template fixtures

These Jinja files are fixed copies of selected production chat-template bodies.
They test rendering and behavioral protocol recognition for Qwen (including
the pinned Qwen3.6 and Qwen3.8 release templates), Mistral,
Llama, Nemotron, Gemma, GPT-OSS, Kimi Linear, LFM2, DeepSeek, Inkling, and
Muse-Glimmer templates.

The source snapshot is encoded in each filename where applicable. Tests verify
the expected body signature and account for the repository file terminator when
the source body has no final line feed. Other whitespace is significant.

Runtime support is never granted by a filename, source repository, model type,
or body hash. The recognizer checks rendered conversations, tokenizer special
tokens, reasoning and visible-text framing, tool-definition rendering, call
parsing, and stop behavior. Two fixtures may therefore share one runtime
profile when their observable protocol is equivalent.

The fixture set covers these wire-format shapes:

- XML-wrapped and structural-token JSON calls;
- JSON call lists and named JSON arguments;
- Python-channel and Python-expression call lists;
- Gemma reasoning/tool channels;
- OpenAI Harmony channels;
- separate reasoning and visible-text message frames; and
- templates that render chat but deliberately lack a safe native-tool surface.

Fixtures preserve ambiguous or unsupported cases so capability detection stays
fail-closed. Adding or changing a fixture must include rendering tests for the
claimed behavior; a template body alone is not evidence of tool support.
