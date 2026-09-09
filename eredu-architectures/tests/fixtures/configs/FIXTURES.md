# Configuration fixtures

These immutable configuration bodies are copied from pinned official
checkpoint revisions. The Qwen3.8 fixture verifies the `qwen3_5` /
`qwen3_5_text` architecture contract together with vision metadata and one
embedded MTP layer.

`nanbeige4.2-3b-0e137298.json` is the unmodified config from
[Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/0e137298720f7241e83b8aabecc4263dcc7d84b3/config.json).
It declares 22 physical blocks, 2 loops, 48 query heads of width 128,
8 KV heads, and a 3072-wide residual stream. The independent tiny scalar
oracle under `../nanbeige/` documents its source and regeneration command.
