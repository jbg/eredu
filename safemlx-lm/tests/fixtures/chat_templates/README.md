# Audited production chat templates

These files are byte-for-byte fixtures of the selected `chat_template` body
from the named Hugging Face tokenizer configuration. The registry matches the
SHA-256 of the selected body only; repository or model metadata is not a
support key.

| Fixture | Source repository | Pinned revision | Selected template |
| --- | --- | --- | --- |
| `qwen2.5-7b-instruct-acbd9653.jinja` | `Qwen/Qwen2.5-7B-Instruct` | `acbd96531cda22292a3ceaa67e984955d3965282` | single `chat_template` |
| `qwen3-0.6b-7e4ae267.jinja` | `Qwen/Qwen3-0.6B` | `7e4ae267688d671ddfca3122e4528ee980cf3234` | single `chat_template` |
| `qwen3-vl-2b-instruct-89644892.jinja` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | single `chat_template` |
| `hermes-2-pro-llama-3-8b-f798274b-tool-use.jinja` | `NousResearch/Hermes-2-Pro-Llama-3-8B` | `f798274b30e7d2d4797c369edcc0cc7473b6e6f2` | named `tool_use` template |

The Hermes fixture body is also byte-identical to the named `tool_use`
template in `NousResearch/Hermes-3-Llama-3.1-8B` revision
`896ea440e5a9e6070e3d8a2774daf2b481ab425b`. The shared signature therefore
selects one generic Hermes profile without inspecting either model's
architecture metadata.

The earlier Qwen3 revision is retained in
`safemlx-lm-utils/tests/fixtures/qwen3/tokenizer_config.json`. Its template
body is audited against `Qwen/Qwen3-0.6B` revision
`16706fc57485378d4ffaf54139b29ccc66ae08fa`.

The upstream Qwen3 template bodies do not end in a line feed. Their `.jinja`
fixtures carry one repository file terminator, which signature tests remove
explicitly. The other upstream bodies end in a line feed and are stored
unchanged. All other upstream whitespace, including trailing spaces, is
preserved byte-for-byte.
