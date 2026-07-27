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
| `mistral-7b-instruct-v0.3-c170c708.jinja` | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | single `chat_template` |
| `ministral-8b-instruct-2410-2f494a19.jinja` | `mistralai/Ministral-8B-Instruct-2410` | `2f494a194c5b980dfb9772cb92d26cbb671fce5a` | single `chat_template` |
| `llama-3.1-3.3-e10ca381.jinja` | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | single `chat_template` |
| `llama-3.2-5816fce1.jinja` | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | single `chat_template` |
| `llama-4-01a91bfb.jinja` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `92f3b1597a195b523d8d9e5700e57e4fbb8f20d3` | single `chat_template` |
| `llama-3.1-nemotron-nano-8b-v1-072b9ab4.jinja` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | `54641c1611fcff44fa4865626462445e0a153fc7` | single `chat_template` |
| `gemma-4-e2b-it-3e22461f.jinja` | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `chat_template.jinja` |
| `gemma-4-26b-a4b-it-4d7ae498.jinja` | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `chat_template.jinja` |

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

The Llama 3.1 fixture is byte-identical to
`meta-llama/Llama-3.3-70B-Instruct` revision
`6f6073b423013f6a7d4d9f39144961bfbfbc386b`. The Llama 4 fixture is
byte-identical to `meta-llama/Llama-4-Maverick-17B-128E-Instruct` revision
`73d14711bcc77c16df3470856949c3764056b617`. Those shared bodies intentionally
share exact registrations; no family or repository metadata is inspected.
The upstream Nemotron body does not end in a line feed, so its fixture carries
one repository file terminator that signature tests remove explicitly.

The Gemma 4 E2B fixture is byte-identical to `google/gemma-4-E4B-it`
revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. The 26B-A4B fixture is
byte-identical to `google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` and
`google/gemma-4-31B-it` revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. These two signatures capture
the audited difference in disabled-thinking generation prompts without
granting support from `model_type`.
