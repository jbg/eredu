# Audited production chat templates

These files are byte-for-byte fixtures of the selected chat-template body
from the named source. Most are selected `chat_template` bodies from Hugging
Face tokenizer configurations; the DeepSeek tool templates are the pinned
deployment templates named in their rows. SHA-256 values are audit provenance
only. Runtime support is selected from tokenizer facts and rendered protocol
behavior; repository or model metadata is not a support key.

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
| `nemotron-nano-v2-6533e8de.jinja` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | single `chat_template` |
| `gemma-4-e2b-it-3e22461f.jinja` | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `chat_template.jinja` |
| `gemma-4-26b-a4b-it-4d7ae498.jinja` | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `chat_template.jinja` |
| `unsloth-gemma-4-26b-a4b-it-94899c0f.jinja` | `unsloth/gemma-4-26B-A4B-it-GGUF` | `3bb10d594514ef4edb7f3a65d41a7e4eb8c5767a` | GGUF `tokenizer.chat_template` metadata |
| `gpt-oss-harmony-a4c9919c.jinja` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `chat_template.jinja` |
| `gpt-oss-harmony-b474759b.jinja` | `openai/gpt-oss-20b` | `10e9d713f8e4a9281c59c40be6c58537480635ea` | `chat_template.jinja` |
| `gpt-oss-harmony-f8d92557.jinja` | `openai/gpt-oss-20b` | `ec854da5735f125fe36f080d8013482590f9ad7d` | `chat_template.jinja` |
| `kimi-linear-48b-a3b-instruct.jinja` | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | `5eaec587113df30a2cd5821b3fa4be7ee8e8aa37` | `chat_template.jinja` |
| `lfm2-classic-b3afba27.jinja` | `LiquidAI/LFM2-350M` | `b3afba27815ee83a64b76162cef4d8a4780d6ca7` | `chat_template.jinja` |
| `lfm2-classic-compact-6d24c6b7.jinja` | `LiquidAI/LFM2-1.2B-Tool` | `6d24c6b7471fcbcec084935e377a5302f4b84389` | `chat_template.jinja` |
| `lfm2.5-8b-a1b-5673e0de.jinja` | `LiquidAI/LFM2.5-8B-A1B` | `5673e0de372b64331504de73bbbc33b0dde71903` | `chat_template.jinja` |
| `lfm2.5-vl-450m-fc6221ca.jinja` | `LiquidAI/LFM2.5-VL-450M` | `fc6221ca597f3315e4f82fc2df606783267b34ba` | `chat_template.jinja` |
| `deepseek-v3-tools-7e28c67d.jinja` | `sgl-project/sglang` | `7e28c67d19ddea8c74fca7b7e6dd2e3e3bec3c37` | `examples/chat_template/tool_chat_template_deepseekv3.jinja` |
| `deepseek-v3.1-tools-ef1ab230.jinja` | `sgl-project/sglang` | `ef1ab2302ab25db09d3bd61da9bded1b71d0d3c8` | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |

The Hermes fixture body is also byte-identical to the named `tool_use`
template in `NousResearch/Hermes-3-Llama-3.1-8B` revision
`896ea440e5a9e6070e3d8a2774daf2b481ab425b`. Both bodies satisfy the same
generic XML-tool protocol probes without inspecting either model's
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

The upstream Kimi Linear template also has no final line feed. Its fixture
carries one repository file terminator, which the rendering test removes
before use.

The Llama 3.1 fixture is byte-identical to
`meta-llama/Llama-3.3-70B-Instruct` revision
`6f6073b423013f6a7d4d9f39144961bfbfbc386b`. The Llama 4 fixture is
byte-identical to `meta-llama/Llama-4-Maverick-17B-128E-Instruct` revision
`73d14711bcc77c16df3470856949c3764056b617`. Those shared bodies intentionally
share protocol behavior; no family or repository metadata is inspected.
The upstream Nemotron body does not end in a line feed, so its fixture carries
one repository file terminator that signature tests remove explicitly.

The Nemotron Nano v2 body is byte-identical in
`nvidia/NVIDIA-Nemotron-Nano-12B-v2` revision
`f428df0ec725fed457b89cfca54dc26500fb88c1`. Both Nemotron-H releases share
one recognized protocol. Their JSON-list call envelope reuses the declarative
Nemotron implementation also used by the Llama-based Nano v1 protocol,
with a separate parameter set for the `<SPECIAL_12>` stop and the reasoning
channel whose `<think>` prefix is already present in the generation prompt.
The upstream body does not end in a line feed, so its fixture carries one
repository file terminator that signature and rendering tests remove
explicitly.

The Gemma 4 E2B fixture is byte-identical to `google/gemma-4-E4B-it`
revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. The 26B-A4B fixture is
byte-identical to `google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` and
`google/gemma-4-31B-it` revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. These two signatures capture
the audited difference in disabled-thinking generation prompts. The Unsloth
fixture carries one repository file terminator; after removing it, the exact
GGUF metadata string has SHA-256
`94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`.
It renders saved reasoning only when the assistant message also contains a
tool call and accepts both mapping and serialized-string tool arguments.
All three fixtures are behavioral recognition tests: their hashes and
`model_type` values do not grant runtime support.

Each GPT-OSS fixture is byte-identical to the corresponding official
`openai/gpt-oss-120b` body: revisions
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`,
`988e56b6fc3c723ad30aa7f18db742f54f921567`, and
`f39accdb091b6c5710dbbf02674c27064e35019b`, respectively. The three
signatures retain the official initial, escaped-newline, and current template
bodies. They share one Harmony dialect because their output channel and
function-call syntax is unchanged; neither GPT-OSS architecture metadata nor
the repository name participates in selection.
The upstream GPT-OSS bodies do not end in a line feed, so these fixtures carry
one repository file terminator that signature and rendering tests remove
explicitly.

The classic LFM2 body is byte-identical in the current official 350M, 700M,
and 1.2B releases. The compact body is byte-identical in the current official
1.2B-Tool, 2.6B, and 8B-A1B releases. Those classic templates render tool
definitions and opaque message content, but do not render structured
`tool_calls` into a verifiable output envelope. They therefore remain
provenance fixtures and are not recognized for native tool generation.

The LFM2.5 templates establish a Python-call list between
`<|tool_call_start|>` and `<|tool_call_end|>`. Their render macros emit calls
such as `name(argument='value')`; non-string values use Python spelling, while
nested mappings are rendered through `tojson`. This is intentionally a custom
dialect: the declarative dialects only admit JSON objects, JSON lists, or their
fixed structural-object encoding and cannot safely describe Python
identifiers, keyword arguments, single-quoted strings, `True`/`False`/`None`,
or the released mixed Python/JSON nested value surface.

The DeepSeek V3 fixture is the exact tool template recommended by SGLang's
DeepSeek V3 deployment documentation. The V3.1 fixture is the maintained
successor for the revised direct `name<｜tool▁sep｜>{arguments}` call surface.
Both bodies end in a line feed and are stored unchanged. Recognition binds
them to the generic declarative named-JSON-arguments shape: V3 adds the literal
`function` kind and a fenced JSON object, while V3.1 directly joins adjacent
call envelopes. Tool names are restricted to the DeepSeek API's documented
64-character ASCII letter, digit, underscore, and dash surface, and argument
objects are constrained by each declared JSON Schema.

V3.1's `thinking` template variable is wired to the caller's standard
`enable_thinking` control. Thinking is rejected while tools are supplied for
both recognized profiles: these selected call surfaces permit free-form text
before a call but do not delimit it sufficiently to preserve reasoning as
reasoning events. Disabled thinking retains each template's exact generation
prompt, including V3.1's prefilled `</think>`.

DeepSeek's released tokenizer templates for V3, R1, and V3.1 preserve prior
tool calls and results but do not render supplied tool definitions, so they
are not recognized native-tool protocols. The maintained SGLang R1 tool
template is also intentionally excluded: its tool prompt indexes bare function
definitions (`tool['name']`) instead of the OpenAI tool envelopes accepted by
the renderer, and its free-form reasoning stream cannot be reported as
reasoning events by this declarative profile. DeepSeek V3.2 is likewise not
mapped to SGLang's older V3.2 Jinja template. The authoritative V3.2 release
replaced that surface with the Python `encoding/encoding_dsv32.py` DSML
encoder/parser, has no selected Jinja template body to sign, and uses
parameter-level DSML rather than this JSON protocol. Recognizing either
candidate would overstate safe support; neither architecture nor repository
metadata is used as a fallback.
