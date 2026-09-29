# Tokenizer and output vocabulary validation

## Vocabulary contract

The facade supplies one immutable, closed tokenizer-validity set to raw text,
templated fallback, semantic, observed, controlled and speculative requests. It
contains actual canonical IDs with consistent mappings in both directions,
including added/special tokens, rather than `0..get_vocab_size(true)`. Grammar
filters and forced choices intersect this set. Core projects it to actual logits
width; missing entries are false and an empty executable intersection is an error.
MLX realizes the mask before the existing sampling policy using negative infinity.
There are no model-specific sizes or invented tokens.

Sparse constraint tries keep original IDs with absent slots, and EOS membership
is checked against mappings. Snapshot forks share immutable validity without
charging a copied mask; history and pending choices retain their existing copy
rules. Vocabulary fingerprints include sparse high IDs. Speculative branch
filters apply validity before draft/target sampling and reject undefined history
IDs. Assistant compatibility requirements apply independently and reject
incompatible draft/target outputs.
A mapped forced ID outside a narrower model output fails when logits width becomes
available. Low-level core callers without a tokenizer may explicitly use `All`.

## Automated validation

```sh
cargo check -p eredu --no-default-features
cargo test -p eredu-core -p eredu-runtime -p eredu-text --lib
cargo test -p eredu --no-default-features --lib --test portable_facade --test backend_conformance
cargo test -p eredu-backend-mlx --features metal --lib backend::runtime::generation::backend::tests -- --include-ignored
cargo build -p eredu-cli
cargo fmt --all --check
git diff --check
```

Native tests require Metal access outside the filesystem/device sandbox.

Regressions exercise ordinary and semantic generation with highest logits on
padding and sparse holes; shorter output prefixes; mapped EOS and missing EOS;
grammar/validity intersection; empty grammar and executable intersections; forced
invalid IDs and valid IDs above token count; speculative forks and invalid draft
histories; snapshot validity retention; and sparse/inconsistent tokenizer mappings
and fingerprints. Existing sampling, capture, intervention, cancellation,
execution-control and snapshot conformance tests cover the shared drivers.

## Official checkpoint generation

The released fixture uses unquantized official weights and sidecars:

```sh
hf download LiquidAI/LFM2.5-1.2B-Instruct --revision 0f604ada3f766f9f257460c4c9f0b5d6f69d431b --local-dir /private/tmp/eredu-lfm2-vocabulary --include '*.json' --include '*.jinja' --include '*.safetensors'
```

The pinned revision is `0f604ada3f766f9f257460c4c9f0b5d6f69d431b`.
The checkpoint tokenizer has 64,402 unique IDs, 0 through 64401;
`config.json` declares 65,536 output positions. The weight SHA-256 in the artifact
metadata is `1ba63d9adb03ae43581db0e136e4416febe0441aff7296397bd455fb6017f73a`.
The recorded raw-text and tool results use Metal with fully resident checkpoint
weights and speculative drafting disabled.

Current semantic profile, without tools:

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-vocabulary --no-auto --device gpu:0 --temperature 0 --max-tokens 16 --verbose 'Reply with the single word Hello.' > /private/tmp/eredu-lfm2-semantic.out 2> /private/tmp/eredu-lfm2-semantic.log
```

The LFM2 semantic contract accepts ordinary reply text ending at `<|im_end|>` or
an admitted EOS token. It rejects tool-call framing when no tools are enabled.
The disabled-tool grammar placeholder is internal to forbidden-trigger setup and
is not a complete valid semantic reply. The portable
`lfm2_without_tools_generates_text_and_stops_at_message_end` fixture covers this
contract, including EOS aliases. This report contains no released-checkpoint
measurement of the no-tool semantic path.

Ordinary raw text:

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-vocabulary --no-auto --device gpu:0 --temperature 0 --max-tokens 16 --verbose --raw 'Hello, my name is' > /private/tmp/eredu-lfm2-raw.out 2> /private/tmp/eredu-lfm2-raw.log
```

Recorded result: 16 tokens with termination at the requested limit. Output:
` Alex, and I'm here to help you with your questions. How can I`.

Required semantic tool generation uses this `/private/tmp/eredu-lfm2-tools.json`:

```json
[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}]
```

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-vocabulary --no-auto --device gpu:0 --temperature 0 --max-tokens 64 --verbose --tools /private/tmp/eredu-lfm2-tools.json --tool-choice required 'What is the weather in Madrid? Use get_weather.' > /private/tmp/eredu-lfm2-tools.out 2> /private/tmp/eredu-lfm2-tools.log
```

Recorded result: `lfm2.python-tools.v1`, 13 generated tokens and termination at the
protocol stop sequence. Structured output:

```json
{"tool_call":{"index":0,"id":"call_0","name":"get_weather","arguments":{"city":"Madrid"}}}
```
