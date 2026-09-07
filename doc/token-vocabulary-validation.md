# Tokenizer and output vocabulary validation

Validated on 2026-09-07 against the working tree based on Eredu
`9efa539fcec71fd47845c298825b032f0c5c2838`.

## Policy and diagnosis

Ordinary and semantic generation already used the same core text machine and
runtime sampling mechanisms. Ordinary generation supplied `TokenFilter::All`;
semantic generation supplied a tokenizer-sized grammar mask. MLX required the
latter to cover every output column. That both exposed the padded-width error in
semantic generation and left undefined output IDs eligible in ordinary generation.

The facade now supplies one immutable, closed tokenizer-validity set to raw text,
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
rules. Vocabulary fingerprints now include sparse high IDs. Speculative branch
filters apply validity before draft/target sampling and reject undefined history
IDs; existing assistant compatibility requirements still apply. This does not add
new speculative model-family support or permit incompatible draft/target outputs.
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

Results: 201 core, 308 runtime, 27 text, 154 facade unit, 22 backend conformance,
5 portable facade, and 4 native sampler tests passed (721 total). Five unrelated
fixture-dependent tests stayed ignored; the four selected native tests all ran.
Native tests require Metal access outside the filesystem/device sandbox. The
linker warned about the existing large debug unwind section; the build succeeded.

Regressions exercise ordinary and semantic generation with highest logits on
padding and sparse holes; shorter output prefixes; mapped EOS and missing EOS;
grammar/validity intersection; empty grammar and executable intersections; forced
invalid IDs and valid IDs above token count; speculative forks and invalid draft
histories; snapshot validity retention; and sparse/inconsistent tokenizer mappings
and fingerprints. Existing sampling, capture, intervention, cancellation,
execution-control and snapshot conformance tests passed.

Logs from this run are in `/private/tmp/eredu-vocabulary-{contract-tests,portable-final,mlx-tests}.log`.

## Official checkpoint generation

Downloaded the original, unquantized official weights and current sidecars:

```sh
hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir /private/tmp/eredu-lfm2-vocabulary --include '*.json' --include '*.jinja' --include '*.safetensors'
```

The resolved revision was `0f604ada3f766f9f257460c4c9f0b5d6f69d431b`.
The downloaded tokenizer has 64,402 unique IDs, 0 through 64401;
`config.json` declares 65,536 output positions. The downloaded weight SHA-256
recorded by hf is `1ba63d9adb03ae43581db0e136e4416febe0441aff7296397bd455fb6017f73a`.
All successful runs below exited 0 and used Metal with fully resident original
checkpoint weights and speculative drafting disabled.

Current semantic profile, without tools:

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-vocabulary --no-auto --device gpu:0 --temperature 0 --max-tokens 16 --verbose 'Reply with the single word Hello.' > /private/tmp/eredu-lfm2-semantic.out 2> /private/tmp/eredu-lfm2-semantic.log
```

Result: recognized `lfm2.python-tools.v1`; generated 12 tokens, ending with
`grammar_complete`, without a vocabulary-size error. Output was
`__eredu_lfm2_tools_disabled__`. This exposes a **separate pre-existing no-tool
semantic grammar defect**, not a successful conversational response:
`Lfm2Dialect::grammar(ToolChoice::None)` defines that literal, the default
`FormatDialect::semantic_constraint_configuration` delegates to that tool grammar,
and the no-tool semantic controller activates it immediately. This change preserves
the existing grammar and parser rather than replacing that protocol policy.

Ordinary raw text:

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-vocabulary --no-auto --device gpu:0 --temperature 0 --max-tokens 16 --verbose --raw 'Hello, my name is' > /private/tmp/eredu-lfm2-raw.out 2> /private/tmp/eredu-lfm2-raw.log
```

Result: 16 tokens, stopped at the requested limit. Output:
` Alex, and I'm here to help you with your questions. How can I`.

Required semantic tool generation used this `/private/tmp/eredu-lfm2-tools.json`:

```json
[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}]
```

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-vocabulary --no-auto --device gpu:0 --temperature 0 --max-tokens 64 --verbose --tools /private/tmp/eredu-lfm2-tools.json --tool-choice required 'What is the weather in Madrid? Use get_weather.' > /private/tmp/eredu-lfm2-tools.out 2> /private/tmp/eredu-lfm2-tools.log
```

Result: recognized `lfm2.python-tools.v1`; generated 13 tokens; stopped at the
protocol stop sequence. Structured output:

```json
{"tool_call":{"index":0,"id":"call_0","name":"get_weather","arguments":{"city":"Madrid"}}}
```

The original templated fallback was also exercised using cached official sidecars
from `3d2b14bdb14c120e685895d23320542f0c80c1f6`. Its `tokenizer.json` and
`config.json` were verified byte-identical to the current revision. The temporary
`/private/tmp/eredu-lfm2-legacy-template` directory contains copies of those older
sidecars, the current `special_tokens_map.json`, and a hard link to the same
original weight file. No repository routing logic or recognized current template
was changed for this run.

```sh
target/debug/eredu --model /private/tmp/eredu-lfm2-legacy-template --no-auto --device gpu:0 --temperature 0 --max-tokens 16 --verbose 'Reply with the single word Hello.' > /private/tmp/eredu-lfm2-fallback.out 2> /private/tmp/eredu-lfm2-fallback.log
```

Result: diagnostics explicitly selected `templated text fallback`; output `Hello`;
reported one generated content token and stop reason `eos`.
