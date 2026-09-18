# Released public prepared-chat tool validation

The recorded CPU routing-source executable passes both Required and Auto for the
concrete sensor request under the unchanged 64 GiB framework limit. Each run emits exactly one
`reading({"value":17})` call, commits 25 tokens and finishes with
`GrammarComplete`; neither emits visible prose or reasoning. The latest functional
rerun takes 31.25 and 24.35 seconds, respectively. The four text/image runs were
serial functional debug runs; these are not throughput benchmarks.
The [CPU routing-source record](validation/bounded-followup-released-tools-cpu-routing-source-2026-09-18.json)
contains the exact requests, events, provenance and executable digest. The
[previous retirement record](validation/bounded-followup-released-tools-retirement-reset-2026-09-18.json)
preserves the preceding 33.41/28.35-second runs. The earlier
[scoped-parser record](validation/bounded-followup-released-tools-scoped-2026-09-18.json)
retains the preceding 29.33/28.38-second runs. This
establishes ordinary released tool generation for that request. The separate
generic-action Required request now completes all 48 allowed tokens without a
funding failure after the parser-frame correction described below. It emits
prose and reaches `MaxTokens`, so it remains an unsuccessful tool request.

The cached official `Qwen/Qwen3.5-0.8B` checkpoint is pinned to revision
`2fc06364715b967f1860aea9cf38778875588b17`. All 13 cached files were freshly
verified against `doc/validation/bounded-followup-native-2026-09-18.json`; the
new record also includes a SHA-256 digest for every file. In particular:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| model.safetensors-00001-of-00001.safetensors | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| tokenizer.json | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| chat_template.jinja | 7,755 | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |

This task reuses the pinned artifacts and previous independent numerical
comparison. It does not repeat those comparisons or treat synthetic fixtures as
released-checkpoint evidence.

The unchanged `prepared_chat_generate` example loads a fully resident execution
plan with drafting disabled, compiles retained tokenizer/template sources, and
runs `PreparedChatRequest` through `start_prepared_chat(...).run(...)`. The request
sets an enforced 64 GiB framework capacity, greedy sampling, seed 0, prefill chunks
of 128 positions, and thinking disabled through the checkpoint's supported
setting. Its tool declaration has one integer argument constrained to enum
`[17]`. The official template is used unchanged.

`validation/prepared_chat_tools.py` invokes the same executable for Required and
Auto policy in separate processes. It requires exactly one complete semantic
`reading` call with `{"value":17}`, consistent indices, valid JSON without
duplicate keys, and successful EOS, grammar-complete or stop termination. It
rejects truncation and cancellation even if a complete call preceded them. The
public result must agree with the terminal semantic event. The record preserves
all events, exact commands, request and executable hashes, failures and elapsed
time. Two focused validator tests cover fragmented valid arguments and malformed
or truncated event streams; those tests are not model evidence.

The recorded native binaries were built from coherent source checkpoints with
Rust 1.98 and actual MLX/Metal support. The host is an Apple M3 Ultra with 256 GiB
RAM and 32 logical CPUs. The run sets `RUST_MIN_STACK=67108864` for the debug
worker's planner frames; default-stack operation is not established by this run.
The current stripped executable is 858,661,048 bytes with SHA-256
`7b4ae36fafeaaf35f3307abbffb2641697d5e3927f567bd964896687306a8bac`.

Rebuild the example with the recorded Rust 1.98 toolchain and MLX/Metal SDK
environment, then pass that executable to the validator. A full-stripped copy
avoids the recorded host's large debug-executable mapping limit:

```sh
RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
CLANG_MODULE_CACHE_PATH=/private/tmp/eredu-followup-clang-cache \
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target \
  cargo +1.98.0 build --offline -p eredu --no-default-features \
  --features mlx,metal,image --example prepared_chat_generate
strip -o /private/tmp/eredu-current-prepared-chat-generate-cpu-routing-source \
  target/debug/examples/prepared_chat_generate

RUST_MIN_STACK=67108864 \
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
CLANG_MODULE_CACHE_PATH=/private/tmp/eredu-followup-clang-cache \
python3 validation/prepared_chat_tools.py \
  --binary /private/tmp/eredu-current-prepared-chat-generate-cpu-routing-source \
  --checkpoint /private/tmp/eredu-bounded-qwen35-2fc06364715b967f1860aea9cf38778875588b17 \
  --pinned-record doc/validation/bounded-followup-native-2026-09-18.json \
  --source-root /Users/jbg/dev/eredu \
  --output /private/tmp/eredu-released-tools-reproduced \
  --capacity 68719476736 --chunk 128 --max-tokens 48
```

## Authenticated image and tools

The same example also passes Required and Auto with real processed image input.
A uniform 256×256 RGB image uses the pinned processor's 16-pixel patches,
temporal width two and merge size two: 256 rows of 1,536 values expand to 64
decoder positions. The authenticated request has 385 positions, so chunk size
128 crosses an uneven final prefill boundary before cached generation. Both
policies commit 25 tokens and the same complete `reading({"value":17})` call,
ending with `GrammarComplete` under 64 GiB. The current executable takes 30.19
and 28.80 seconds, including loading and planning.

The [image validation record](validation/bounded-followup-released-image-tools-cpu-routing-source-2026-09-18.json)
retains the pinned artifacts, exact request hashes, generator/version metadata,
commands, semantic events and process counters. Expanded arrays stay outside the
tracked tree. The [previous image record](validation/bounded-followup-released-image-tools-retirement-reset-2026-09-18.json)
preserves the preceding 34.93/32.81-second runs. This checks actual image encoding
and source-authenticated tool generation; the sensor value is supplied in text,
so it does not establish visual
recognition or independent image-logit accuracy.

| Input / policy | Wall seconds | Peak RSS bytes | Peak process footprint bytes |
| --- | ---: | ---: | ---: |
| Text / Required | 31.25 | 3,127,853,056 | 5,613,948,216 |
| Text / Auto | 24.35 | 3,128,885,248 | 5,617,798,480 |
| Image / Required | 30.19 | 3,186,294,784 | 5,679,861,048 |
| Image / Auto | 28.80 | 3,186,229,248 | 5,679,369,480 |

These are whole-process counters, not framework-managed allocation totals.
All four runs produce identical semantic event output. They include the
canonical metadata/observation census, corrected CPU broadcast span validation,
expired-view cleanup and removal of inactive parallel-control slots with their
final prospective control census. The recorded CPU routing-source artifact also
includes canonical typed-zero/Transpose traces and CPU view, row-movement and
selector qualification.
These dense resident requests do not validate addressable-bank partition geometry,
Host/Disk paging or CLI reset/snapshot paths; those have separate validation.

Generate the input in a Python environment with Transformers and tokenizers,
using only the pinned local checkpoint, then use the same semantic validator:

```sh
python validation/prepared_chat_image_request.py \
  /private/tmp/eredu-bounded-qwen35-2fc06364715b967f1860aea9cf38778875588b17 \
  /private/tmp/eredu-image-tool-request
```

Add `--request-template /private/tmp/eredu-image-tool-request/image.request.json`
to the validator command above and select a fresh `--output` directory. It retains
the processed payload while selecting Required/Auto and the explicit token/chunk
limits. Caller-owned processing buffers remain outside framework accounting;
`prepare_chat_input` funds and authenticates every retained framework copy.

## Accounting changes and evidence scope

The shared native collector now prices one actual Work population while retaining
cumulative validation aliases, capture copies, publication attempts and saved
roots. Paged state and foreground parameter sources keep their separate lifetime
bounds. Parser operations reuse a paid temporary frame peak while charging
recursive overlap and persistent growth. Cold source inspection, schema construction/copy
and JSON/Lark emission now use the same prospective scoped-frame mechanism.
Both released text and image reruns include these corrections. These changes modify the real producers
and their quotes together; they do not enlarge the capacity or refund spent work.

Earlier timeouts, admission refusals, component diagnostics and parser failures
remain in the [initial record](validation/bounded-followup-released-tools-2026-09-18.json),
[intermediate totals](validation/bounded-followup-released-tools-current-2026-09-18.json),
[diagnostic record](validation/bounded-followup-released-tools-diagnostic-2026-09-18.json),
[Work producer record](validation/bounded-followup-released-tools-work-2026-09-18.json)
and [scoped parser record](validation/bounded-followup-released-tools-scoped-2026-09-18.json).
Their binaries, commands, exact requests and failures identify historical source
checkpoints. A failed candidate's quota is not a measurement of the successful
request's allocation usage.

Caller-owned processing, request JSON and validation output are outside the
framework allocation domain. The configured capacity is a limit, not observed
resident memory. Whole-process counters include loading, I/O and allocator caches;
they do not measure managed allocation usage. These released tests establish the
stated tool and image semantics, not general tool usefulness, independent image
logit accuracy, distributed execution or default-stack operation. Distributed
and synthetic ownership tests have separate records in
[the validation overview](bounded-followup.md).
