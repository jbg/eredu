# Released prepared-chat validation

The official `Qwen/Qwen3.5-0.8B` checkpoint is pinned to revision
`2fc06364715b967f1860aea9cf38778875588b17`. The
[checkpoint manifest](validation/qwen35-checkpoint.json) contains sizes and
digests for all 13 files. Checkpoints and expanded media arrays belong outside
the tracked source tree.

The [result record](validation/prepared-chat-released-results.json) identifies
the tested executable, exact requests, semantic events and process counters.
It covers fully resident ordinary generation on Apple M3 Ultra/Metal, with a
64 GiB framework capacity, greedy sampling, seed 0, prefill chunks of 128,
48-token maximum and thinking disabled. The official template is used directly.

Both Required and Auto emit exactly one `reading({"value":17})` call, with
`call_0`, index 0, 25 committed tokens and `GrammarComplete`. Neither emits
visible prose or reasoning. The value is supplied in the prompt. This validates
the stated request, not general tool usefulness for arbitrary prompts.

`validation/prepared_chat_tools.py` checks fragmented arguments, unique JSON
keys, call indices, terminal events and agreement with the public result. It
rejects truncation or cancellation even when a complete call precedes them.
Validator unit tests do not constitute model evidence.

## Authenticated image and tools

A uniform 256×256 RGB image uses the pinned processor's 16-pixel patches,
temporal width two and merge size two. Its 256 rows of 1,536 values expand to
64 decoder positions. The 385-position request crosses an uneven final prefill
boundary at chunk size 128. Both tool policies produce the same complete call
and termination as the text request.

The image generator uses the pinned local processor. `prepare_chat_input`
funds and authenticates retained framework copies. Caller processing buffers
remain outside that account. This exercises the real encoder and authenticated
tool path; the text-supplied value makes it unsuitable as image-recognition or
independent image-logit evidence.

| Input / policy | Wall seconds | Peak RSS bytes | Peak process footprint bytes |
| --- | ---: | ---: | ---: |
| Text / Required | 31.25 | 3,127,853,056 | 5,613,948,216 |
| Text / Auto | 24.35 | 3,128,885,248 | 5,617,798,480 |
| Image / Required | 30.19 | 3,186,294,784 | 5,679,861,048 |
| Image / Auto | 28.80 | 3,186,229,248 | 5,679,369,480 |

These serial debug runs include loading and planning. They are functional
measurements, not throughput benchmarks. Whole-process counters include allocator
caches and other memory outside framework admission. A capacity is a ceiling,
not observed usage. These requests do not validate distributed banks, paging,
CLI reset, default-stack behavior or downstream application memory policy.

## Reproduction

Use Rust 1.98.0 and the native environment in
[bounded validation](bounded-inference-validation.md). Set `QWEN_CHECKPOINT` to
the pinned local checkpoint directory and `VALIDATION_OUTPUT` to a writable
directory outside the source tree. The debug worker uses a 64 MiB stack.

```sh
export RUST_MIN_STACK=67108864
CARGO_INCREMENTAL=0 cargo +1.98.0 build --offline -p eredu \
  --no-default-features --features mlx,metal,image --example prepared_chat_generate
strip -o "$VALIDATION_OUTPUT/prepared-chat" target/debug/examples/prepared_chat_generate
python3 validation/prepared_chat_tools.py \
  --binary "$VALIDATION_OUTPUT/prepared-chat" \
  --checkpoint "$QWEN_CHECKPOINT" \
  --pinned-record doc/validation/qwen35-checkpoint.json \
  --source-root . --output "$VALIDATION_OUTPUT/text" \
  --capacity 68719476736 --chunk 128 --max-tokens 48
```

For image input, use a Python environment with Transformers and tokenizers:

```sh
python3 validation/prepared_chat_image_request.py \
  "$QWEN_CHECKPOINT" "$VALIDATION_OUTPUT/image-input"
```

Supply `--request-template "$VALIDATION_OUTPUT/image-input/image.request.json"`
to the validator and select a separate output directory. The input generator's
metadata records its package versions and processor settings.

The separate [numerical record](validation/qwen35-numerical-results.json) compares
ordinary resident prefill and three teacher-forced cached decodes against MLX-LM.
All eight full 248,320-value rows are finite, match reference argmax and satisfy
`abs(actual-reference) <= 0.25 + 0.02*abs(reference)`. Its executable, reference
versions and score hashes are distinct from the prepared-chat result record.
