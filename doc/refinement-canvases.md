# Seeded and pinned refinement canvases

A `RefinementCanvas` contains exactly the selected model's canvas extent:
`seed_tokens` supplies optional initial model token IDs, and `pinned_positions`
identifies unique seeded positions. Unseeded positions receive full-vocabulary
noise. Seeds without pins affect initialization only. Pins retain their token
identity throughout every refinement iteration and every subsequent block.

The portable refinement driver supplies one-hot feedback for pinned positions
and excludes them from confidence averaging. Random draws still cover the entire
canvas, so applying pins does not shift the stream's draw coordinates. Raw model
score observation precedes this processing. A finalization constraint that
excludes a pin leaves no admissible candidate and receives a typed rejection.

The facade exposes `LoadedModel::refinement_canvas_length()` and
`PreparedObservedGeneration::with_refinement_canvas`. Ordinary observed generation
and controlled sessions consume the same configured request and operation driver.
The lower-level `TextGenerationConfig::with_refinement_canvas` retains an immutable
shared canvas. Geometry and token IDs are admitted before sampling or context
execution; facade requests additionally require tokenizer-valid seed IDs.

For example, a caller can seed a template, leave one position unseeded, and pin
all remaining positions:

```rust,no_run
use eredu_core::refinement::RefinementCanvas;
let canvas = RefinementCanvas {
    seed_tokens: vec![Some(10), None, Some(11)],
    pinned_positions: vec![0, 2],
};
// The vector length must equal model.refinement_canvas_length().
let prepared = prepared.with_refinement_canvas(canvas);
```

`CaptureTransform::TokenScores { position: Some(p), token_ids }` reads the selected
raw scores at zero-based sequence position `p`, with full-vocabulary normalization.
`position: None` selects the final row. Each selected position must exist in every
captured invocation. These are position distributions for refinement, not causal
sequence likelihoods. Captures remain subject to native, host and encoded byte
budgets; selected-row reductions avoid copying the complete canvas to the host.

A caller uses ordinary controlled steps to encode context, perform refinement,
collect score captures, and cancel before finalization when no token commitment
is wanted. This does not construct a second inference engine. Snapshots, restore,
fork, reseeding, draft reconciliation and monotone resource budgets preserve the
canvas and its pins. Workspace envelopes include pinning temporaries, and
snapshot estimates include the seed layout, fixed-token tensor and mask.

The CLI owns typed question compilation and result interpretation, described in
[the CLI guide](../eredu-cli/README.md#typed-decision-reads). The library owns only
canvas control, inference and score observation.

Native CPU conformance covers seeded/pinned snapshot replay and fork, ordinary
versus controlled commitment, selected-position score reductions, incompatible
finalization constraints, and CLI reads across resident, host-layerwise and
disk-streamed weights. The neutral policy fixture also checks fixed feedback and
confidence averaging independently. Native accelerator and distributed pinning
runs have no validation result recorded here.

The focused conformance commands are:

```sh
cargo test -p eredu-runtime --lib generation::refinement::tests
cargo test -p eredu-backend-mlx --no-default-features --lib loaded_block_snapshots_replay_and_fork_context_refinement_and_partial_finalization
cargo test -p eredu-backend-mlx --no-default-features --lib selected_position_scores_read_requested_row_and_reject_out_of_range
cargo test -p eredu-backend-mlx --no-default-features --lib constrained_finalization_uses_each_final_position_domain
cargo test -p eredu-cli --no-default-features --features mlx --test decisions
```

## Preparation profiling

Decision-read timing separates canvas tokenization, chat preparation, capture
discovery and admission. The first capture discovery hashes the complete physical
checkpoint graph for content-exact provenance. Subsequent discovery calls share
the resolved identity. Large multi-member artifacts use up to four hashing
workers while preserving full-content identity and per-file metadata change
checks.

The pinned official tokenizer sidecars at revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b` reproduce the 97-token duplicate-charge
request without model weights. Their SHA-256 values are asserted by the opt-in
portable profile:

```sh
EREDU_DIFFUSION_TOKENIZER_DIR=/path/to/sidecars cargo test -p eredu --release --no-default-features --lib released_diffusion_gemma_chat_preparation -- --ignored --nocapture
cargo run --release -p eredu-core --example artifact_fingerprint -- /path/to/checkpoint 3
```

A release profile measures tokenizer loading at 0.293 seconds, grammar compiler
construction at 0.560 seconds, chat preparation at 0.00236 seconds and prompt
encoding at 0.000142 seconds. An eleven-member fixture of 128 MiB per member
(1.375 GiB total) measures median fresh fingerprint resolution at 0.187 seconds
with parallel hashing, versus 0.616 seconds with one worker. The fixture uses
warm filesystem cache; full-checkpoint and GPU end-to-end latency are not measured
by these numbers. All runs produce identity
`sha256:c9b49771af6b3d2163cb64c9f33b2765c11d2d576a9a6473cd7fa34652072e12`.

The hashing fixture is reproducible with ordinary binary files. For each member
index 0..10, write `bytes((byte + index) % 256 for byte in range(256))` repeated
524288 times to `member-00` through `member-10` in the benchmark directory.
