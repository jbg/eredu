# Public example migration inventory

The initial simple chat examples called the superseded allocating `prepare_chat`
shape and token/generation conveniences. Their replacement compiles the
loaded tokenizer and selected template through `RetainedConfiguration`, prepares
a borrowed request under an explicit capacity, then uses `PreparedChatRequest`
with the same manual or uninterrupted cursor. Literal output remains an explicit
policy; token IDs, special spellings, stopping and numerical probes must survive.

This slice owns `gemma4_generate` and `chat_probe`. The former includes a one-token
greedy inspection followed by a fresh seeded literal run. The latter compares
literal output, semantic run and manual semantic advancement; it does not use
snapshots, forcing, intervention or capture payloads. Both can use one original
source and the canonical prepared session without reducing their purpose.

The initial inventory assigned these other consumers to separate slices:

- `component_reference_probe`, its `component_reference_association` module, and
  `component_sparse_probe` use exact external prefix IDs, architecture-selected
  capture/intervention sources, overlays, and numerical reference comparisons.
  Replacing their prefixes with a rendered chat would invalidate those checks.
- `controlled_generate` demonstrates snapshot/restore/branching and interventions;
  `observed_generate` and `intervened_generate` preserve capture and bounded record
  delivery. Their migration required the complete observation/control source
  composition, completed by the later ordinary-frontdoor work linked below.
- `gemma4_speculative_generate` exercises a drafter and ordinary/speculative parity.
- `qwen35_moe_bench` measures raw token prefill/decode and intentionally controls
  iteration independently of EOS. `managed_plain_memory_probe` compares the raw
  ordinary baseline with already source-funded managed sessions. Neither should
  silently gain chat policy or different termination while migrating.
- The iOS C ABI example has streaming callbacks, plain fallback, cancellation and
  statistics crossing an unsafe application boundary. Its migration is separate
  from these Rust command-line examples.
- Architecture discovery and component analysis modules are inspection-only;
  they do not own a generation entry point.

`prepared_chat_generate`, `native_tool_calling` and `gguf_generate` have separate
owners and were excluded from this slice. The later
[ordinary-frontdoor migration](bounded-followup-ordinary-frontdoors.md) completed
the component/capture/control callers, and the speculative callers now use
`PreparedChatSpeculativeRequest` and `generate_prepared_chat_speculative`. These
metadata results do not establish native numerical or distributed execution.

Implemented: `gemma4_generate` compiles and retains the complete loaded sources,
then uses literal prepared sessions for greedy inspection and the seeded run.
The run keeps checkpoint maximums up to the existing 120-token ceiling, EOS
termination and visible special spellings. `chat_probe` uses the same source,
request and cursor for literal output, semantic `run`, and manual `advance`,
retaining its report fields and caller-owned output collection.

Both commands expose `CAPACITY_BYTES` with a documented 1 GiB default; source and
generation settings use that same explicit capacity. Both examples pass `cargo check --offline -p eredu --no-default-features
--features mlx,metal --example gemma4_generate --example chat_probe` against the
coherent frozen source and configured native dependencies (52.61 seconds; no
`DOCS_RS` bypass). Only the two exact example files were copied into that tree.
This is compile coverage; native execution and released-checkpoint results are
not asserted by these source edits.
