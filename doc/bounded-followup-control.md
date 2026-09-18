# Recorded prepared sessions

The public recorded wrapper now owns a `PreparedChatSession` and a funded record
journal. It no longer constructs a separate continuation, cursor, decoder or
semantic pipeline. Startup consumes the same original prepared-chat request;
text versus semantic output, exact token input, media and observation options
remain policy on that request.

The shared original semantic-copy producer now accepts a borrowed
`PreparedTextHostJournal`. The concrete journal supplies physical preparation
bytes independently from its logical retention/copy contribution. The original
cursor/parser account includes its producer before copying, and the existing
snapshot transaction charges the combined logical amount once. Resume aliases
and retained journal errors carry the same destination authority and cumulative
copy lease. An empty journal uses the same worker for ordinary chat snapshots.

`SnapshotBudget::prepare` prospectively pays the actual shared owner, initialized
platform synchronization storage and retention shell under the supplied metadata
account. It does not confer native execution or physical-copy authority.

Validation on the qualified Rust 1.98 toolchain:

- Portable facade `cargo check -p eredu --no-default-features --offline` passes.
- The semantic/cursor/journal copy test verifies exact physical fit, one-byte
  refusal before journal work, immutable sources, independent parser branches,
  escaped journal custody, retained error custody, and nonrefundable cumulative
  copying across all three attempts.
- The budget test verifies constructor refusal and account lifetime through
  independently escaping budget aliases.

The complete wrapper, including original snapshot/restore/branch composition,
passes a portable library type check in a coherent source copy. Live public
recorded-session tests compare all four ordinary text/semantic/tool policies,
forced commitment, exact delivery timing, pause/resume, cancellation before a
prediction and retained failure prefixes. They pass on the default test stack.
The child journal test preserves a partial tool call and independently escaping
cancellation custody; all three original record producer/refusal tests pass. The portable public backend-conformance matrix passes all 126 cases. The neutral fixture now constructs capture banks from its original request, snapshots the actual funded collector, and resumes through a new original request. Child capture limits and intervention replacement use the same source-derivation workers as the native backend. Remaining failures and missing validation are implementation work, not capability restrictions.

Records are constructed prospectively from the actual metadata account. A single
prepaid terminal error owner retains the consumed cursor and borrows its committed
prefix; repeated steps do not copy history. Original capture/record refusal stays
available alongside that cursor failure. Callback unwinds fence the wrapper.
Snapshot metadata and restored semantic prefixes retain their shared original
copy destination; branch cancellation aliases retain it independently. Serial
exchange moves the complete canonical machine and its journal, controls, timing
and cumulative budgets together.

The synchronous host capture adapter uses an authentic `CaptureSummaryClaim`
from the original bank. It validates exact source shape and storage length, then
passes dense or uniform fixed values through the existing summary reducer and
strided selection. It cannot manufacture native backing or replay a claim. Three
focused runtime tests pass: exact quota and one-byte refusal, nonfinite/strided
results, once-only use, and escaping result/error custody. The public conformance
fixture keeps these real bank owners instead of wrapping an already constructed
frame in unrelated authority. Its original model has no neural tensor storage;
its immutable prompt IDs, scalar sampler, model identity and capture checkpoint
are still prospectively funded and copied.

An explicit empty intervention declaration remains a real source and remains in
branch lineage. `TextResumeFacts::has_interventions` reports whether the actual
installed source contains operations; capture-only record classification no
longer mistakes source presence for active edits. The read-only identity remains
unchanged. Rejected sampler updates preserve the prior state when the original
worker rejects before mutation.

A committed capture transaction with no selected frame can be snapshotted after
its terminal drain. Repeating that drain preserves checkpoint readiness; an
aborted transaction cannot acquire readiness. All 37 funded-session runtime
cases pass with this regression. Public fixture preparation now moves freshly
paid candidate reports into the shared planner instead of cloning report
strings, and resumed state geometry includes the exact cached frontier.

The public backend-neutral conformance binary passes all 126 cases, including
terminal restoration, partial Unicode, stop lookbehind, pending choices, branch
exchange, cumulative budgets, provider causes and final pool retirement. Resume
failure transport translates the actual backend cause once, preserves its
operation/classification, and retains the independent host copy through the same
prequoted error owner. This portable result does not substitute for native
hardware validation or close unrelated cold-producer gaps.

`output_checkpoint()` now returns a closed, read-only owner that retains its
original payer. Descriptive `GenerationOutputCheckpointData` values remain the
unchanged wire shape inside enclosed records and snapshots; they convey no
source authority. The same paid text producer constructs the live checkpoint.
Four record-producer tests pass, including every reached checkpoint refusal and
escaped payer retirement. The prepared-session parity test also passes all four
policy modes while retaining a checkpoint after the records, model and source
handles retire, then observes the real pool return to zero.

Reproducible portable validation uses the qualified Rust 1.98 toolchain:

```sh
cargo test -p eredu --no-default-features --test backend_conformance --offline -- --test-threads=4
cargo test -p eredu --no-default-features --lib api::control::records::tests --offline
cargo test -p eredu --no-default-features --lib recorded_session_preserves_ordinary_events_forcing_pause_timing_and_source_custody --offline
```
