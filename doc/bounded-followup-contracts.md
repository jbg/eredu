# Capture, control, and cancellation consolidation

Inventory before implementation, compared with `c513e17c583ad7cbc1caa1c9e8d9a4a2c5f62fa3`:

- Core introduced a fallible capture drain alongside the old raw drain. Its default converts a raw `CapturedStep` to `CapturedStepDelivery::Legacy`; the machine then parks shared frames when a raw consumer drains. Runtime and facade repeat this representation split. The canonical delivery will retain the immutable frame owner, preserve failure/pending custody, and drain fallibly after exact completion. Remove raw backend/drain compatibility and the legacy carrier; wire decoding is diagnostic ownership, never original funding.
- Facade control has one execution driver but two sealed record modes (`LegacyText` and `PreparedInputV2`), two snapshot/branch metadata families, and representation-specific startup/dispatch. The canonical records retain prompt attribution and custody. Remove mode parameters and obsolete record forms, migrating ordinary token input to real prompt attribution rather than stripping custody from prepared input.
- `PrefillExecutor` chains boolean cancellation through token cancellation to boundary-aware cancellation. The replicated executor preserves obsolete boundary-less entry, although admitted controls require exact boundaries. Keep one token-and-boundary contract; authenticate the same request/source and sample cancellation after retiring the prior span immediately before distributed consensus. Remove the boolean/token overloads and optional boundary fallback.

Meaningful distinctions remain: diagnostic deserialization versus live allocation authority; exact native completion versus host delivery; admitted capture funding versus explicitly unenforced capture policy; and before/after prefill boundaries.

Implemented contracts:

- The sole backend hook is `try_take_text_capture`; public machines and detached continuations drain through `take_captured_delivery` / `take_completed_delivery` after exact completion. Raw drain entry points and their parked-frame slot are removed. Pending and failed frames remain with the backend owner.
- The canonical `SharedCapturedStep` retains payload and custody directly; the delivery wrapper is removed. Ordinary collectors reserve and preallocate the same immutable frame shell before constructing records. Speculative abort attribution is finalized inside runtime before that owner is published. Observation events now have one `Token` and one `CaptureFailure` representation; cloning preserves custody.
- Logical speculative prefill reports likewise use only retained delivery. Ordinary aggregate construction now reserves a destination before window execution; original admitted aggregation preserves its existing destination and funding.
- Controlled sessions, snapshots and branches no longer have a record-mode type parameter. `ControlledGenerationRecord` retains actual prompt attribution for token and prepared inputs. Its typed event carries startup, snapshot and branch metadata or a borrowed-accessible progress event. Closed record and snapshot owners preserve custody; wire records are independent diagnostics and cannot restore or construct a source.
- Prefill cancellation has one boundary-aware token contract. Replicated execution authenticates the boundary and request, retires preceding resources and then samples cancellation for consensus.

Ordinary capture now includes the concrete retained frame shell in its logical host quota. This is real output ownership storage, reserved before records or native capture work; quotas are not bypassed or expanded silently. Diagnostic copies in tests are explicit caller-owned copies.

Graph inventory before consolidation:

- `ArchitectureExecutionGraph` carries borrowed, single-group and owned representations; the owned representation exists to adapt the former allocating graph producer. `LayeredArchitecture` and ingress graph discovery layer a metadata companion over that producer, and partition/resident consumers select a fallback based on the destination.
- Prediction chains and several composite architectures reconstruct the same graph on discovery. Retain their validated graph during architecture construction, charging that destination, and loan it from one graph declaration contract. Single-group declarations remain allocation-free and are validated from the source's actual group identifier.
- Remove the owned compatibility representation and allocating declaration defaults. Owning consumers explicitly materialize a declaration through their chosen metadata destination; that copy remains distinct from source discovery and authentication.

Snapshot metadata also retains its already-acquired destination authority and logical snapshot reservation while aliases escape. It adds no second charge; the existing reservation conservatively remains live until every snapshot metadata owner retires.

The final delivery audit removed enums that had become transparent wrappers around a single retained owner: APIs return `SharedCapturedStep` and `SharedSpeculativePrefillReductions` directly. Their diagnostic wire decoding and borrowed field access live on those canonical owners; no conversion can detach their custody.

Graph implementation now retains dynamic prediction and composite graphs during construction. `LayeredArchitecture::execution_graph` loans those source declarations or a validated single group; the metadata companion and owned enum variant are removed. Observation checks compare source group identifiers without constructing a graph. Media adapters retain one optional diagnostic destination so invalidated source evidence still produces funded errors. Explicit owning consumers use the same graph-copy worker with their selected destination.

All emission policies now reserve the same logical immutable-frame envelope through the shared observation policy. Native funded frame storage remains separately admitted at its actual physical layout. An explicit empty receipt needs its real envelope budget; underfunding rejects before constructing the owner. Control failures retain their original typed causes and expose borrowed source-chain or classification views without detaching their payload. The filter-specific identity alias is removed in favor of the existing general `SharedStorageIdentity`.

Before consolidation, the decoder source entry `take_original` had no callers and passed no pool to a worker that must reject shared sources without one. Consumers used `take_original_for_pool`. The canonical `take_original(claim, pool)` contract now makes the same account check mandatory in the loaded/aggregate source worker. Unique versus shared decoder storage remains meaningful ownership; the unauthenticated forwarding entry is removed.

The pre-consolidation source-carrier inventory found that `CapturePlanSource` split raw `Arc<AdmittedCapturePlan>` from `SharedCapturePlan`, with paired session and dense/sum/routed receipt constructors and optional-source fallbacks. Both carried the same immutable admitted semantics; only the latter preserved physical source identity and late accounting attachment. The implemented contract uses `SharedCapturePlan` for all sources, including explicitly unenforced ones. The carrier, paired constructors, optional-source branches and obsolete conversion-timing fixtures are removed. Callers create a shared owner explicitly at source construction; source identity is never reconstructed from an existing admitted alias. Retirement tests observe the actual shared source registration instead of a weak reference to the removed raw Arc.

Source consolidation is implemented: sessions, checkpoints, partition work, receipts and their errors retain `SharedCapturePlan` directly. Session and dense/sum/routed receipt construction each has one source contract. Failed child re-admission cannot publish an alias; successful children own independent source identity. The removed raw-Arc fixture is replaced by existing identity tests and actual shared-registration retirement probes. An empty funded schedule still requires terminal draining but creates and charges no frame; an explicitly requested empty receipt reserves its envelope.

Focused qualified validation after source migration: runtime capture suite passes 483 tests, including ordinary/funded quota parity, pending/failed delivery, source identity, transaction completion and late attachment. Before that final source-only migration, facade backend conformance passed 125 tests, including typed retained errors and snapshot authority retirement; authenticated chat binding passed four tests. Compiler qualification uses the actual pinned rustc path with `CARGO_INCREMENTAL=0` so fresh allocation layout checks are enabled.

The final boundary-aware prefill conformance suite also passes all 74 tests after the source consolidation.

## Current audit verdict

The representation and lifecycle consolidation described above is implemented.
The final bounded review found no remaining raw capture-source carrier, raw
backend drain compatibility hook, owning graph-declaration compatibility arm,
observed generation engine, or control record-mode family. Graph declarations
remain borrowed or validated single-group views; owning copies use explicit
destinations. Ordinary and controlled facade runs compose the canonical prepared
session, and captures retain their shared source and output owners.

The latest no-default public checkpoint passes 126 backend conformance tests and
38 portable facade tests, with two external-checkpoint tests ignored. Facade
units pass 372 tests with six ignored; the previously passed exhaustive schema
refusal sweep was filtered from that final rerun. Evidence:
`/private/tmp/tokenizer-final-portable-onig-coherent.log`. The later narrow
regex declared-pattern visibility change has its own focused tests recorded in
[tokenizer composition evidence](bounded-followup-tokenizer-composition.md).

This verdict is scoped to those contract representations and shared lifecycle
workers. It does not certify every backend path from static inspection. The
original Qwen-VL partition-media constructor/quote and genuine distributed reset
are implemented at the checked metadata boundary. The selected-executor optimized
artifact passes all seven public native cases, including combined image/audio
tools, resume-time child sampling, and TP2, PP2 and combined TP2/PP2 media.
Actual executor group-submission accounting closes the earlier TP2 admission
regression while preserving its resident registry. All six distributed CLI cases
also pass. Separate CLI/control and focused residency verdicts are tracked in
[partition source evidence](bounded-followup-partition-source.md) and the
[consolidation overview](bounded-followup.md). Explicit ordinary allocation
policies in low-level consumers are not compatibility fallbacks after an
original-source refusal.

The latest routed Host/Disk source work also retains one admitted immutable
exclusion owner across native and cold binding, plus the complete per-unit
constructor inventory under the same manager custody. Exact selected-name
validation precedes adoption; lease-row absence cannot grant exclusions or erase
independent bank placeholder constructors. Neutral routed/composite destination
coverage and the full architecture suite pass **772 tests with one ignored**.
The new owner/constructor backend fixtures passed in the scoped **66/66** native
backend checkpoint. The later selected-executor optimized public and CLI matrices
pass seven and six cases, respectively. Both corrected readiness fixtures also pass native execution, with exact
source-account retirement after the live child becomes terminal.
Exact evidence and the pre-existing cold partitioned prediction-extension limitation are recorded in the
[current parameter-source checkpoint](bounded-followup-partition-source.md#current-routed-parameter-source-checkpoint).


## Final prefill interface audit

The source trait still exposed an owned cache-identity hook whose default wrapped
it in a new shared allocation, although every production identity producer
already retained a shared owner. The canonical contract shares that existing owner;
the owned hook and its allocating fallback are removed. Two uncancellable/
unbudgeted forwarders had only test consumers and added no lifecycle behavior;
those tests now use the cancellation-aware source gateway with explicit
admission policy.
The progress, state-only and custom-operation entries retain distinct result or
operation contracts over the same source/lifecycle worker.


Focused verification passes 73 prefill tests (6.05 s) and 12 nonzero bounded
readout tests (38.32 s), including tensor/pipeline cuts, cache identity, cancelled
or failed preparation, uneven spans and cached decode. Logs:
`/private/tmp/eredu-prefill-canonical-identity-check2.log` and
`/private/tmp/eredu-prefill-canonical-bounded-readout.log`.

## Metadata producer audit

The final call-site audit found paired ordinary/metadata methods beyond the
already consolidated graph representation. Some state, identity, unit-path and
media-ingress defaults delegated to allocating ordinary producers. Parameter
descriptions also reconstructed owned tables that retained-source consumers
could borrow. These hooks now use one producer with an explicit optional metadata
destination, and a borrowed-or-owned parameter-description result whose owning
consumers fund their copies. An enforcement refusal does not select the ordinary
destination. Partition schemas and collective declarations follow the same
contract; their paired metadata hooks and default unsupported branches are removed.

The audit also found that observation binding validation regenerated row
declarations and names without receiving the checked caller's destination.
Closing the producer hooks alone does not close that call path; its validation
worker and declarations now receive the same destination. Runtime's full
integration suite passes 206 tests, including paid rebinding, exact source identity
and retirement through the last weak binding alias. The focused observation-path
suite passes 10 tests; the observation-source/capture suite passes 45. Evidence:
`/private/tmp/tokenizer-canonical-observation-binding-tests.log` and
`/private/tmp/tokenizer-canonical-observation-source-tests.log`. Final portable
public suites pass 126 backend-conformance tests and 38 portable-facade tests,
with two external cases ignored. Native qualification is tracked separately;
the focused source-contract checkpoint below passes 29 cases. Seven Metal public
cases and six paged control cases also pass; subsequent CPU source and final
distributed CLI qualification are recorded separately.

The state/parameter contract checkpoint passes all 205 runtime integration tests
(6.08 s, `/private/tmp/eredu-runtime-canonical-metadata-fixtures3.log`). Shared
physical parameter expansion passes exact format/member-placement parity and
refusal at every reached funding callback, with no later callback or tensor
access (one test, 0.01 s,
`/private/tmp/contracts-physical-parameter-expansion-final-tests.log`), including
the final fixed-control census. These results do not stand in for native execution.

The earlier canonical-observation architecture checkpoint passed 768 tests with one ignored. All 66 Moshi
tests pass, including ordinary affine declarations and nonzero selected-residency
execution with cached decode. Final logs:
`/private/tmp/tokenizer-canonical-observation-architecture-all.log`,
`/private/tmp/tokenizer-canonical-observation-runtime-final.log`, and
`/private/tmp/tokenizer-canonical-observation-public-tests.log`.
The requested portable feature checks and backend no-default metadata check pass;
the latter is compile-only. Commands and scopes are retained in
`/private/tmp/contracts-final-feature-results.json`.
The later 772-test architecture result above supersedes this count without
turning either neutral suite into native acceptance evidence.

## Final reset signature checkpoint

After the canonical reset placement hook began receiving the actual component
role and the state constructor began consuming the prepared reset context, the
portable `backend_independence` integration suite passes all 206 tests, with no
ignored or filtered tests (20.57 s build, 6.05 s execution). This supplements the
23 focused reset library tests; it exercises the integration implementations
affected by the signature migration. The backend no-default-feature metadata
check also passes (41.42 s). Reproduce from the repository root:

```sh
export RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc
export CARGO_INCREMENTAL=0
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR=/private/tmp/eredu-prefilter-reference/target
cargo test --offline -p eredu-runtime --test backend_independence \
  > /private/tmp/contracts-final-reset-backend-independence.log 2>&1
DOCS_RS=1 cargo check --offline -p eredu-backend-mlx --no-default-features \
  > /private/tmp/contracts-final-reset-backend-no-default.log 2>&1
DOCS_RS=1 cargo check --offline -p eredu-backend-mlx --lib --tests \
  --no-default-features --features metal,image,audio \
  > /private/tmp/contracts-host-hybrid-test-fixtures-check3.log 2>&1
```

The last command passes in 41.20 s and typechecks the corrected Hybrid copy,
Device-retirement and inactive parallel-control fixtures. Both `DOCS_RS=1`
commands use stub native bindings: they prove Rust metadata/type consistency,
not native linking or execution. The next native fixture run is recorded
separately in the [native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json).

Retained SHA-256 evidence for this checkpoint:

| Artifact | SHA-256 |
| --- | --- |
| `/private/tmp/contracts-final-reset-backend-independence.log` | `3f095224d6e16fc0bc86c6a4834e7e1997fe09178543a0c34c9124f9415cb4da` |
| `/private/tmp/eredu-prefilter-reference/target/debug/deps/backend_independence-790d9a4061bc0a5b` | `e7c198804c75b508e25dd87995903a323003ff56c8695492da28cbfff3d25b6d` |
| `/private/tmp/contracts-final-reset-backend-no-default.log` | `8c34a45fc49cc6b903ace0ffcb54218222c7134c9f354f490ab33c80be15f8fd` |
| `/private/tmp/contracts-host-hybrid-test-fixtures-check3.log` | `6cc790fc07ea93de9c0503e5f2dfd20c340ce9bb0ed035754c9349393a831443` |

## Final Host-view feature checkpoint

The final checked Host-backed Array-view attachment and its Rust consumers pass
`safemlx` library/test metadata (3.01 s), backend library/test metadata (40.39 s),
and backend no-default-feature metadata (50.05 s). The combined test check also
consumes the corrected paid Hybrid-copy and Prepared-stream control-retirement
fixtures. Using the pinned Rust/offline environment above, reproduce with:

```sh
DOCS_RS=1 cargo check --offline -p safemlx --lib --tests \
  --no-default-features --features metal,safetensors \
  > /private/tmp/contracts-host-view-safemlx-metadata2.log 2>&1
DOCS_RS=1 cargo check --offline -p eredu-backend-mlx --lib --tests \
  --no-default-features --features metal,image,audio \
  > /private/tmp/contracts-host-view-backend-metadata3.log 2>&1
DOCS_RS=1 cargo check --offline -p eredu-backend-mlx --no-default-features \
  > /private/tmp/contracts-host-view-backend-no-default.log 2>&1
/usr/bin/clang++ -std=c++20 -fsyntax-only \
  -I safemlx-sys/src/mlx-c \
  -I third-party/tokenizers-0.23.2/target/debug/build/safemlx-sys-257004b9a0b09799/out/build/_deps/mlx-compile-src \
  safemlx-sys/src/mlx-c/mlx/c/original_buffer.cpp \
  > /private/tmp/contracts-host-view-native-syntax.log 2>&1
```

The C++ syntax check exits successfully against the actual rebuilt native MLX
headers; its log is empty. All four checks are compile-only: `DOCS_RS=1` uses stub
native bindings, and `-fsyntax-only` does not link or execute. They do not establish
Metal retirement, finite-cap reuse, or public Host/Disk control parity. Those
native runs are recorded separately in the native work matrix.

| Artifact | SHA-256 |
| --- | --- |
| `/private/tmp/contracts-host-view-safemlx-metadata2.log` | `1dd2b78de48abad676226b4ffbe48a5bf0441481777a681cb4579c15b45c5467` |
| `/private/tmp/contracts-host-view-backend-metadata3.log` | `a1a8187afdc84b258a424be90f0f01b0c01527bf62d601754347395b51d7e1d4` |
| `/private/tmp/contracts-host-view-backend-no-default.log` | `3ff15b3cb887f006930540de8c62b5ba71b154ee32487c68dca06426fa9f4c63` |
| `/private/tmp/contracts-host-view-native-syntax.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Prediction registration retirement feature checkpoint

The canonical recovery result, same-node prediction registration retirement,
and retained branch-bank activation pass the combined backend library/test
metadata check (39.27 s) and no-default-feature metadata check (30.42 s).
Using the pinned Rust/offline environment above, reproduce with:

```sh
DOCS_RS=1 cargo check --offline -p eredu-backend-mlx --no-default-features \
  --features metal,image,audio --lib --tests \
  > /private/tmp/contracts-prediction-retirement-metadata5.log 2>&1
DOCS_RS=1 cargo check --offline -p eredu-backend-mlx --no-default-features \
  > /private/tmp/contracts-prediction-retirement-no-default.log 2>&1
```

These checks use stub native bindings and establish compile consistency only.
Subsequent native execution passes **52/52**: 48 recovery tests, the genuine
Host/Disk bank-activation and prediction-retirement test, and three nonzero
three-residency chat/plain/prefill cases. Exact commands and logs are retained in
`/private/tmp/eredu-prediction-retirement-focused-checks.json` (SHA-256
`8e1ea3c41bb418aa791321f5d7191f78ecf22f44238e2cf4aac7e545fc77345f`).
The subsequent shared Array/Host receipt artifact completes the full Dense
Resident, Host and Disk branch/control matrix under the unchanged 64 GiB ceiling.
The later routed-source checkpoint passes the focused native source and envelope
cases described below. Full routed execution remains a separate final public/CLI
qualification, recorded in the [current CLI scope](bounded-followup-cli.md) and
[native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json).

| Artifact | SHA-256 |
| --- | --- |
| `/private/tmp/contracts-prediction-retirement-metadata5.log` | `3ea031c10d892715b6128643c713fd51d0b41019499e9e5f10092c6e39efb2b3` |
| `/private/tmp/contracts-prediction-retirement-no-default.log` | `1fd488069697d418cb98d840234cc0a97233cbf0fb5aace53c5636e71078f924` |

## Lazy Host source receipt checkpoint

The sealed source-publication result now moves its existing custody and scalar
attachment receipt into the retained Host owner. Later publication checks the
actual native allocation, exact capacity, pool, and original account health before
omitting a redundant request sidecar. A failed source receipt cannot fall through
to a load-time receipt. The factory, borrowed observations and validation controls
are included in the existing prospective metadata census.

Portable `eredu-runtime --lib retained_` tests pass **36/36**, including closed,
foreign and quarantined Text/Speculative origins. The actual realtime-frame
account test passes **1/1**, including the same retained-origin cases. Combined
backend library/test metadata passes in **59.38 s**, and backend no-default-feature
metadata passes in **43.66 s**. Both backend checks use `DOCS_RS=1` stub bindings;
they establish compiler consistency, not native execution.

Exact pinned/offline commands, log and source hashes, the earlier eight-test
native baseline, and the current nine native filters are recorded in
`/private/tmp/contracts-lazy-host-receipt-checks.json` (SHA-256
`1b33bfc4826e7936e14f4d75655a956fedc56244b73603a23e2396625a97d76b`).
This lazy Host receipt checkpoint passes **9/9** native tests, including persistent Host
reuse with later-request retirement, initial receipts, ordinary/prepared origins,
closed donors, foreign and quarantined refusal, and independent/shared root
accounting. Exact native commands and log hashes are retained in
`/private/tmp/eredu-source-receipt-backend-checks.json` (SHA-256
`8ae9f19d6209d877cda4a21278a9719f21bb2894a9bcbdfe0b40e76c0626bab1`).
The same historical production checkpoint passes all seven public parity tests
and all six paged-control tests. Its CLI completes Dense/Resident and Dense/Host
but stops in Dense/Disk: a later trace-boundary request needs 3,413,467,366 bytes with
3,186,023,554 available under the unchanged 64 GiB ceiling. This is recorded in
`/private/tmp/eredu-native-source-receipt-cli.log`; the passing Host receipt tests
did not establish retirement of every Disk owner. The later canonical Array
receipt closes that observed Disk accumulation; its full Dense result is linked
in the current CLI scope above.

| Artifact | SHA-256 |
| --- | --- |
| `/private/tmp/contracts-lazy-host-receipt-metadata3.log` | `ec6d19547948b06eefd742e607c0671e7bfab9ffd3810cd0a1ce85234bdf5e76` |
| `/private/tmp/contracts-lazy-host-receipt-no-default.log` | `365653f095eb5ab56a4d61a18cca1cb84c1a729084cf265dc6c47fd3f0f9a9c3` |
| `/private/tmp/contracts-lazy-host-origin-tests.log` | `ec0d50d3df4e8d08fa28835d9d609df692bb10449280c6794f0152c7c349b4e6` |
| `/private/tmp/contracts-lazy-host-realtime-origin-test.log` | `10a7efe7cb87662536d7b544ed4663dd0e7acf03eb83d4c585ae7d4c6a621fb2` |


## Canonical cached Array receipt checkpoint

The later Disk census identified healthy closed request accounts retained by
registrations on cached CPU parameter Arrays. Those Arrays have independent
backing from their Host sources, and the final local unit legitimately survives
successive depth-two windows. The temporary scalar probes were removed after the
reproduction; no cache eviction or capacity change was used to hide retention.

Host sources and canonical Array cells now share one private scalar attachment
proof. A cell mints it only after successful checked originating attachment with
an exact match to the actual registration payer. Later observations snapshot
receipt presence independently and authenticate native generation/capacity plus
healthy original custody before skipping a redundant request sidecar. Plain-first
collection preserves canonical provenance without dropping a native handle under
the manager loan. Zero-capacity Host output carries no attachment proof.

The portable native-registration module passes **22/22** tests, including the new
actual attached-owner equality case (already included in that count). Final
combined backend library/test metadata passes in **33.40 s**; no-default backend
metadata passes in **44.07 s**. Both use pinned Rust 1.98, offline mode and
`DOCS_RS=1` stub bindings, so they establish compiler consistency only. Source,
command and log hashes are recorded in
`/private/tmp/contracts-canonical-array-receipt-checks.json` (SHA-256
`0c7b01218e497cdeffdf67b1d27e4c6065f055e5a6685c05217845b330aad267`). Native execution was pending at that compiler checkpoint.
Subsequent fixture corrections include both real factory source births in the
named collector's final baseline and distinguish initial physical Disk reads
from legitimate warm-window reuse. The latest native checkpoint below passes
those collector/refusal and four-request Disk retirement cases; no cache flush,
forced reread or capacity increase is used.


## Qualified source-contract native checkpoint

The selected routed source retains its actual independently cached bank and
policy. Its bounded local-row envelope combines completed equation populations
before joining the authentic maximum-row copy source once. Copy/aggregate
constructors, records, kernels, completion frontiers and storage remain charged
separately; CPU equation tape entries continue to match the actual CPU producer.
Runtime row specialization charges the accepted request and preserves the
immutable source's original account.

The addressable-equation source checkpoint passes **29/29** cases: two CPU/Metal population
composition cases, three genuine CPU/Metal bank envelope cases, four grouped CPU
source cases, seven scalar/floating Slice cases, two finite-row candidate cases,
ten source-publication/collector cases, and the four-request persistent CPU Disk
case. That Disk case verifies nonzero state/token parity, actual initial source
reads, stable cached physical identities, exact later-request retirement and
final source teardown. Bank envelope cases compare every admitted row count with
its exact source recipe, including separate copy costs and native kernel
transitions. They do not substitute for full routed-model execution.

Exact commands, counts and log hashes are retained in
`/private/tmp/eredu-addressable-equation-source-backend-checks.json` (SHA-256
`fde31670fd49438b1aa82e39a52fac3f15af4bce4f84060b96c069a900e292dd`). The final combined backend library/test metadata check passes
in **34.77 s** and no-default metadata in **23.59 s**; those compiler-only results
and their source hashes are recorded in
`/private/tmp/contracts-addressable-copy-population-checks.json`.
Subsequent Metal public and paged-control runs pass seven and six cases,
respectively, with the documented 64 MiB debug test stack. CPU source follow-up
and the final distributed CLI rerun have separate outcomes in the
[CLI scope](bounded-followup-cli.md) and
[native work matrix](validation/bounded-followup-native-work-matrix-2026-09-18.json),
separately from this completed focused contract qualification.
