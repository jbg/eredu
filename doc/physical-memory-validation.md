# Physical memory validation

The physical-domain ledger is validated independently of process RSS and native
residency telemetry. Neutral topology fixtures establish accounting behavior;
native tests establish the allocation, placement and completion facts supplied
by a particular backend and hardware configuration. Neither substitutes for the
other.

## Neutral conformance

`eredu-runtime`'s `domain_ledger` suite runs the same coordinator operations over
unified and separate topologies. It covers shared backing, independent copies,
source pins, transfer overlap, managed placement allowances, mixed limits,
concurrent admission, failed transactions, retained publication and a deterministic
reference accounting model. The `domain_workspace` suite checks physical-domain
reduction through the existing allocation-lifetime trace. Core tests cover
topology identity, limit resolution, checked arithmetic, requirement categories
and atomic component subtraction.

```sh
cargo +1.98.0 test -p eredu-core
cargo +1.98.0 test -p eredu-nn --all-features
cargo +1.98.0 test -p eredu-runtime
cargo +1.98.0 test -p eredu-architectures
cargo +1.98.0 test -p eredu --no-default-features --test portable_facade
cargo +1.98.0 test -p eredu --no-default-features --test backend_conformance
```

The local profile passes the complete core, NN, runtime and architecture suites:

| Crate | Unit and integration tests passed | Documentation tests passed |
| --- | ---: | ---: |
| `eredu-core` | 635 | 11 |
| `eredu-nn` (`--all-features`) | 238 | 5 |
| `eredu-runtime` | 2,407 | 26 |
| `eredu-architectures` | 1,333 | 5 |

The architecture suite has two explicitly ignored cases. The separate borrowed
parameter-preparation conformance run passes all four selected cases, including
callback failure, source-loan release and retry.
The portable facade passes 38 tests with two explicitly ignored cases, and the
backend-conformance suite passes all 128 tests with default features disabled.
The focused capture suite passes 511 cases with shared partition-frame and
delivery-timing plumbing. Explicit two-, three- and one-token invocations retain
nonzero spatial and summary values; invalid invocation geometry is rejected
before frame attachment, and retained aliases keep their funding until retirement.
The three selected speculative architecture cases
also pass with exact phase and depth forwarding. The affected portable workspace
regression selection passes 22 cases: 17 domain-report cases, four metadata-funding
cases and one finite-copy case. These cover source branch peaks, alias retention,
unknown controls, recorded metadata quotations and one-byte refusal. The actual
coefficient-upload projection case and both reset-readiness compile-fail examples
also pass.
Five focused runtime transaction cases pass: one-use conditional control
occurrences and loan return, rejection of replay and false success, source-plan
agreement with the actual output driver's successful and failed prefixes,
rejected-control fencing, and mechanism-binding restoration on success, error,
unwind and funding refusal. Their source queries construct no native work or
execution authority.
The model-control trace tests retain the actual group and route order, reject a
foreign trace, and keep model-internal agreements distinct from transaction
phases. The partitioned prediction quotation case passes prefill, proposal and
replay against the selected rank-local state and retained communication source;
missing communication is rejected before state projection. These are descriptive
workspace and ownership checks, not native collective execution.

Four focused dependency-allowance cases pass: exact fit and atomic refusal,
estimate/category separation, alias custody and independent equal labels,
checked overflow and topology rejection under unlimited limits, and cumulative
bounded-input admission through the same ledger. The affected 13-case domain
ledger conformance matrix passes, including concurrent multi-domain transactions,
managed placement, publication, transfer overlap and the reference accounting
model. These are neutral accounting results.

Focused persistence tests pass shared-manifest and failure custody, admission
refusal, prepaid rollback, and authentication against the supplied opened file.
Persistent block imports preserve identity, digest and both metadata accounts;
streamed copies preserve nonzero payloads and reject changed sources before
writing the destination. A malformed restored commit epoch leaves live state,
revision and distributed outcome unchanged, and a subsequent valid restore
succeeds. The bounded checkpoint-copy regression covers a nonzero multi-buffer
payload, truncation after a written prefix and a destination write failure.
Fixed-state input validation rejects malformed declarations before allocating
payload storage and retires refused payloads with their paying owners. Four
canonical numerical-publication cases cover unified and separate placement,
managed allowances, shared identities, atomic refusal, failed attachment and
independent observer/directory retirement. The ordinary completed-numerical
publication regression also passes without introducing a directory row.
The source-publication lifetime case covers both raw and funded speculative
source tables through successful and failed attachment. It checks fixed capacity,
one charge for prepaid backing, retention of the original payer after a move,
and final alias or error retirement.
Three canonical Host-table cases pass empty and nonempty backing discovery,
refusal of a foreign ledger, alias and pin custody, exact fit, one-byte exhaustion,
checked overflow, and destruction of a partial prefix before its payer retires.

The crate-boundary and feature checks in [the architecture rules](../AGENTS.md)
also apply. Neutral fixtures do not establish CUDA execution, multiple physical
accelerators, or native unified-memory behavior.

## Native profile and reproduction

The local profile is Apple Silicon, macOS 26.6.2 (25G83), Rust 1.98.0
(`88d9e12ae`, LLVM 22.1.8), Apple clang 21.0.0 (`clang-2100.3.34.2`), and the
MacOSX26.5 SDK. The native ownership profile qualifies libc++ 210106. Tests use
a 64 MiB Rust test stack and serial execution. Every test in a Metal-feature
executable runs outside the filesystem sandbox, including tests selecting a CPU
stream.

```sh
export RUSTC="$HOME/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc"
export RUSTDOC="$HOME/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustdoc"
export RUST_MIN_STACK=67108864
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export EREDU_REQUIRE_MUTABLE_PAIR_QUALIFICATION=1
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export SDKROOT="$DEVELOPER_DIR/SDKs/MacOSX26.5.sdk"
cargo +1.98.0 test -p eredu-backend-mlx --no-default-features \
  --features metal,accelerate --lib --no-run --message-format=json > native-build.jsonl
```

Select the executable from the build's `compiler-artifact` JSON record whose
target is `eredu_backend_mlx` and whose profile has `test: true`. Do not infer
the artifact from a previous build's filename. Large unstripped executables
exceed the macOS loader's mapping limits; create a separate copy with
`/usr/bin/strip -o <test-copy> <executable>` and run that copy with
`--test-threads=1`. `--include-ignored` is necessary for the explicitly selected
native CPU mechanism and state-copy regressions. A CPU stream in this executable
is native CPU evidence, not evidence of GPU execution. Backend, facade and CLI
test executables require fully stripped copies; retaining local symbols can
still exceed the loader limit.

The complete CPU `safemlx` wrapper suite passes 767 unit tests and 24 integration
tests, with two unit cases and one timing benchmark explicitly ignored. The
build's compiler-artifact records confirm that neither `safemlx` nor
`safemlx-sys` enables Metal or CUDA. The complete Metal wrapper suite passes
785 unit tests and 25 integration tests, with all 13 explicitly selected GPU
cases also passing. Both CPU and Metal documentation suites pass 115 tests with five
explicitly ignored examples in each profile.
The CPU and Metal commands use separate feature configurations:

```sh
cargo +1.98.0 test -p safemlx --no-default-features --features accelerate \
  --tests -- --test-threads=1
cargo +1.98.0 test -p safemlx --no-default-features --features metal,accelerate \
  --tests -- --test-threads=1
cargo +1.98.0 test -p safemlx --no-default-features --features metal,accelerate \
  --lib -- --ignored --test-threads=1
cargo +1.98.0 test -p safemlx --no-default-features --features metal,accelerate \
  --test events --test host_transfer --test scoped_physical_funding \
  -- --ignored --test-threads=1
cargo +1.98.0 test -p safemlx --no-default-features --features metal,accelerate \
  --test timing -- metal_ --ignored --skip metal_timing_submission_overhead_benchmark \
  --test-threads=1
cargo +1.98.0 test -p safemlx --no-default-features --features accelerate \
  --doc -- --test-threads=1
cargo +1.98.0 test -p safemlx --no-default-features --features metal,accelerate \
  --doc -- --test-threads=1
```

The explicit GPU selection covers completed readback without staging, native
allocation attribution, custom kernel outputs, transfers, completion-owned
scope retirement, two-stream event ordering and timestamp behavior. Timing
benchmarks remain excluded. Subprocess isolation for the explicit retirement
test preserves its no-implicit-reclamation assertion after other tests
deliberately stop CPU workers; child results are not counted twice.

Focused backend tests pass ordinary CPU and Metal text generation under finite
and unlimited limits, comparing controlled and uninterrupted nonzero output.
They also check exact fit, one-byte exhaustion, retained output and final
retirement. Grouped CPU tests pass the chunked and sequential equations and
their callback-storage populations. Native prediction-storage coverage passes
across resident, host-layerwise and disk-streamed configurations. Bounded
realtime Host and Disk tests pass against ordinary execution, including depth
tails and cached frames. The indexed bank query passes member-value, retained
result and subsequent-generation checks, including final request retirement.
Speculative completion preserves nonzero duplicate roots and the exact iterator
prefix. Four Hyper workspace regressions pass, including Metal residual cycles
against a scalar reference. The focused Metal attention matrix passes its
independent numerical reference and allocation-peak checks across F32, F16 and
BF16, contiguous and strided inputs, masks, sinks, soft caps, sliding tiles and
completed key-block phases. The shared placement source changes also pass V4
controlled/uninterrupted generation at the complete domain quote and refusal
one byte below it.
Activation retention, typed floating fill, ordinary canonical-cache retention,
and admitted canonical-cache alias retirement pass. The native CPU cross-stream
Fence regression passes in CPU-only and Metal-linked profiles under a finite
observer allowance, including final-alias retention, refusal and recovery.
The distributed resident, host-layerwise and disk-streamed capture cases pass,
including nonzero numerical comparisons and delivery-timing assertions. The
Boolean broadcast regression preserves authenticated backing identity and
nonzero values across its supported aliases. Indexed registration refusal
retains its prepaid host custody until the escaped error retires. The Metal
autoregressive capture case passes exact two-, three- and one-token invocations
with nonzero captured values. Ordinary communication completion and header
refusal preserve their admitted account through the final result or error owner.
Deadline overflow and cancellation timeout preserve quarantine custody. The
CPU Index source retains unknown input precision and full alias backing, and
independent CPU workspace composition preserves its qualified workers. The Metal
routing caller regression preserves unknown input precision and requires the
actual explicit F32 casts; the corresponding CPU mechanism remains unqualified.
Prepared allocator placement and copied-U32 source regressions pass in the
CPU-only profile.

Categorical admission rejects an unproven controller before native preparation.
The focused ordinary pooled-attention caller regression and CPU autoregressive
capture admission with nonzero two-, three- and one-token invocations pass.
The projected capture trace preserves those distinct invocation sizes as well.
The normalization caller alternative regression passes without inventing input
precision. The CPU and Metal grouped-normalization caller cases pass for unit,
learned and offset gains, positive and empty rows, and invalid group geometry.
The packed Sum source and caller case passes with full backing retention and
the actual slice-update packing operation. Fourteen neutral physical-domain workspace cases pass, including
mixed-domain output populations, allocation counts, retained views, state
displacement and borrowed-source exclusion.

Native acceptance remains incomplete. The GPU routing predicate and its CPU
alternative pass their focused regression. The grouped Metal caller preserves
unknown operand precision while requiring the actual F32 banks. The ordinary V4
gated-product and pooled-position callers pass their focused regressions. Full
V4 admission passes at its complete quotation and rejects one byte below it.
The actual GPU-to-CPU-to-GPU routing worker preserves finite control allowances,
alias ownership, refusal and recovery with both Metal synchronization modes.
The corresponding CPU-only crossing regression also passes. Retained GPU stream
loans authenticate their immutable source across execution contexts, while
constructors continue to reject active scopes without consuming their source.
The actual two-rank CPU factory constructs its retained source. Partitioned
speculative capture still requires native verification of shared observer and
capture funding; both CPU and Metal factories reach native prediction-state
preparation. The focused
`tests::distributed_pipeline_ring::ring_two_process_inkling_original_autoregressive_expert_sources`
regression uses the same AR provider worker with a nonzero Inkling EP2 fixture.
It covers the separate initialized-integer source bank across prefill,
verification, commit and replay against a local reference; its native result is
pending. The real two-rank Broadcast regression passes with its retained
publication group, three completion-owned arrays and nonzero output. Invalid
roots reject before submission, and quotation itself submits no native work.
Generated floating initialization and geometry-only initialization also pass
their source regressions: typed uploads retain their actual backing, while a
geometry declaration alone supplies no physical allocation provenance. Neutral
admission accepts the actual partition collector's supported
transforms independently of diagnostic wording, while rejecting mismatched
execution, artifact, catalog and phase identities.

The stale paged-catalog source and retained failure payer regression passes.
The nonzero checkpoint rollback and retained refusal payer regression passes.
Resident and Host-spilled paged CPU inference pass the nonzero
controlled/uninterrupted comparison and exact final-owner cleanup. The ordinary
Disk writer releases completed staging independently while retaining its durable
file and source pin; the focused test also verifies escaped Host aliases, foreign
context rejection and repeated-handoff refusal. Facade Host-paged and Disk-paged
execution pass their nonzero controlled/uninterrupted output comparison with real
tier transfers. Their snapshot, restore, fork and reset cases also pass, including
one-byte reset refusal, retained output custody and final-owner cleanup. Paged
tensor-parallel inference passes its two-process Ring comparison of ordinary,
managed and controlled output. Its retained Metal and CPU source populations
include the lazy predecessors, actual communication completion frontiers and
both possible fast Fence buffers. The distributed CLI regression remains
unaccepted.
The shared cache-transfer stream
retains native CPU registry storage for the registry's lifetime, while its
wrapper storage retires with its last alias. Its focused lifetime and foreign-ledger
rejection test passes. Installing its cold owner during loading preserves the
live loading guard and the installed owner's identity without a new reservation.
Host staging on CPU retires independently of a copied Device destination.
Host staging retained by a GPU view keeps its logical charge until the final
backing alias retires. Preparing its retirement attachments rejects one-byte
exhaustion while retaining the failure's payer.
The exclusive ordinary Host writer preserves its bytes and creator-thread
retirement in both CPU-only and Metal-linked native tests.
Its prepaid owner survives constructor refusal, worker unwind and the final
Host alias in both profiles. Fresh request allocation tests on CPU and Metal
preserve older cached roots and their charges, reject allocation when the new
source has no allowance, and retire new backing only after its final alias.
The constructor policy that permits borrowing cached roots retains its separate
ownership behavior in the focused compatibility test.
The independent cache-manager constructor preserves its source namespace,
ledger and final-owner custody with its quoted mutex controls. The rotary
quotation regression covers explicit input products, traditional and interleaved
layouts, and full and partial rotary dimensions through the selected caller.
Resident paged inference, the shared Disk writer's success and collision custody,
and V4 exact-fit admission also pass with the fresh request allocation policy.
Ordinary distributed expert admission passes the two-rank expert-parallel and
four-rank tensor/expert-parallel Ring cases. Both verify nonzero logits,
cache continuation and the exact logical owner/count exchange population through
the admitted communication worker. Speculative communication-source composition
still requires native verification; neutral suites do not establish it.
Completed-backing scan aliases preserve their allocation identity, nonzero values
and metadata payer through the final alias. Ordinary CPU controlled and
uninterrupted inference remains numerically equivalent under finite and unlimited
limits after the cold resource handoff.
The native CPU fixed-state importer preserves nonzero values through a bare
alias after its materializer retires. Its file staging releases independently,
ordinary registered-copy discovery accepts the surviving backing, and the final
alias retires the canonical record. Its empty-tensor case also passes. The
corresponding Metal import case passes with an actual GPU stream, including a
positively identified allocation-free empty tensor. It validates construction,
completion and ownership, rather than an arithmetic GPU kernel.
The nonzero Hybrid cache persistence case passes save, import, state comparison
and resumed controlled generation. Imported outer and nested Host tables retain
their canonical storage attachments through the continuation.
Completed scalar readback observes an already-signalled native event before
checking backing attribution; its focused regression rejects a pending source
and reads the completed nonzero value repeatedly. The owned communication case
also preserves its original account through the final owner. Public preparation
passes exact-fit and one-byte-short physical-domain admission, and V4 passes its
complete-quote boundary with temporary diagnostic custody included in the peak.
Concurrent public preparations share the same domain ceiling, reject further
admission through the actual exhausted metadata account, release only their own
reservations, and admit a replacement after one owner retires.

Focused neutral checkpoint tests pass source identity, exact metadata debits and
retirement of request/lease handles before their last paying owner. The neutral
autoregressive intervention fixture passes genuine admitted-role attachment,
shared partition observation, exact two-, three- and one-token geometries,
nonzero scale results and final funding retirement.

## Distribution

The release validator builds all published packages from their archives, checks
package contents, compiles packaged tests and examples, and checks consumers
using ordinary versioned inter-crate dependencies. The native MLX patches are
part of the `safemlx-sys` archive; no Cargo dependency override is used.
Validation of the final source archives and packaged consumers is pending.

```sh
python3 validation/validate_release_packages.py --allow-dirty \
  --target-dir target/physical-domain-packages
```

## Native coverage limits

CUDA validation is deferred to a separate run. The local machine has no CUDA
device, and this record includes no run on a CUDA validation runner.
CUDA compilation, multiple-device execution and
managed-memory native execution are therefore not established. Their neutral
conformance tests establish ledger behavior only.

GCC and MSVC ownership-layout qualification remains incomplete for the original
allocator, scheduler, main-thread guard, stream registry, worker TLS and
submission registry. CUDA backing placement and managed-domain allowances do
not establish those additional execution facts. Paths requiring an unqualified
profile return incomplete-attribution errors under finite and unlimited limits.
This is a software qualification gap, separate from unavailable CUDA hardware.

Fixed-state cache imports qualify Float32, Float16, Bfloat16, Int32 and Uint32
constructor sources. The Float64 constructor recipe is not yet attributed;
persisted Float64 state returns a typed missing-source error under finite and
unlimited limits. This is an implementation gap, not a hardware limitation.

Prepared routing controls have explicit source populations for forcing or
excluding experts, zeroing contributions, and biasing raw, transformed or ranking
scores. The CPU source requires actual contiguous F32 dense operands, input rank
one through four, unpartitioned selection and checked signed-index geometry.
Metal control populations compose the selected router's existing primitive
stages, retaining their dense/affine representation and arithmetic prerequisites.
These source contracts do not establish numerical validation for every action,
device or representation; native evidence belongs to the particular tested case.

Cold descriptors outside the qualified native lowerings still have incomplete
allocation populations. Examples include zero-row joint routing, partitioned
selection, packed encodings without a selected native recipe, and grouped
lowerings whose primitive or scratch-backing census is unavailable. Known tensor
and host-staging subtotals remain available. Missing backing-control facts appear
in `unpriced_host_operations`, and the total physical requirement remains
incomplete. Admission requiring those facts rejects under finite and unlimited
limits. These are software coverage gaps, separate from hardware availability.
The mixed Metal completion query qualifies one GPU stream and one retained CPU
source. A graph combining separate CPU router and Ring streams still needs a
three-stream completion source; this is also an attribution coverage gap.

Inference continues to require complete workspace and ownership evidence.
`MemoryOverheadPolicy` remains an independent neutral evaluator; this ledger
does not integrate it into inference admission or establish a process-memory
ceiling.
