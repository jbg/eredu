# Parameter and metadata consolidation inventory

The opening inventory describes the pre-consolidation state. The implementation
and validation checkpoints below record removal of those duplicate surfaces;
current native acceptance is maintained in [native validation](bounded-followup-native.md).

The bounded-inference changes added a borrowed source visitor alongside owned
parameter visitors, mutable owned/borrowed adapter pairs, retained-value
fallback traversal, and a transparent workspace funding wrapper over the core
account. Current producers and consumers still maintain both traversal forms.

* Canonical parameter observation borrows the retained `ParameterSpec` and
  current trainability. Remove owned callback defaults, borrowed opt-in flags,
  unavailable-compatibility callbacks and allocating fallback traversal. Source
  completeness and auxiliary retained tensors remain explicit; mutable slot
  replacement still has its distinct quiescent ownership requirement.
* Owned topology output is an explicit collection/conversion at a consumer
  boundary. It does not justify an independently maintained traversal engine.
  Derive and container implementations must use the same canonical contract.
* `WorkspaceMetadataFunding` has no storage or accounting responsibility beyond
  `HostMetadataFunding`; its two associated error/account aliases have no distinct
  semantics. Reexport and use the core names directly. Workspace-specific
  allocation helpers return NN errors and therefore remain NN policy through an
  explicit extension contract, with no wrapper allocation or conversion.
* The graph-interface audit continues before graph changes: distinguish retained
  source graphs from genuinely separate executable/metadata projections. Remove
  only compatibility representations, preserving source custody and accounting.

Validation will use existing behavioral traversal, mutation, source-lifetime,
derive, funding-refusal and account-lifetime tests, plus compiler checks of every
migrated consumer. No repository-shape tests will be added.

The funding wrapper has been removed. NN reexports the core account, funding
handle and error under their canonical names. `WorkspaceMetadataAllocation` and
`ParameterMetadataAllocation` implement only NN-specific fallible allocation and
parameter-copy operations on that core handle. The funding lifetime/refusal tests
pass, and NN plus architecture checks pass after consumer migration.

Immutable and mutable visitor callbacks now receive `ParameterMetadataView`.
The owned default conversions, borrowed opt-in flags and unavailable callbacks
are removed. Callers collecting owned metadata explicitly call `to_owned`; callers
reading names borrow them. Architecture declaration sharding no longer wraps
ordinary and borrowed metadata in a second enum. Native operator visitation uses
the audited declarations, with the legacy flattened-map traversal removed.

Mimi previously assembled complete names during every recursive parameter visit.
Its constructor now binds names once into the retained leaf specifications; later
visits borrow those identities. Canonical sources also describe attention KV and
streaming convolution buffers, and expose their actual slot bounds. The codec
all-features check passes. All eight codec unit tests and all five artifact
conformance tests pass with the qualified Rust compiler, including the new test
of 246 stable named fields and four live streaming buffers. The conformance
suite also checks numerical execution, streaming, layout and artifact loading.

The required source traversal now supplies named and retained-value projections;
independently maintained producer traversals and unavailable defaults are gone.
Incomplete sources return their concrete classification error, including through
ordinary parameter collection and binding. Derive, container, native and fixture
producers implement that same contract. The NN library run passed 204 tests and
exposed one metadata-only fixture field incorrectly marked as an unknown retained
field; correcting its explicit annotation made both topology tests pass. The
architecture production check also passes. Native consumer validation continues
with the shared native suite. The graph portion is consolidated independently
with source graphs borrowed and caller-owned output copies explicit.

Authoritative checks use
`RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc`
and `CARGO_INCREMENTAL=0`; the real compiler path is required for the runtime's
qualified allocation profile. Focused commands are `cargo test -p eredu-nn --lib`
and `cargo test -p eredu-codec --all-features`. Their temporary logs are
`/tmp/parameter-sources-nn-2.log`, `/tmp/parameter-source-topology.log` and
`/tmp/parameter-codec-tests-2.log`.

The native table audit found one more accounting-driven representation fork:
`NamedParameterTopology` dispatches between an ordinary BTreeMap and a shared
funded row table, with separate key iterators; five native operator fields retain
the ordinary map directly. The canonical form will be the sorted immutable row
table for both construction policies. Funding custody is optional only for the
explicit ordinary constructor; checked construction retains its reservation,
exact preallocated row limit, owned specification funding and final shared owner.
The production map representation and enum dispatch will be removed. A test-only
borrowed map adapter may remain for malformed-topology fixture inputs.

The production table fork is removed. All native operator fields retain
`NativeParameterTable`; exact source names come from the existing borrowed native
field traversal. Ordinary owned specification collection sorts once, while the
funded builder reserves its complete row capacity before accepting rows. Both
retain the same shared immutable table, and grouped named construction uses one
builder worker. The test-only map adapter supplies independent malformed inputs.
Named-field validation is separate from source completeness: ordinary wrappers
preserve an explicitly incomplete auxiliary inventory, while funded construction
continues to require a complete source. Compiler and native behavioral validation
were in progress at this table checkpoint; subsequent shared parameter-table
and released numerical results are recorded in [native validation](bounded-followup-native.md).
