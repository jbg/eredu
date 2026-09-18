# Native bounded execution

## Numerical and residency workers

Ordinary and admitted execution share numerical, layer-acquisition and
materialization workers. CPU, Metal and CUDA select their own arithmetic and
storage mechanisms. FP8 setup shares finite kernel definitions and fixed
invocation controls. Native source facts describe the actual operator inputs,
graph nodes, queued jobs, backing births and completion roots.

Layer acquisition has explicit finite-attempt and recovery policy. Foreground
and background I/O retain their completion witnesses. Host validation and
pinning compare the locked manager, destination and immutable owners. Temporary
parameter constructors use the selected CPU or Metal census. A bounded operation
does not invoke global cleanup or retry through an unfunded path.

## Completion and retirement

An inner event becoming ready does not retire an enclosing role with live
children. Completion requires its exact terminal snapshot. One monotonic
deadline governs nested readiness checks, with remaining time passed inward.
Timeout, polling error and busy recovery preserve source and producer custody.
Communication quarantine retains unresolved native resources until completion
or safe teardown.

Original backing receipts distinguish initial storage from request-local
publication. Persistent arrays retain their physical charge across requests;
Host sources have independent charges. Equal geometry or shared-looking pointers
do not establish alias credit. Reset cannot overwrite a receipt-bearing cell
without accounting for its owner.

Explicit reclamation drains submission, host, ordinary and native owner queues
as retirement progresses. Native callbacks can enqueue Rust custody whose final
destruction requires host reclamation. Reclamation never waits for unresolved
work, runs under a held native runtime lock, or certifies a live scope. Session
synchronization proves completion and health/idle, drains ready owners and checks
expired installed views. Startup and bounded reset do not perform global cleanup.

## Paged storage, copies and reset

Paged Host demotion attaches displaced Device reservations to the actual native
backings. Escaped aliases retain their physical charge until final drop. A
thread-affine prepared retirement receiver can destroy only its own completed
callback. Array availability and transfer-event synchronization are separate
witnesses for checked Host-view attachment.

Hybrid and KV state use a grouped copy worker. Every child table, including empty
tables, has funded metadata; zero payload does not imply zero header cost. Child
publication retains its charge and only the outer table emits prompt completion.
Paged copies retain their manager and failure custody.

`ResidentTableResetState` checks component roles and passes the prepared context
through shared empty-manager construction. Hybrid layers can combine paged
attention and fixed Device children. Empty local partitions retain their paid
context. Immutable layout, global indices and pool identity survive reset while
old pages and escaped tensors keep their owners.

Distributed reset agrees `SessionReset` before installation and
`SessionResetPublication` afterward. Pre-install refusal leaves state intact;
post-install disagreement fences the session and transport. These two exchanges
spend two cumulative attempts and refund no capture, copy or transport budget.

## Distributed media and source geometry

Observed and unobserved decoder execution share component workers and retained
model/control context. The sum-wave API validates all inputs under the payer,
constructs reductions before waiting and completes the whole wave. Quotes fund
placeholder, result, root-reference and transport destinations before allocation.

CPU broadcast/reshape validates the readable stride span and alias/copy plan;
broadcast repetition does not enlarge physical backing. Grouped F32 projection
uses its selected tiled GatherMM worker and includes all four input owners,
typed coordinates and queued index validation. Stable sorting, scatter and
reductions retain their specified arithmetic order. Indexed parent completion
has four actual operands.

Empty Broadcast, Full and typed structural views retain zero-byte Data metadata
and the complete source backing. Concatenation funds empty blocks as well as
populated inputs. Row gathers and additive scatter preserve repeated-destination
semantics. Selector gather validates selected-axis width and readable strides;
`LogAddExp` has its own floating worker. Frontend construction and evaluated
primitive populations have separate counts.

Selected CPU selector quotation covers dense F32 operands, one selection
partition, four scoring transforms, optional input/bias/coefficient transforms
and exact supplied integer IDs. Other configurations require matching source
qualification. This is an implementation coverage boundary, not a model-family
restriction.

## Kernel names and masks

The native kernel-name helper takes one value snapshot at entry and appends
borrowed values with a fold. Its census includes the selected argument pack,
string capacities and cache ownership. Passing the destination as an argument
preserves snapshot semantics.

`masked_scatter` requires the mask to match the input prefix, appends singleton
axes and broadcasts over unmasked dimensions. A `[1,17]` mask therefore matches
`[1,17,16]` rows. Quotes and numerical execution use that same geometry.

## Support limits

The tested environment is Apple CPU/Metal with Ring collectives. CUDA, NCCL and
other hardware require their own native validation. Public and CLI tests use an
explicit 64 MiB Rust test-thread stack; default 2 MiB debug-stack execution is
not established. Fresh funded file opening outside Unix is unqualified.

Cold partitioned prediction-extension destination projection returns
`PreparedExecutionError::UnavailablePrediction`. Bounded distributed speculative
bank admission returns `PrefillScopeUnavailable` when its producer provides one
stream with local cancellation. These limits concern prediction-extension and
speculative mechanisms, not ordinary next-token generation or tool calling.

See [validation](bounded-inference-validation.md) for commands and scoped results.
