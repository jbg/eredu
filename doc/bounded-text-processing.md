# Tokenizer and text memory policy

## Stock tokenizer execution

Eredu uses the published Hugging Face `tokenizers` implementation for model
construction, normalization, added-token matching, pre-tokenization,
postprocessing and encoding. Ordinary and managed encoding consume the same
serialized model configuration through upstream public APIs. BPE, WordLevel and
deterministic Unigram keep their model semantics; Eredu does not implement a
second tokenization engine.

Managed sources own an immutable stock tokenizer, a vocabulary index and Eredu's
fixed-buffer decoder program. The decoder compiler checks whether the selected
decoder profile supports incremental generation. A grammar input derivative
removes `Prepend` normalizers and changes Metaspace's prepend scheme to `Never`
through the public serialized configuration, retaining its relationship to the
original source. Changes to that derivative do not mutate the original tokenizer.

Encoding calls the upstream worker once and retains its complete `Encoding`
with the returned ID slice. Offsets, masks and other upstream fields remain
owned with it. An upstream error is retained through the public cause chain;
failed encoding does not publish partial IDs.

## Admission estimates

`TokenizerMemoryEstimate` reserves 64 KiB plus 128 bytes per serialized source
byte for construction, and 64 KiB plus 512 bytes per UTF-8 input byte for each
encoding operation. `TokenizerPlan::with_memory_estimate` selects different
fixed and proportional estimates; the compiled source retains that policy.
Planning checks arithmetic without constructing the tokenizer. Malformed
configuration is diagnosed by the admitted upstream construction attempt.

These allowances cover opaque dependency storage, transient model copies,
regular-expression work and upstream encoding output. They are estimates,
not measured capacities or enforceable dependency/process memory ceilings.
Reservations remain associated with the source, completed output or failure
until retirement. Independently owned operations require their own admission.

Eredu's decoder and generation-domain buffers retain their own checked geometry
and destination admission. Model weights, state/KV caches, activations,
transfers, residency and native execution use separate resource contracts.
Host dependency estimates do not replace those contracts. Input and concurrency
limits must also be selected for the application; a small estimate does not
constrain an upstream allocator or interrupt an expensive regex operation.

## Cache policy

`ModelCachePolicy { capacity }` configures upstream BPE and Unigram cache entry
limits. The default is 10,000 entries; zero prevents new cached entries. Managed
tokenizer construction disables model caches before encoding. The cache policy
is an entry count, not a byte limit or native allocator-cache policy.

BPE caches are per model and per thread. Applying a different policy through
upstream's public tokenizer API clones and replaces the model, so both copies
coexist temporarily. Clearing a cache invalidates entries but does not reclaim
retired BPE cache generations held by other threads. Construction headroom and
application worker limits must account for these behaviors.

## Chat templates

Ordinary and prepared chat use stock MiniJinja with shared compatibility helpers.
Signed `range` supports descending progressions and checked wide intermediates,
and rejects zero steps and more than 100,000 elements. Source normalization
uses MiniJinja's public parser to rewrite slice expressions into one shared
filter. The filter preserves omitted negative bounds, empty slices and Unicode
scalar indexing. Upstream still compiles and executes the complete template.
The MiniJinja version is pinned because its public parser API does not carry a
semver stability guarantee.

`ChatMemoryEstimate` defaults to 64 KiB plus 128 bytes per serialized source byte
for construction and per serialized request-input byte for rendering. Temporary
normalization and filter storage are covered by these estimates. Rendering also
admits its first-party output destinations.

`ChatRenderLimits` defaults to 8 MiB of serialized input, input depth 128, 1 MiB
per output variant, 10 million VM instructions per variant and recursion limit
256. Each variant renders once with the same request clock observation. Output
limits and fuel do not bound intermediate dependency allocations or elapsed
time. Failed rendering retains its partial output under the original account.

Managed grammar and tool-schema work has its own configurable
`DependencyMemoryPolicy`, selected through
`LoadedModel::prepare_chat_with_grammar_memory`. Its default is 64 KiB plus
128 bytes per logical input byte. Grammar sessions, snapshots and activation
tokenization retain the selected policy; template and tokenizer estimates remain
separate. See [backend architecture](backend-architecture.md#bounded-inference-ownership)
for the source, lifetime and admission contracts.

## Validation

The text and runtime suites cover tokenizer composition, Unicode, added tokens,
cache policy, incremental decoding, shared range/slice behavior, source identity
and refusal lifetimes. Portable facade conformance compares ordinary and
controlled generation, including non-default grammar headroom. These behavior
tests do not establish dependency peak memory or tokenization throughput.
