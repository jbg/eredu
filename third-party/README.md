# Pinned parser sources

These workspace members provide dependency-owned allocation facts and checks
before parser-table growth. They are selected with `[patch.crates-io]` in the
workspace manifest. Backend selection, resource custody and memory admission
remain in Eredu's existing neutral contracts and runtime.

| Package | Upstream archive | SHA-256 |
| --- | --- | --- |
| derivre | 0.3.12 | `a892ec3953f4ca893e8d2086ed03d4b58e19c77ffb864a574b332d7d40c175f3` |
| llguidance | 1.8.0 | `207e72ede15e79f1e7a7f0e2683387da031c1931f6de650020afb80c571efb67` |
| regex-syntax | 0.8.11 | `d6f6ff9a378485b298a5286656da665ba74413d36db0979633275d2e708145d4` |
| referencing | 0.52.1 | `d38a014525040cdc9893361b7419bcf1f43b7ba7055eabaf6d91d6b75caa8b3f` |

The original crates.io archives were checked against the pre-change Cargo.lock
before extraction. `parser-upstream.json` records those hashes, each original
file's SHA-256, and the archived VCS metadata. Archive hashes are authoritative;
llguidance's archived VCS metadata reports a dirty checkout. Each package retains
its upstream licenses, original manifest and source artifacts.

These forks inherit the workspace `unsafe_code = "forbid"` lint. The local
llguidance package builds only its Rust library. Its upstream C-ABI modules,
static/shared C libraries, header generation and header-copy build script are
excluded from compilation. The original C-ABI sources remain as provenance
artifacts; this fork does not expose that API. No portable unsafe-code exception
is introduced. Published third-party dependencies outside these forks retain
their own upstream boundaries.

The llguidance archive includes integration-test sources that import an
unpublished `llg_test_utils` helper, and a benchmark that references a schema
outside the archive. Their targets and unused test dependencies are omitted from
the local manifest; their original sources remain recorded. The self-contained
upstream library tests, local public-parser tests and Eredu's grammar conformance
tests are executable. Derivre's packaged upstream tests remain enabled.

Compiled llguidance grammars have one closed shared owner used by ordinary
parsers and funded Earley state. Publication reserves its shell through the actual
compiler account; final retirement frees the shell before the immutable graph
and its account. The factory accepts that shared source without a deep copy.
There is no infallible deep clone of a compiled grammar. Explicit independent
copies remain fallible operations with their own construction requirements.

The hash-cons table's retained-capacity API describes its owned vector capacity
and hash-table allocation. A prepared table may insert within its fixed
capacities without allocation, including duplicate lookup when full. Preparing
those capacities itself allocates and requires the caller's existing authority;
a requested element count is not a byte-budget certificate.

Incremental insertion reserves explicit scratch headroom and writes directly at
the uncommitted frontier. Successful insertion publishes those same words;
duplicate, rejected, dropped or unwound insertions clear their written scratch.
Write failures remain terminal for that insertion. A deliberately forgotten
guard leaves the owner fenced against further mutation until destruction, while
committed lookups remain valid. Scratch contributes to the reported capacity.

Expression encoding shares one implementation between the ordinary arena and
the prepared writer. The raw API checks encoded length, preserves flags and byte
layout, and returns a table-local `u32` identifier. It does not seed an ExprSet,
validate child references, or supply a complete expression/cache memory bound.

The lexer checks a new state's count and representable table geometry before
inserting its interned description or growing descriptor/transition storage.
Existing states remain reusable at the limit. These are local table guarantees:
derivative computation, tokenization, grammar compilation, mutable parser state,
copies and other caches still need complete accounting before finite managed
grammar admission can succeed.

The state cap is installed before the two sentinel states are built. Limits below
two reject with a typed configuration error before grammar compilation. Initial
states, first-byte warming, optional large-lexeme precomputation, initial parser
rows and skip selection all preserve the original typed cause on failure; a
failed lexer is never published as a successful parser. Independent clones keep
the same per-instance cap. Other compiler/parser allocations still require their
own bounds.

The regex-syntax fork adds borrowed allocation admission to the existing AST
parser and HIR translator. Ordinary construction supplies the unenforced policy
to those same workers. The dependency reports its actual box, vector and string
producers; it does not depend on host accounts or backend resources. Iterative
tree teardown uses explicit geometric stacks, with storage credit reserved
before node creation. HIR move/retirement sentinels and diagnostic formatting do
not allocate. The caller retains its funding owner through syntax, parser and
error retirement. Integration status and focused validation are recorded in
`doc/bounded-followup-regex-syntax.md`.

Matcher failures preserve the known closed configuration/parser causes and their
diagnostic text. They do not retain arbitrary constructor errors that might own
unrelated resources. Diagnostic cause inspection after a caught callback panic
does not block on or recover a poisoned lexer; it preserves the original failure.

TokenParser's ordinary and deep copies use one exhaustive field-copy helper.
Deep copying constructs its independent parser once, removing the former
temporary ParserState copy while retaining history, caches, limits and errors.
Ordinary copies still share the lexer. This reduces copy work; independent-copy
allocation authority and complete byte bounds remain separate requirements.

When updating a fork, verify the replacement archive, refresh provenance, review
the local changes against the recorded upstream files, and run its relevant
upstream and Eredu behavior tests. Do not edit Cargo's registry cache in place.

## External HF tokenizer dependency

`tokenizers` 0.23.2 is vendored as an external dependency and explicitly excluded
from workspace membership. Both direct text/facade manifests select the same relative pinned path; the
root `[patch.crates-io]` also unifies registry users. No registry-cache change or
extra build configuration is needed for normal source/path consumers.
Its crates.io archive SHA-256 is
`7afbf6e88718afcc138bad01d6ccc3051dbbc3b2ce9793d8b8a3aeb610969cfc`.
`tokenizers-upstream.json` pins every original archive file and VCS metadata;
`tokenizers-borrowed-decode.patch` records the safe local accessor/test changes.
The original package manifest, license and upstream unsafe code are preserved.
Unlike the parser forks above, this package does not inherit workspace lints;
its pre-existing external dependency boundary is unchanged. Every workspace
package retains its existing lint inheritance and portable unsafe prohibition.

The local additions borrow the concrete model/added-token decode vocabulary and
Replace pattern without cloning maps, strings, regexes or normalized caches.
They preserve canonical added-first spellings and special membership. They do
not bound HF construction, runtime caches, residency, encoding or arbitrary
caller work, and they do not provide original admission. Eredu's fixed compiler
uses those borrows for a checked one-attempt construction; text/facade retains
all tokenizer and termination policy.

Registry publication needs the borrowed accessors upstreamed or a published
patched dependency; vanilla crates.io 0.23.2 lacks these methods. No publication
is implied by this local fork.


The HF borrowed decode patch also provides a direct ID iterator for the facade's
loaded metadata view. It traverses the same four concrete model forward maps and
added reverse keys as the existing visitor, without a callback, copied vocabulary,
dense-ID scan or new allocation. Unigram uses its final forward-map population,
preserving repeated-spelling behavior. The generic Model trait and encoding/cache
implementations are unchanged. The same upstream archive and local patch ledger
remain authoritative.

### Explicit HF model-cache construction policy

The additive `tokenizers-model-cache-policy.patch` applies after the borrowed
accessor patch. Its JSON manifest pins every exact before/after file and the
unchanged upstream archive. The same existing relative Cargo paths consume
this external dependency; no registry mutation or extra Cargo configuration is
required. The upstream unsafe boundary and workspace exclusion remain unchanged.

`ModelCachePolicy::NoModelCaches` reaches concrete JSON/model constructors before
BPE creates its descriptor or Unigram creates its cache map/lock. Clone preserves
absence. Existing constructors and `Deserialize` retain Legacy defaults. This
choice does not bound input parsing, model/matcher/regex/global state, residency
or encode/decode. BPE cache clearing/resizing is not a TLS retirement mechanism.
Bare HF JSON omits resource policy; explicit seeded reconstruction is required.
The facade's private version-2 recipe carries that policy before nested HF
construction; old recipes retain Legacy semantics. The reported current cache
presence is not source provenance, an original allowance, or a managed capability.

Registry publication still requires a published patched dependency or upstream
support for these APIs; unmodified crates.io 0.23.2 does not provide them.


### Static HF ByteLevel alphabet

The additive `tokenizers-static-alphabet.patch` follows the borrowed-accessor
and model-cache-policy patches. Its provenance pins that exact predecessor and
the final external source tree. ByteLevel normalizer/pretokenizer/decoder/offset
processing share one compile-time readonly byte alphabet and a checked scalar
inverse; three lazy heap maps are removed. The existing GPT regex is acquired
only when `use_regex` is enabled. Public alphabet methods still allocate their
owned sets; serialization, Legacy/NoModelCaches choices, whole-token fallback,
Unicode alignment and offset behavior remain unchanged.

This closes only those mapping allocations and unnecessary disabled-regex
initialization. Other regex/global state, per-call buffers and HF construction
remain unbounded here. No original owner/account API or managed gate is added.
The existing relative dependency paths, workspace exclusion and upstream unsafe
boundary remain unchanged; registry publication still needs a published patched
dependency or upstream support.


### Packed HF BPE model construction

`tokenizers-packed-bpe.patch` follows the borrowed-accessor, cache-policy and
static-alphabet patches. A checked borrowed model-JSON plan constructs actual
Packed storage in the existing HF BPE with one reserve per destination and no
model cache. The ordinary merge engine and borrowed decode vocabulary consume
the same model. All partial allocation/semantic failures retain their actual
buffers and fixed causes; final controls and capacities have source-derived
requirements. Legacy builders, serde, Clone, training and owned vocabulary
results retain their ordinary allocating behavior.

This is a model constructor, not full-tokenizer construction or a funded owner.
Non-null dropout and ambiguous reverse IDs remain explicit construction-profile
gaps; enclosing tokenizer components, globals, operation scratch and resident
HF/input custody remain separate work. The external dependency stays excluded
from workspace membership and retains its existing unsafe/feature boundary.
Existing relative Cargo paths consume this fork; vanilla registry 0.23.2 does
not provide the added API. Publication needs upstream or a published fork.


The original BPE residence successor applies `tokenizers-bpe-residence.patch` after
`tokenizers-packed-bpe-compile-correction.patch`. Its JSON provenance pins the
complete corrected 110-member predecessor and 111-member successor. It adds a
borrowed vocabulary view and disabled-by-default `packed-bpe-test-support` for
the existing seven capacity-overflow frontiers. Existing merge/storage semantics,
Legacy APIs, default features and the external unsafe boundary are preserved.

The repository's relative path dependencies consume this exact external fork.
Vanilla crates.io tokenizers 0.23.2 does not provide these additions; registry
publication requires upstream support or a published patched dependency. No
publication, registry mutation or workspace-membership change is part of this
unit. Runtime reaches the concrete model only through non-Clone text adapters.
Complete tokenizer construction/residence and managed encoding remain separate.

### Packed added vocabulary

`tokenizers-packed-added-vocabulary.patch` follows the exact 111-member BPE-residence successor and produces the 114-member fork indexed by its provenance JSON. The scanner formerly in BPE compilation is shared as a private borrowed JSON utility; BPE errors keep their original classes and offsets. The added constructor retains real model/normalizer borrows, uses four actual destination reserves and returns a move-only partial failure. Its packed matcher feeds the existing extraction pipeline.

Legacy defaults retain the DAAC matcher. Packed `AddedVocabulary::get_vocab` and `get_added_tokens_decoder` return explicit owned maps through Cow; borrowed token/ID/spelling APIs avoid those copies. Full Tokenizer owned-map APIs keep their behavior. Clone, mutation conversion, serialization and ordinary HF construction are outside admission. The fork remains an external dependency outside workspace membership, with the upstream unsafe boundary and features unchanged. Registry publication would require upstreaming these APIs or publishing the patched dependency; vanilla tokenizers 0.23.2 is not a replacement.

### Packed added-token prefix search

`tokenizers-packed-added-prefix-search.patch` follows the corrected packed-added HF114 successor. Its provenance JSON pins both complete manifests and the exact patch. It replaces repeated whole-pattern scans with allocation-free prefix-range search over the same immutable packed spelling order, preserving raw/normalized extraction semantics. The upstream archive, features and unsafe boundary are unchanged.


## External regex-automata workspace prerequisite

`regex-automata` 0.4.18 is an external vendored dependency selected by the root
registry patch and excluded from workspace membership. Its published archive is
`ad8553b9b26413251cbf30e620595c7a41b3887f03da04579c0e6b0d6a06b4b2`.
`regex-automata-workspace.json` and its patch record complete upstream provenance
and four Rust deltas: a closed borrowed workspace, tests, child export and private
SparseSet reserve helpers. All ordinary APIs/features/manifests and upstream
unsafe statements remain unchanged; no workspace lint is weakened.

The PikeVM workspace's seven buffers are fallibly prepared from actual NFA
geometry. Capturing delegates and Unicode word assertions retain that mechanism.
The capture-free tokenizer profiles now use immutable DFA tables through the
same fancy VM and iterator workers for ordinary and enforced construction.
Private recipes are emitted by the pinned ordinary compiler in both byte orders.
The checked source constructor validates exact input/profile/table structure
before allocating, then declares and copies five vector tables and one boxed
owner. Every partial allocation remains owned on failure. The superseded NFA
recipe constructor and capture-table adapter have been removed.

## fancy-regex 0.19.0 canonical engine fork

Tokenizer and schema consumers now use the exact selected 0.19.0 engine.
The 0.17 production source is removed; its historical patch and archive hashes
remain as provenance. `parser-upstream.json` records all 58 original 0.19 files,
its archive hash and upstream revision; the MIT license remains in the fork.
The portable workspace member inherits the unsafe-code prohibition.

The original 0.19 VM and iterator progression each have one shared worker.
Explicit tokenizer workspaces retain capture-free immutable DFA delegates and
native character-class instructions; capturing delegates use the tagged NFA.
The regenerated closed inventory has 15 distinct DFA sources. Original dynamic
source construction and ordinary pooled scratch retain their upstream behavior;
general schema compiler/runtime funding remains unfinished and is not granted
by the closed tokenizer constructor.

`doc/bounded-followup-schema-regex.md` records the semantic upgrade audit,
independent pristine reference, exact storage/refusal tests and current full
split measurements. The older measurements in `bounded-followup-tokenizer.md`
remain historical evidence for the preceding 0.17 consolidation. No runtime
source compiler runs inside the admitted immutable-table constructor. These
APIs are absent from vanilla crates.io releases; publication requires the forks.

MiniJinja lexical successor: apply `minijinja-checked-frontend.patch` after the complete `minijinja-original-chat.patch` (including its safe-module lint correction), then verify the exact member manifest in `minijinja-checked-frontend.json`. The new default-syntax constructor shares ordinary scanning and literal workers; full parser/codegen/VM ownership remains unfinished.

MiniJinja expression successor: apply `minijinja-checked-expression.patch` after the original-chat and checked-frontend patches and verify every member in `minijinja-checked-expression.json`. The actual ordinary grammar is shared with a closed flat expression owner; full statement/codegen/chat-profile admission remains unfinished.

MiniJinja full-template syntax successor: apply `minijinja-checked-statements.patch` after original-chat, checked-frontend and the Enclose-corrected checked-expression patch. Verify all 117 members in `minijinja-checked-statements.json`. The same ordinary grammar constructs either boxed ordinary ASTs or closed source-bound flat syntax; this does not admit general codegen or additional runtime chat profiles.

MiniJinja macro capture successor: apply `minijinja-checked-captures.patch` after original-chat, checked-frontend, Enclose-corrected checked-expression and checked-statements. Its `minijinja-checked-captures.json` records all 128 exact members. The ordinary metadata worker and closed four-buffer workspace share one traversal; the unchanged original meta source remains a test-only independent reference. This prerequisite does not close general compiler/runtime or whole-request admission.

MiniJinja constant-fold successor: apply `minijinja-checked-constants.patch` after the complete original-chat/frontend/expression/statements/capture queue, then verify the exact member manifest in `minijinja-checked-constants.json`. The ordinary fold dispatcher and fixed unary kernels are shared; the closed descriptor route retains one checked reserve and explicit unfinished operation/materialization errors. General code generation and public/default activation remain unclosed.


## Existing serde_json numeric source producer

`serde_json` 1.0.151 is selected at the same locked version through a local
`[patch.crates-io]` path and remains excluded from workspace membership. Its
crates.io archive SHA-256 is
`c841b55ecdae098c80dcae9cf767f6f8a0c2cdb3416bbef72181df4d0fe73f14`.
`serde-json-upstream.json` records the archive member hashes and original VCS
metadata. The MIT/Apache licenses and upstream manifests/sources are retained;
no dependency or version is added and no registry cache is modified.

`serde-json-bounded-number.patch` adds one borrowed numeric plan and owning
control/error-layout queries. Apply `serde-json-bounded-roundtrip.patch` next;
its JSON pins the exact predecessor and changed member images. The consumed
parser uses the same ordinary Number deserialize/end sequence, conversion and
rounding. The query knows the private syntax-error Box body; numeric source
slices exclude string, container and custom visitor paths.

With `float_roundtrip`, the bounded invocation reserves its ordinary digit
scratch once to the numeric token length. The lexical float Bigint constructor
uses a reserve derived from the actual MAX_DIGITS, cached exponent range and
binary exponent limits. Its multiplication uses the existing iterative
small-power worker, and the shared limb shift uses resize/copy_within. The two
possible Bigints need no growth or temporary product vectors within that float
profile. The owning query reports both backing requests, digit scratch and
fixed parser/lexical controls. Ordinary callers retain the same conversion and
rounding; the slow path also uses these shared allocation workers.

The actual facade feature union includes alloc, default, float_roundtrip,
indexmap, preserve_order and std; no dependency feature is disabled. Successful
Number results own no heap payload; synchronous scratch retires before return.
The original syntax error is relocated in its same Box to the whole document,
and the enclosing source allowance retains that error through teardown.
`arbitrary_precision` has separate retained Number string ownership and remains
an explicit pre-admission layout refusal. Numeric spellings must fit the
ordinary signed digit counter. Publishing requires these additive hooks to be
upstreamed or a published patched build.

`toktrie-1.8.0` is copied from the exact Cargo.lock registry archive; its upstream
archive hash, VCS revision and file hashes are in `toktrie-upstream.json`. The
local change exposes source-derived construction destinations and owning failure
prefixes around the same ordinary trie insertion/encoding workers. It also makes
DFS scratch explicit and preserves duplicate-token order with an in-place sort.
The crate inherits the workspace `unsafe_code = "forbid"` lint. Its two
unchecked trie/bit accesses use the existing checked slice/bit operations. This is a trie construction producer, not complete grammar admission.

The pinned hashbrown 0.17.1 dependency exposes an allocation-free prospective `HashTable::try_reserve_layout` query. Ordinary reservation and quotation share the reached growth/in-place-rehash choice; the actual `TableLayout` computes both requests. A quote names the full new allocator request while the old table remains live, excludes allocator-private excess/bookkeeping, and expires on mutation. The untouched upstream unsafe implementation remains an external dependency excluded from workspace membership, like the published crate; newly authored quote and conformance modules forbid unsafe code. `hashbrown-requested-layout-upstream.json` records the archive checksum and original file hashes.


`jsonschema` 0.52.0 is pinned from its published archive, checksum
`0ee5867390f14ee43278d8d05d3af5f5314eeccb4684e6049033f2ba4ae3d4ae`,
and selected by the root registry patch. Its external dependency boundary and
features are unchanged; its local Rust unsafe lint is strengthened to `forbid`.
The upstream test/benchmark targets are omitted from this production source
copy; the pinned archive retains them. `jsonschema-workspace-upstream.json`
records the original source hashes.

The first local adaptation shares recursive memoization between ordinary
validation and a private paid context. Context stack growth reserves the full
new Vec request, and cache insertion uses the existing pinned hashbrown exact
prospective table request before the shared reserve/insert worker. Original
contexts borrow an initialized cold hash seed, avoiding lazy global seed
allocation inside admission. First refusal is retained until context teardown;
all actual scratch precedes its payer loan in drop order. This is context
construction machinery, not a full-schema admission API: input parsing,
unevaluated masks/sets, uniqueness, regex, content and numeric scratch remain
to be joined before public tool completion can use it.


## Reference registry and URI source storage

The referencing 0.52.1 fork owns prospective registry, cache, index, pointer,
anchor, vocabulary and resolver allocation facts. Its ordinary and enforced
entry points use the same producers. Bundled meta-schema bytes enter the shared
funded JSON producer under the actual source owner; preparation does not adopt
lazy global parsed values. External retrievers require a prospective callback
contract under enforcement, including their returned values and errors.

The already selected fluent-uri 0.4.1 fork owns the actual normalization,
resolution, owned-copy and fragment buffer requests. Its archive SHA-256 is
`bc74ac4d8359ae70623506d512209619e5cf8f347124910440dbc221714b328e`;
`parser-upstream.json` retains all original file hashes and VCS metadata.
The independent validation executable checks original hashes before comparing
URI behavior against the pristine archive library. See
[`doc/bounded-followup-referencing.md`](../doc/bounded-followup-referencing.md)
for the concrete tests and the callback boundary.

## ECMA regex source translation

The exact selected jsonschema-regex 0.52.1 source is a portable workspace member.
Its shared ordinary/funded workers expose original AST/HIR parsing, visitor
stacks, replacement strings, syntax group/name tables, literal optimizations,
and witness construction through the neutral regex-syntax allocation callback.
Fixed syntax and allocation failures remain distinct. The original archive
hashes and VCS revision are retained in `parser-upstream.json`; its declared MIT
license notice is included from the same project's existing 0.52.1 value fork.
See [`doc/bounded-followup-schema-regex.md`](../doc/bounded-followup-schema-regex.md)
for refusal tests and the independently linked pristine-source comparison.

## Schema text and numeric invocation producers

Exact idna 1.1.0, idna_adapter 1.2.2 and icu_normalizer 2.3.0 sources keep the
original UTS 46 and normalization data and workers while exposing prospective
borrowed allocation hooks. Exact fraction 0.17.0, num-rational 0.4.2 and
num-bigint 0.4.8 sources expose the original finite-float rational conversion and
limb operations through the same ordinary/funded workers. These portable members
inherit the unsafe-code prohibition; safe checked conversions replace upstream
unsafe sites. All archives, file hashes and upstream licenses remain recorded.
The caller owns source/error custody and resource admission. Neither family
selects a backend or introduces a second parser/arithmetic engine.

See [`doc/bounded-followup-schema-compiler.md`](../doc/bounded-followup-schema-compiler.md)
for exact refusal, pristine-source comparison and performance evidence, plus the
remaining optional arbitrary-precision and dynamic-source boundaries.

## Shared ordered collection mechanism

The local JSON default ordered map now consumes `eredu-collections`' single
safe AVL worker. That workspace foundation contains the local node implementation
and its generic behavior/refusal tests; JSON serialization tests remain in this
fork. The JSON allocation adapter still calls its original policy before each
exact node allocation. Archive provenance and upstream licenses are unchanged;
no upstream file is claimed as the provenance of the local AVL implementation.
