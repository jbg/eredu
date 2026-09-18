# Schema compiler allocation inventory

## Current scope

The original compiler, selected regex engines, default numeric/format producers,
source census and invocation caches now share prospectively funded workers. The
chronological records below preserve their intermediate inventories and evidence;
statements about then-pending regex construction, default float conversion, IDNA
or owned JSON parsing are superseded by the later completed sections.

Arbitrary-precision JSON **parsing and retained Number transport are implemented**;
they no longer have a global source-profile refusal. The remaining optional
numeric limits are narrower and still explicit in current source: funded schema
integer classification/equality and arbitrary-precision `multipleOf` operations,
plus `serde_json::bounded_number::Plan` used by the static chat numeric-template
producer. The separately typed F64 plan has its own supported representation.
These limits do not describe the default JSON number profile or the completed
source/event parser, and no consumer substitutes an approximate f64 value.

The optional llguidance meta-validation adapter has not yet selected the funded jsonschema
API; its current scope is documented separately in
[Cold grammar declaration compilation](bounded-followup-grammar.md). Released
model success does not establish support for these optional profiles.

## Historical producer inventory

The existing `ValidationOptions::build` uses the ordinary compiler: draft and
meta-schema resolution, a referencing registry, a shared compilation context,
keyword factories, pattern compilation and immutable schema-node assembly.
`OriginalValidationSource` currently inspects already-built immutable storage and
funds validation scratch. That inspection does not authorize any preceding cold
compiler allocation.

The canonical implementation will keep this compiler and its keyword workers.
`build_with_funding` will pass an explicit retained compilation callback through
those same workers. The resulting `Validator` and typed compilation failure will
retain that callback; ordinary `build` will select the unenforced policy. No
retained-size census will stand in for prospective allocation checks, and a
refusal will stop the current producer before a fallback or another allocation.

The concrete producers to fund are:

- referencing registry resources, URI resolution, anchors and resolver state;
- context cache tables and shared shells, pending schema nodes and pattern caches;
- locations, keyword rows, property names, numeric/schema copies and diagnostics;
- pattern translation and the existing regular-expression compiler;
- concrete keyword validator boxes, collection capacities and immutable node
  control blocks; and
- successful validator custody and failure custody through graph retirement.

Referencing owns its own registry/URI producer checks. JSONschema adapts the same
retained callback to that dependency-neutral contract. Opaque extension callbacks
must expose their actual construction contract before a funded build invokes
them; the compiler cannot infer allocation bounds from a callback type or from
its eventual output.

This inventory precedes implementation. The JSONschema slice owns its compiler,
keyword and path producers; facade tool declaration parsing, retained source
headers and public error composition remain facade responsibilities.

The baseline default JSON object backing was the standard library's B-tree.
Its stable safe API exposes neither the next insertion layout nor node occupancy.
Charging a guessed split envelope would not describe the actual reached
producer. The source construction work therefore includes one canonical safe
ordered AVL tree with explicitly owned boxed nodes and a prospective reservation
before each actual new node. The
ordinary and funded map operations must use that same tree and preserve ordered
Map/Entry and iterator behavior. Prefix parsing and Value construction continue
through the existing JSON deserializer; JSONschema does not add a parser.

## Implemented producer boundary

The compiler now carries the retained source through registry construction,
URI normalization, context caches, immutable node shells and validator custody.
The same ordinary keyword workers reserve concrete validator boxes, vectors,
property keys, literals, annotations and selected diagnostic shells. Literal and
JSON tree copying use the owning dependency's prospective workers. Refused
reference storage travels as a fixed allocation marker until the original
compiler source restores its typed cause; it is not converted into an allocating
schema diagnostic.

The default object map uses one safe AVL implementation for ordinary and funded
insertion. Each absent key reserves the exact node layout before construction;
replacement allocates no node. Iteration allocates no backing. Shared iteration
uses compact height-bounded node cursors. Mutable iteration uses disjoint-borrow
frontiers. A predicate panic during `retain` leaves a valid reusable tree; only
already visited values may have changed. The insertion-order profile reserves
through the pinned IndexMap's actual table and entry producer.

The sorted profile passes eleven focused tests, including five owned-value/prefix
parser tests, randomized map operations against the standard B-tree, alternating
iterator directions, exact refusal behavior, and nested clone/serialization on
the normal test stack. Existing upstream map tests pass in both profiles (three
sorted and four insertion-order tests). Release measurements of insertion,
lookup, traversal and removal versus the pinned upstream map's B-tree backing
were 30.81 ms versus 16.58 ms for 20,000 seven-entry maps and 129.13 ms versus
73.96 ms for four shuffled 50,000-entry maps. The explicit per-node allocation
has a measured cost; the large-map algorithm remains logarithmic per operation.
These timings precede the nonallocating retain panic-safety change and are not a
claim of whole compiler performance. Logs are `/private/tmp/native-serde-*-tests.log`,
`/private/tmp/native-serde-*-upstream-map.log`, and
`/private/tmp/native-serde-map-measurements.log`.

## Cold validation and public bridge

`Validator::build_with_funding(schema, policy)` constructs default options and
compiles through the shared worker. `ValidationOptions::build_with_funding`
borrows an already owned configuration. Both accept `None` as the explicit
unenforced policy and `Some(Arc<dyn CompilationFunding>)` as the original source.
The typed result retains schema, reference, JSON-source or allocation errors
without formatting them into strings. The graph and failure retire their owned
storage before their retained callback. This bridge is available for integration;
it is not a claim that every builtin producer is qualified yet.

Funded meta validation compiles exact bundled bytes through the same source
compiler into an independently paid graph. Drafts 4, 6 and 7 have fresh successful
compilation and invalid-schema diagnostic parity coverage. The diagnostic path
uses the ordinary validator dispatch, with the first storage refusal retained
before its result is interpreted. Property-name string nodes, combinator error
branches, required names, literal copies, paths and error shells use prospective
workers. No unpaid diagnostic retry follows a funded validation attempt.

The ordinary JSONschema unit suite passed 1,320 tests after the initial diagnostic
dispatch migration (`/tmp/native-schema-unit-tests-3.log`). One upstream test is
filtered because its external JSON-Schema-Test-Suite fixture is absent from the
published archive. Six focused source/refusal tests subsequently passed with an
expanded 19-case diagnostic matrix and fresh draft 4/6/7 meta compilation
(`/tmp/native-schema-funding-tests-5.log`). These are milestones, not evidence for
later untested edits or the remaining producers below.

## Intermediate cold compiler inventory

At this intermediate checkpoint the canonical selected fancy engine was the
pinned `fancy-regex` 0.19.0 used by tokenizer and schema consumers, and its dynamic
producer funding was still in progress. The completed engine work below
supersedes that status. The separately selected standard engine is `regex` 1.13.1.
`jsonschema-regex` 0.52.1 source translation has prospective hooks and independent
parity evidence; that alone does not qualify engine construction or search
scratch. Modern bundled meta graphs reach that pending constructor. No
payer-dependent engine substitution is permitted.

Other remaining producers include floating and arbitrary-precision number
conversion, allocation-bearing builtin format validation, remaining pattern
runtime/diagnostic paths, and optional generated/async compilation surfaces.
Opaque callbacks refuse before unqualified construction or execution. Every
reached cold failure must preserve its first refused allocation.

Diagnostic construction also reaches deferred evaluation paths. The previous
error clone copied an empty lazy cell, allowing each alias to allocate its own
joined path later. The canonical path now shares one immutable deferred owner.
Construction reserves the exact future string and shared-string layouts before
publishing that owner; its one initialization cell performs at most one joined
path construction. Aliases retain the same source and allowance. Instance paths
use the same stack-first segment collector with prospective deep-path storage.
The same diagnostic dispatch retains first-refusal custody and funds the reached
error payloads and collection storage. Remaining unqualified builtin producers
are still integration work; they are not model or schema-language limitations.

Cold seed audit found `ahash::RandomState::new/default` initializes boxed
process-global seed/source storage, including when reached through empty map
construction. Schema-owned maps and validation sources are being consolidated
onto one fresh seed worker: the existing host entropy provider fills an inline
buffer, then `RandomState::with_seeds` constructs the same keyed hasher without
adopting global allocated storage. Ordinary and funded preparation share this
worker. The retained graph census also needs prospective funding for its actual
visited-owner vector; the count it reports remains observational and never
substitutes for source-construction funding.

Email source inventory: `email_address` 0.2.9 validates borrowed local/domain
parts, then copies the successful complete input into its owned `EmailAddress`.
That one source copy must be admitted inside its existing parse worker. The
schema layer additionally owns empty-quoted-local-part rewriting and the IDN
mask string. Punycode character insertion and final UTF-8 construction remain
the original algorithm, now with reached vector/string reservations. No separate
email grammar or approximate hostname validator is introduced.

The public source bridge and funded source initialization/census passed eight
focused suites, including refusal at every reached census producer and shared
source retirement. The available JSONschema suite then passed 1,323 tests
(`/tmp/native-schema-unit-tests-4.log`). The default content descriptor lookup
now uses fixed function declarations instead of lazily allocated hash maps.
Legacy content invocation conversion remains separately unqualified pending its
producer work; modern annotation-only assertions preserve their ordinary empty
validation body.

Hostname/punycode and non-IDN email validation now use prospective source and
invocation loans in their ordinary workers. The exact email fork passed all 79
upstream unit tests, its reached-refusal integration test, and 14 doc tests
(`/tmp/native-email-tests-3.log`). The standalone
`doc/validation/email-source-compare.rs` oracle linked the unmodified archived
source from `/private/tmp/eredu-email-upstream-oracle/email_address-0.2.9` as
`email_address_reference` and the actual workspace email rlib as `email_address`.
All 6,144 address/option combinations matched exact success values and semantic
errors (`/tmp/native-email-reference.log`). The schema format suite passed all
302 tests after integration (`/tmp/native-schema-format-tests-1.log`). The IDNA dependency hooks and independent comparison are recorded below.

Legacy content validators now retain explicit builtin/custom source declarations.
All five builtin encodings use the original data-encoding `decode_len` /
`decode_mut` operations with an admitted destination; base16 preserves its
Unicode-uppercase retry. JSON content uses the shared owned JSON source producer.
Conversion scratch belongs to the invocation loan, while names/locations belong
to the compiled source. Invalid decoded UTF-8 retains its original owned bytes in
the diagnostic. Custom callbacks remain explicitly unqualified before paid
invocation. Nine funded suites now cover 29 diagnostic shapes plus every reached
content invocation refusal and callback non-entry; the available upstream schema
suite passed 1,325 cases (`/tmp/native-schema-funding-tests-8.log`,
`/tmp/native-schema-unit-tests-6.log`).

The optional arbitrary-precision JSON source remains separate pending work:
its ordinary parser scans into owned strings, transports its private numeric map
tag, reparses numeric strings and formats integer Number destinations. Reserving
only the final Number copy would not qualify these producers. Its ordinary
feature behavior is preserved; enforced content declares that numeric parser
unqualified before invocation. Source-profile refusals are not invalid-JSON
answers. Full-source event planning now lets the ordinary parser diagnose deep
nesting while reserving at most its actual recursion limit, as prefix planning
already does. The source-length profile still requires explicit admission.

IDNA source consolidation now covers schema `idn-hostname` / `idn-email` through
exact `idna` 1.1.0, `idna_adapter` 1.2.2 and `icu_normalizer` 2.3.0 forks. Their
compiled Unicode data remain immutable static tables. The original UTS 46,
punycode and normalization workers prospectively admit output strings, domain /
label / decoded-punycode SmallVec buffers and normalization scratch. First refusal
stops further reservations and wins over later semantic errors. Stable canonical
combining-class order uses in-place insertion sorting for short runs and a paid
stable counting pass for longer runs; all Unicode mappings are unchanged. Checked
scalar/UTF-8 conversions and slice offsets replace upstream unsafe operations.
Optional run-time data loading remains a separate source constructor requiring
its own retained owner. External output callbacks and sinks must provide their
own storage contract; the schema path uses closed builtin workers.

Validation of that source boundary:

- The local/pristine archive-linked oracle in
  `doc/validation/idna-source-compare.rs` passed 231,732 exact ASCII/Unicode output,
  validity-status and borrowed/owned-shape comparisons over 6,437 sources.
- Every reached IDNA allocation was refused independently, with no later callback;
  the refusal test and 15 upstream unit/unitbis cases pass. ICU's 35 UTF-8/UTF-16
  conformance cases pass (`/tmp/native-idna-specific-tests-1.log`,
  `/tmp/native-icu-tests-3.log`).
- The large upstream IDNA fixture has 12,790 passes and 14 failures in both the
  local and independently built pristine pinned sources. Those failures reflect
  fixture expectations for Unicode code points admitted by selected ICU 2.3;
  fixtures were not weakened (`/tmp/native-idna-tests-1.log`,
  `/tmp/native-idna-upstream-tests.log`).
- The actual IDNA-enabled schema binary passed all 1,360 available unit cases and
  all nine funding suites, including 32 diagnostic shapes. The archive-absent
  `validator::tests::validate_ref` fixture remains filtered
  (`/tmp/native-schema-idna-unit-tests-1.log`,
  `/tmp/native-schema-idna-funding-tests-1.log`).

Release performance is measured for the **ordinary, explicitly unenforced**
worker against pristine archives, not for a funded invocation. For 2,000 ToASCII
calls with STD3/check-hyphens/ignore-DNS-length, current total local / pristine
times are 30.875 / 22.875 microseconds for the 16-byte ASCII domain;
1.004 / 0.715 milliseconds for the 22-byte Unicode domain;
16.309 / 11.210 milliseconds for a 609-byte combining run; and
18.943 / 11.477 milliseconds for 428 bytes of many labels. The measured cost is
1.35–1.65 times pristine. Cached enforcement and inlined policy wrappers remove
repeated virtual policy checks, but first-failure propagation through the shared
normalization/IDNA worker remains measurable. This is recorded as a performance
regression, not numerical or throughput parity
(`/tmp/native-idna-reference-release-results-3.log`).

Default numeric consolidation uses the exact selected `fraction` 0.17.0,
`num-rational` 0.4.2 and `num-bigint` 0.4.8 sources. The existing decimal/i128 fast
path remains first. Its existing fraction fallback now carries an explicit
borrowed account through float scaling, original decimal formatting/parsing,
rational reduction/division, and actual limb copies, shifts, GCD, scalar/long
multiplication, and Knuth division. Ordinary operations delegate these same
workers with explicit unenforced policy. There is no replacement modulo or
rational approximation. Each reached backing/control reservation stops before
allocation on refusal, and invocation funding remains separate from the compiled
schema source. The float and integer diagnostics share the funded diagnostic
builder.

Finite f64 source magnitudes are at most 1,024 bits; the original decimal fallback
has at most 324 fractional places (1,077 denominator bits). Cross products from
one ratio division therefore remain below the original recursive large-number
division threshold. The multiplication operands before reduction remain within
the original long-multiplication threshold. Larger recursive integer algorithms
and arbitrary-precision JSON conversion remain explicitly unqualified for paid
calls pending their own propagation. Those are remaining optional integration
work, not a different arithmetic engine or inherent schema limitation.

The three numeric forks retain archive/file hashes and licenses, inherit the
workspace unsafe-code prohibition, and use safe versions of the original scalar
carry/divide and UTF-8 conversions. Same-type generic ownership conversion uses
an allocation-free checked `Any` downcast. The optional random generator preserves
its original u32 draw stream with safe digit assembly.

Numeric evidence:

- Fraction's 266 library tests, rational's 52 library tests, and bigint's 168
  library/integration cases pass (`/tmp/native-numeric-tests-2.log`,
  `/tmp/native-bigint-tests-1.log`).
- The fraction funding test refuses every reached operation over ten ordinary
  and extreme float/divisor pairs, preserving the first refusal and making no
  later call (`/tmp/native-fraction-allocation-tests-2.log`).
- The independent pristine-archive executable
  `doc/validation/numeric-source-compare.rs` passed 35,340 exact ordinary/paid
  fraction and quotient comparisons over 8,835 sources, including 8,192 sampled
  bit patterns and exponent boundaries. It reached 3,144,273 reservations
  (`/tmp/native-numeric-source-reference-1.log`).
- The integrated IDNA-enabled schema binary passes all 1,361 available unit cases
  and ten funding suites, now covering 33 diagnostic shapes. The unavailable
  archive fixture remains filtered (`/tmp/native-schema-numeric-unit-tests-2.log`,
  `/tmp/native-schema-numeric-funding-tests-2.log`).

The numeric release probe measures 2,000 **ordinary/unenforced** conversions plus
divisions against pristine archives. Decimal `1070468.14 / 0.01` took 4.805 /
4.058 ms, large `1e300 / 0.7` took 3.273 / 2.487 ms, and tiny
`1e-200 / 0.123456789` took 0.879 / 0.701 ms. The current 1.18–1.32 times ordinary
cost is a recorded performance regression alongside the IDNA results, not a
claim of throughput parity.

Both current serde map profiles also pass all five shared source/prefix/owned
Value tests after the full-source depth census change
(`/tmp/native-serde-depth-sorted.log`, `/tmp/native-serde-depth-insertion.log`).
The exact dependency source hashes for this default boundary are recorded in
`doc/validation/schema-default-producers-2026-09-18.json`.

Historical arbitrary-precision producer inventory (source/event construction is
completed in the next section; arithmetic/predicate limits remain): serde's
ordinary numeric scanner owns a String, preserves normalized exponent spelling,
transports it through its private numeric map, and reparses/copies it into Number.
Small scalar Number constructors format their owned string too. The planned
shared worker must fund those actual producers and preserve the full Number
identity in events and downstream retained JSON records. Enabling only a final
Number copy or coercing a huge number to f64 would not qualify the profile.
Runtime source/event integration and the larger original big-integer algorithms
remain explicit pending work; the current typed profile refusal remains until
these producers are connected.

### Exact optional numeric source and retained carrier

The ordinary arbitrary-precision scanner, Number constructors and private numeric
object transport now use prospective String growth and the original numeric
parser. The event protocol borrows the actual immutable Number; runtime trees
retain paid Number values in an indexed numeric buffer rather than converting
them to f64. Capture readers apply the original Number primitive conversions.
The facade borrows that same value for schema numeric operations. `Plan::parse`
and `parse_prefix` require the caller's actual allocation policy in addition to
their prepaid static scratch/control requirements. A consumer's terminal storage
failure stops further parsing and preserves its original cause and paid prefix.

The source parser no longer rejects arbitrary precision globally. JSON content
validation uses that same event/owned-Value worker under both policies. Numeric
comparison/equality and large-integer arithmetic qualification remain separate
work; their precise existing refusals have not been removed. The scalar template
numeric plan also retains its explicit feature refusal pending its static
source-storage contract. None of these remaining consumers may infer an f64
approximation or switch to an unenforced parser.

Fresh direct serde checks pass five default and six arbitrary-precision suites,
including every reached refusal for valid and invalid private numeric transports.
The runtime numeric carrier's default-profile refusal/retirement test passes.
The optional-profile runtime check is recorded separately when complete.
`doc/validation/json-number-source-compare.rs` compares local source and event
results against the pristine 1.0.151 archive after verifying all archive member
hashes in `third-party/serde-json-upstream.json`: 3,016 sources and 15,080 exact
comparisons pass in **each** profile, including original diagnostics, spelling,
primitive conversions, nested numbers and the private object representation.

Representative ordinary parsing timings (20,000 iterations, minimum of seven
release measurements, nanoseconds total; no funding callback supplied):

| Profile/source | Local | Pristine | Ratio |
| --- | ---: | ---: | ---: |
| default `42` | 131,375 | 111,917 | 1.174 |
| default 30-digit decimal | 5,484,000 | 5,446,583 | 1.007 |
| default four-element array | 1,635,166 | 1,623,042 | 1.007 |
| arbitrary precision `42` | 1,314,792 | 1,020,209 | 1.289 |
| arbitrary precision 30-digit decimal | 4,076,333 | 2,492,292 | 1.636 |
| arbitrary precision four-element array | 5,610,250 | 4,007,750 | 1.400 |

The optional representation has measurable source-construction overhead; this is
not throughput parity. Native inference and tokenizer search throughput are not
measured by these JSON source microbenchmarks. Logs are
`/tmp/native-serde-numeric-reference-{default,arbitrary}.log` and
`/tmp/native-serde-events-{default,arbitrary}-tests.log`.

### Regex consumer integration inventory

The canonical fancy-regex constructor now has actual prospective storage hooks,
and scoped search owns invocation caches. Schema still has a constructor refusal
and generic regex bodies lack source/operation declarations. The chosen worker
is the existing selected regex engine with explicit constructor and invocation
funding. Ordinary and enforced schema matching will use scoped search through
one `RegexEngine` contract; the old direct persistent-pool call will be removed
from that consumer. Pattern, patternProperties, additionalProperties and
unevaluatedProperties must pass the same validation context. Cold precomputed
property-pattern indices must fund the existing match and map/vector producers.
The standard regex engine remains a distinct explicit engine choice; its outer
constructor still needs actual producer hooks and must not silently choose the
fancy engine. Retained regex source census needs the owning dependency's exact
shared-allocation traversal, not a constructor sum or a post-hoc charge.

The remaining regex source census follows actual allocation owners. Prefilter
trait-object Arcs, Aho automata, packed-pattern Arcs and Teddy/Rabin–Karp backing
must report their real Vec capacities and shared shells. Existing `memory_usage`
queries use logical lengths, omit some shells and double-count shared pattern
storage; they are not the canonical retained census. The replacement is a
nonallocating borrowed visitor with caller-owned identity deduplication. Search
algorithms and prospective constructor hooks remain unchanged.

### Compiled regex source and invocation closure (2026-09-18)

The default fancy-regex engine now uses its actual funded constructor and scoped
search workspace from the same selected automata/VM workers. Property-pattern
precomputation uses those same operations and prospectively grows its actual
hash table, property text and matched-index vectors. Literal optimizations retain
their existing selection rules. Pattern, pattern-property, additional-property
and unevaluated-property visitors describe their actual graph and invocation
storage. Diagnostics retain the same property order and original typed refusal.

A borrowed source-allocation visitor now traverses fancy/automata graphs and
prefilters through actual shared identities. Prefilters include Arc shells,
Aho DFA/NFA backing capacities, packed pattern storage, bucket vectors and owned
memmem needles. Repeated aliases skip their children; source inspection never
uses cumulative construction reservations as retained size. Ordinary persistent
regex pools that have been used reject immutable source inspection explicitly.
Source construction uses invocation-owned workspaces under both funding policies.
Prepared validation retains and reuses these caches for the complete invocation,
then drops them before its funding loan. Ordinary validation keeps its existing
shared-pool policy. Both policies enter the same engine and search workers;
funding does not select an engine. Warmed ordinary sources reject immutable
source inspection explicitly.

Focused evidence: paid meta validation passes Draft 4, 6, 7, 2019-09 and 2020-12.
Six representative regex/property graphs pass every reached construction,
source-inspection and invocation refusal, preserve first refusal, and spend no
source funds during invocation. Aho tests verify all three automaton forms,
shared alias deduplication, stability across searches and actual spare vector
capacity. The initial complete schema sweep passed 1,326 tests. The subsequent
oversized-regex diagnostic and standard-wrapper work is recorded below; the
compiler never changes engines based on funding policy. Optional arbitrary-precision arithmetic/predicate and static
numeric-template planning gaps listed above also remain distinct.


The per-match cache prototype exposed a real performance regression. On the
pinned local release engine, 20,000 calls (minimum of seven trials) took 72.16 ms
versus 0.427 ms for an anchored 16-byte identifier, 2.20 ms versus 0.316 ms for
23-byte Unicode text, and 12.23 ms versus 2.62 ms for an 8-byte backreference
input, comparing a newly constructed scoped cache with a warmed shared pool.
This was a same-engine cache-policy comparison, not an upstream throughput
claim. The implementation now reuses an exact-owner cache within each prepared
validation context and preserves ordinary shared-pool reuse. A fresh measurement
of warmed invocation-owned versus warmed pooled searches gave 0.431/0.442 ms,
0.327/0.344 ms and 3.014/2.817 ms respectively (ratios 0.975, 0.949 and 1.070).
The reproduction source is `doc/validation/schema-regex-scoped-compare.rs`;
constructor cost still belongs to the first match in each funded invocation.
No cache survives that invocation's payer, and immutable source storage does not
adopt its allocations. Pattern compiler rows now retain their existing shared
engine owner instead of cloning the compiled pattern and its persistent pool.


Final post-correction library validation passed **1,328 tests**, with only
`validator::tests::validate_ref` filtered because its external fixture is absent.
The oversized delegated-regex failure now retains its actual typed engine cause
while preserving the existing schema diagnostic. The exact-owner cache test
checks reuse, rejection of equal-text/different-owner substitution and retirement
with invocation scratch. Source-refusal sweeps allow an index that a fresh
entropy-seeded compilation never reaches; every reached refusal still stops at
the original typed error. Twelve independent source-sweep runs also passed.
Evidence and source hashes are in
`doc/validation/schema-regex-owned-cache-2026-09-18.json`.

The complete schema benchmark retains a real cost for funded invocation scope.
For 2,000 validations of objects with 1, 16 and 128 patterned properties, ordinary
pooled validation took 0.121, 1.646 and 12.787 ms; separate funded invocations took
8.976, 11.500 and 27.355 ms (74.256×, 6.984× and 2.139×). Construction and census
were outside these measurements. These are current same-code policy comparisons,
not pristine-engine regressions. Each funded invocation pays its initial cache
construction and accounting; caches are reused within it and never outlive its
payer. `doc/validation/schema-regex-invocation-compare.rs` reproduces this scope.

### Standard regex wrapper inventory before edits

The selected `regex` 1.13.1 wrapper still copies pattern input in its common
`Builder::new`, creates retained pattern Arc storage in its four shared build
workers and formats lower-engine construction errors. Its underlying canonical
regex-automata workers already provide prospective source, cache and exact
retained-census hooks. The wrapper must propagate the same neutral policy into
those workers and its own pattern/error producers; it must not reconstruct an
alternate matcher in the schema crate. Ordinary constructors remain adapters to
the same workers with explicit unenforced policy. The schema will retain each
original shared regex owner, use the same explicit shared-pool versus invocation
cache policy, and inspect actual source allocations. This thin wrapper closure
is separate from optional arbitrary-precision predicates/arithmetic and the
static numeric-template source planner, which remain pending.


### Standard wrapper qualification and source identity correction

The pinned `regex` 1.13.1 wrapper now admits its original pattern-copy, shared
pattern owner and error-formatting destinations and passes the borrowed policy
into the canonical automata constructors. All four string/byte, single/set
builders use those same workers under ordinary and enforced policies. The
schema's explicitly selected standard engine retains its actual shared source
and uses the same invocation cache worker as fancy-regex. Its scoped boolean
operation calls the original earliest half-match search; cache policy does not
change the selected engine or search operation. Original engine failures remain
typed in the schema diagnostic chain.

The archive's optional nightly `Pattern` integration is excluded from this safe
stable fork's compiled API. Its source and archive provenance remain retained;
no unsafe exception was added. The published archive omits generated fuzz tests
and the Fowler fixture directory, so its integration harness runs the supplied
fixtures only. This is a test-data limitation, not a production-profile refusal.

Exhaustive source census exposed a real identity defect in the earlier visitor:
an empty `Arc<str>` data pointer lies one past its allocation and can equal the
base of an adjacent string buffer. Canonical Arc visitors now identify the
actual allocation base, using the same checked header/alignment layout already
used for its byte count and safe pointer arithmetic. Aliases still deduplicate;
bytes and admission costs are unchanged. The regression directly checks empty
shared-slice identity, and twelve complete schema source/invocation refusal
sweeps passed after correction.

Current validation: **1,329 schema library tests pass**, with only the absent
external `validator::tests::validate_ref` fixture filtered. The wrapper's four
source/refusal/diagnostic/census/cache tests pass under default and no-default
features; its supplied upstream suite passes 56 tests with one upstream ignored
test. Stable all-features compilation passes. Aho's two census tests and the
empty-Arc regression also pass. Both configured schema engines exercise reached
source and invocation refusal, exact source retention and immutable census.

An independent registry reference, including pristine lower dependencies,
passes **36,131 exact comparisons across 316 patterns**: Unicode and byte spans,
captures, replacements, sets, diagnostics and scoped booleans. The reproduction
source is `doc/validation/standard-regex-compare.rs`. It links local `current`
and pristine `reference` aliases of regex 1.13.1, plus an explicit local
`current_automata` alias. Reference lower versions are regex-automata 0.4.18,
regex-syntax 0.8.11, aho-corasick 1.1.5 and memchr 2.8.3, resolved outside the
workspace patch table. The archive checksum and current source hashes are in
`doc/validation/standard-regex-source-2026-09-18.json`.

Ordinary release performance remains slower than pristine on these short
searches. For 20,000 calls, minimum of seven trials, the final identifier
(16 bytes), Unicode (23 bytes), and literal (18 bytes) cases took respectively
0.458/0.361 ms, 0.327/0.276 ms and 0.274/0.213 ms, local versus pristine
(ratios 1.268, 1.185 and 1.290). Direct calls to the same local meta search took
0.456, 0.349 and 0.257 ms; the new wrapper does not add another search mechanism.
For 200 constructors the corresponding local/pristine times were
3.335/2.668 ms, 9.879/9.500 ms and 4.461/4.116 ms (1.250, 1.040 and 1.084).
An earlier trial's search ratios were 1.155, 1.354 and 1.162. These results expose
remaining shared-engine accounting overhead and normal short-trial variability;
they are not presented as zero-cost enforcement or a performance improvement.

The complete schema invocation benchmark was also refreshed after the common
cache worker change. For 2,000 validations with 1, 16 and 128 patterned
properties, ordinary pooled validation took 0.122, 1.833 and 14.783 ms, while
separate funded invocations took 8.734, 11.150 and 26.025 ms (71.540×, 6.083× and
1.761×). Source construction/census were outside the timed interval. This
supersedes the earlier same-code invocation numbers above; the first-cache and
per-invocation accounting costs remain real. Optional arbitrary-precision
predicates/arithmetic and static numeric-template source planning remain open.
