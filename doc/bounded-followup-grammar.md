# Cold grammar declaration compilation

## Current construction path

`ConstraintCompiler::from_original_tokenizer` derives grammar metadata from the
retained tokenizer source and validates its configured EOS mappings. Empty EOS
configuration uses `INVALID_TOKEN` and an empty EOS list; token zero is never
silently reclassified. The original trie and slicer use that source's pool.
Prefix-normalization support is described in
[Tokenizer prefix sources](bounded-followup-prefix-sources.md).

Declaration preparation invokes llguidance's immutable grammar compiler directly
with a borrowed `TokTrie` and explicit `ParserAllocationFunding`. It no longer
constructs a temporary mutable matcher, `ParserFactory` or serialized-tokenizer
substitute. The completed `CGrammar` moves into a paid shared declaration shell,
retains its original compiler account, and is authenticated with the same recipe
at startup. The frozen-JSON/factory runtime and alternate allocating controller
have been removed; see [Canonical constraint controller](bounded-followup-constraint-controller.md).

Ordinary and enforced compilation use the same dependency workers. The funding
policy chooses enforcement, not a parser, collection algorithm or regex engine.
The original producers now pay their reached heap growth:

- regex-syntax AST/HIR, Unicode/class expansion, scratch and diagnostics;
- derivre expression/hash-cons arenas, simplification and prefix scratch,
  Unicode/JSON-quote caches, derivative/relevance tables, alphabets and transitions;
- Lark lexical sources, token payloads, syntax nodes, literal decoding, names,
  conditions, nested grammars and suffix-automaton substring construction;
- JSON owned values, ordered IndexMap/IndexSet tables, schema normalization,
  references, numeric grammar production and pattern-property helpers; and
- grammar symbols, rules, nullable conditions, repetition/shortcut tables,
  lexer names/classes and compiled destinations.

The pinned hashbrown/IndexMap workers report actual reserve layouts. IndexMap
pays its index and entry allocations in their ordinary order, without a hidden
smaller-allocation retry. The owned JSON sink shares serde's stream-deserializer
worker and preserves duplicate-key/property order. Default and no-default
llguidance builds use the same funded referencing 0.52.1 registry; the alternate
`context_simple` resolver and unused JSON merge engine are gone.

`ParserError` carries derivre/llguidance compilation failures. Resource refusals
stay inline; semantic text and typed cause boxes are paid before construction.
Location/context augmentation preserves the first typed cause, does not format
after a resource refusal, and captures no compilation backtrace. Completed
declarations, failed prefixes and escaped error/source aliases retain the
original account through payload retirement.

The mutable parser's fixed-frame lifetime is a separate completed correction.
Its private operation-scoped workspace pays the actual nested live-frame peak,
reuses slots for sequential visits and does not refund the account. Heap growth,
source copies and escaped errors keep separate charges. The exact released
five-mask measurements and reached-copy refusal coverage are in
[Prepared parser frame lifetime](bounded-followup-parser-frames.md).

## Cold inspection and emission frame scopes

The previously identified recursive inspection and compiler-emission gaps now
use the dependency's shared borrowed `Scope`/`Frame` mechanism. The scope pays
its own bookkeeping before construction, holds every parent frame until its
children return, and grows only for the reached live peak. Sequential siblings
reuse that paid peak; heap growth remains cumulative. A frame cannot outlive its
scope or original payer, and a recorded allocation refusal is preserved even if
the next frame would fit existing capacity.

`PendingGrammarDeclaration::from_compiled` passes its actual compiler funding to
both retained-capacity inspection and source-copy planning. `CGrammar`, nullable
`ParamCond` trees, lexer declarations and recursive `RegexAst` sources fund their
inspection before descending. The compiled copy plan borrows its exact lexer
plan; child inspections repeated by the finite destination constructor are part
of that constructor's explicit quote. Startup reads the stored copy receipt and
shared immutable grammar, without another unpaid recursive inspection. Small
condition errors keep the recursive stack independent of the larger outer lexer
error while preserving that error through the compiled-grammar failure type.

JSON schema normalization, intersections, disjointness, references and recursive
schema copies now retain their concrete branch frames. The subsequent JSON
emitter has its own actual operation scope: nested array/object/union emission,
field-count `ordered_sequence` recursion and AST nonempty inspection share it.
Validated, no-validation and `x-guidance` entry points join the same emitter.
Lark rule, token, expansion and nested-grammar emission likewise uses one scope;
nested Lark compilers borrow it while the parent remains live. A nested JSON
compiler pays its own scope alongside the live Lark parent. These changes retain
the existing algorithms, grammar semantics and destination-copy authority.

Source errors preserve the first typed callback refusal. Completed declarations
and escaping failures keep their original payer after local scopes retire. The
source inspection, emission and mutable-parser scopes describe different actual
lifetimes; a later copy quote or a successful released generation is not used to
pay an earlier traversal.

## Optional dependency operations

The facade calls the compiler with `Logger::new(0, 0)` and selects llguidance
features `ahash,lark` with default features disabled. Two dependency operations
therefore remain outside that built-in public profile:

- Nonzero grammar logging still calls allocating pretty printers such as
  `grammar.to_string(...)`. A caller enabling those logs needs producer funding;
  zero logging does not exercise or qualify them.
- The optional llguidance `jsonschema_validation` module still initializes an
  ordinary lazy `Validator` and builds `anyhow` diagnostics. The underlying
  jsonschema compiler now offers a funded build/diagnostic API, but this optional
  llguidance adapter has not adopted it. This is distinct from the funded schema
  validation already used by facade tool arguments.

Opaque reference retrieval remains separately qualified. These optional limits
must not be described as missing default grammar execution, and default success
must not be used as evidence that their allocation paths are paid.

## Evidence and reproduction

Construction milestones passed 70 llguidance tests after grammar/derivre/Lark
conversion, then 71 after paid owned-JSON lowering and 74 after diagnostic,
reference and declaration-custody integration. The JSON-schema source test refused
all 1,721 reached requests and retained failure custody; Lark and conditional
construction tests also sweep reached allocations. Derivre's 60 library tests,
14 basic cases, 10 language-emptiness cases, Fowler corpus and eight substring
cases passed at the recorded source-construction checkpoint. The owned JSON sink
passed five focused tests under each map profile.

The later mutable-frame checkpoint passed the 77 preexisting llguidance tests,
two new workspace tests and 47 facade constraint tests (two external-tokenizer
probes ignored there). Its copied-parser test refuses all 189 reached mask
requests while the historical source rejects new debits. Current public/unit
suite counts are maintained in
[Ordinary facade consolidation](bounded-followup-ordinary-frontdoors.md), and
released device results are separate from dependency allocation evidence.

Commands use Rust 1.98.0, `CARGO_INCREMENTAL=0` and `--offline`; host parser tests
use `/private/tmp/eredu-grammar-check`:

```sh
cargo test --offline -p llguidance --lib
cargo test --offline -p eredu --no-default-features --lib scoped_mask_refusals_keep_exact_destination_payer_and_failed_prefix
```

The final JSON-emitter checkpoint passes nine tests (including four new frame,
public-entry, semantic and custody regressions), with a 4.84-second build and
0.02-second run in `/private/tmp/tokenizer-json-emitter-tests4.log`. Its 48-field
optional object case reaches recursion independently of schema nesting and
checks early, middle and final refusals. Actual parser acceptance for nested
arrays, objects and unions agrees across ordinary/funded compilation and all
three public JSON entry points. Four schema frame/copy tests and the compiled
source-inspection refusal test also pass against that same binary:
`/private/tmp/tokenizer-final-schema-frame-tests.log` and
`/private/tmp/tokenizer-final-compiled-inspection-tests.log`. The latter uses
256 nested conditions on the default test stack.

The joint final frame run passes all ten JSON-emitter, schema-copy/intersection
and Lark-emitter tests in 2.38 seconds after a 4.41-second build:
`/private/tmp/tokenizer-final-cold-emitter-frame-tests.log`. Lark covers actual
language acceptance for recursive rules, tokens and nested grammars, plus every
reached emitter-allocation refusal and final payer retirement.

After the independent frame-census review and exact diagnostic-capture additions,
the complete coherent llguidance library suite passes **90 tests**, with no
failures, ignored cases or filters. The final build took 4.81 seconds and the run
48.15 seconds, including the exhaustive original construction-refusal sweeps:
`/private/tmp/tokenizer-final-cold-llguidance-all.log`.

Reproduce these focused emitter checks with the facade's actual dependency
feature profile:

```sh
cargo test --offline -p llguidance --no-default-features --features ahash,lark --lib json::compiler::
cargo test --offline -p llguidance --no-default-features --features ahash,lark --lib json::schema::frames::tests::
cargo test --offline -p llguidance --no-default-features --features ahash,lark --lib compiled_source_inspection
cargo test --offline -p llguidance --no-default-features --features ahash,lark --lib ::frames::tests::
cargo test --offline -p llguidance --no-default-features --features ahash,lark --lib
```

Specific mutable-frame logs and the unchanged upstream archive/license hashes
remain in [Prepared parser frame lifetime](bounded-followup-parser-frames.md)
and `third-party/parser-upstream.json`. The focused tests establish the named
producer, semantic and custody properties; optional operations listed above
require their own funding integration.
