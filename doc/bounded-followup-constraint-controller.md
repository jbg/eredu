# Canonical constraint controller

The facade no longer constructs the old `GrammarState` or the independent
`ConstraintRuntime::{Forbidden, Auto, Active}` states. The unregistered
`from_generation_plan` constructors, vocabulary repacking owner, and late
forbidden-source copy constructors were removed with those states.

`ConstraintController` now contains either the plain controller, an original
forbidden source with its paid history, or the original paid grammar controller.
The forbidden variant requires the actual original source; it cannot contain an
unwitnessed input copy. Current and provisional filters call the same core
forbidden decision worker. Grammar activation, token commitment, EOS aliases,
completion and parser copies use the original grammar worker. Ordinary sampler
callbacks use these same operations, including independently funded successors
when immutable history is shared.

The model-independent constructor takes the actual compilation receipt, source
validity and funding. Its backend callback supplies only the original forbidden
tokenizer source. Production startup and neutral tests use that constructor;
the test fixture does not implement a second selector or parser.

Tests that compare grammar language behavior now use a direct dependency
`Matcher` as the independent oracle. Runtime behavior tests compile a real
retained tokenizer, template/profile, controller receipt and semantic binding
under a finite 1 GiB fixture capacity. Explicit activation boundaries, token
masks, EOS aliases, failed-commit atomicity, independent copies and escaped
source/error custody remain covered. Lower forbidden-source tests additionally
exercise empty, binary and multibyte token entries and distinct equal-byte
source identities.

The cold inspection compiler still accepts an ordinary tokenizer environment.
That inspection capability is distinct from session startup: it no longer has
a facade runtime-state reconstruction path. This cleanup does not claim that
all inspection producers have moved to the source-owned compiler.

The production source path requires its retained original trie, declaration,
compilation receipt and matching slicer. The serialized tokenizer reconstruction
fallback is gone. One semantic fallback is intentional: a non-storage grammar
failure or grammar warning may select the syntax-only tool grammar, while the
already compiled full-schema validator still checks complete calls. Names,
protocol and call cardinality remain constrained. The sticky first funding
refusal and typed allocation/reference transport failures cannot select that
fallback. This preserves the existing grammar/schema division without an
unenforced execution retry.

A related observed conformance failure exposed a neutral error-chain defect:
`FundedCaptureError::Backend` used transparent error forwarding, which skipped
a backend leaf with no nested cause. It now exposes that same leaf as its
source, preserving display text and storage layout. A focused test follows the
leaf through `BackendFailure` and checks final retirement; no diagnostic copy
or additional error allocation was introduced.

At the controller-consolidation checkpoint, focused portable validation passed
46 constraint tests and 26 sampler tests. The subsequent scoped-frame checkpoint
passed 47 constraint tests; its additional refusal and custody evidence is in
[parser frame lifetime](bounded-followup-parser-frames.md).
The neutral capture error-chain regression also passed, and the real observed
conformance case reached the exact injected backend leaf. The recorded facade
unit binary passed 372 distinct tests with four ignored across complementary
runs: 371 passed with only the exhaustive schema-row sweep filtered out, and all
15 schema tests passed including that sweep. The schema authority fixture excludes
its funding-owner setup from invocation refusal cuts.

A repeated accepted-construction probe found 27,276–27,280 callbacks for the
same three schema rows across 64 successful runs. The exhaustive refusal test
now distinguishes an unreached cut from an actual refusal: success requires the
trace to end at or before the cut, no recorded refusal, all three rows, and
unchanged source-retirement checks. Every reached refusal still must stop at
its first callback and retain the exact typed cause and payer. No production
schema fallback or compiler behavior changed. The final exhaustive rerun passed
in 341.60 seconds; the other facade library tests passed in 31.52 seconds. Logs:
`/private/tmp/tokenizer-schema-reached-refusal-final-tests.log` and
`/private/tmp/tokenizer-current-facade-library-complement.log`.

These portable checks establish controller behavior and custody. Later
[released text and image tool runs](bounded-followup-released-tools.md) pass
Required and Auto for the concrete sensor request under the unchanged 64 GiB
limit, with one complete `reading({"value":17})` call and `GrammarComplete`.
Those native runs retain their own checkpoint, executable and stack
qualifications. The separate generic-action Required request finishes its
48-token allowance without a funding refusal but emits prose and ends at
`MaxTokens`; it is still an unsuccessful tool request. Current CPU/distributed
acceptance belongs to the [native matrix](bounded-followup-native.md), not these
portable checks.

```sh
cargo test --offline -p eredu --no-default-features --lib runtime::chat::constraints -- --test-threads=2
cargo test --offline -p eredu --no-default-features --lib api::tests::sampler -- --test-threads=2
cargo test --offline -p eredu --no-default-features --lib runtime::chat::tool_schema:: -- --test-threads=2
```

## Historical mutable lexer census correction

The measurements below precede the subsequent [operation-scoped frame
workspace](bounded-followup-parser-frames.md), which is the current worker.

A released-tokenizer Required-tool run reached a funding refusal after the legal
prose prefix `I`. A bounded callback trace located the 14,131-byte request in
`PreparedLexer::run`: the mutable worker was charging constructor frames that
owned a complete lexer and an owning constructor error. The worker borrows the
already prepared lexer and returns a non-owning operation error. Its control
census now describes those actual borrowed frames. The regex-vector state
worker likewise counts its borrowed receiver rather than its retained owner.
Cold constructor controls, heap growth, failure custody and grammar semantics
are unchanged.

The pinned tokenizer has SHA-256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
At the same 200,000-callback limit, incremental first-mask funding decreases
from 1,784,743,777 to 588,655,777 bytes. This trace deliberately stops before
finishing the mask; it is a producer measurement, not evidence of complete
released generation. The actual source-copy request remains 794,322,049 bytes.

All 76 preexisting llguidance library tests pass. A new regression repeats
cached mutation under an 8 KiB operation allowance, compares ordinary lexical
results, checks unchanged transition population, and refuses each reached
callback in turn. The failed mutable owner makes no later funding requests.
The source-backed optional facade probe uses a finite source pool and callback
account and performs no model or device work.

```sh
cargo test --offline -p llguidance --lib
cargo test --offline -p eredu --no-default-features --lib released_required_grammar_funding -- --ignored --nocapture
```

The ignored probe requires `EREDU_RELEASED_TOKENIZER_JSON` pointing to the pinned
artifact outside the tracked tree. Diagnostic logs are
`/private/tmp/tokenizer-lexer-census-tests.log`,
`/private/tmp/tokenizer-lexer-census-regression.log`, and
`/private/tmp/tokenizer-released-grammar-probe3.log`.

The complete source-backed regression at that checkpoint passed through the
canonical controller: startup, a full first mask allowing `I`, commitment, a nonterminal
completion check, and the second mask rejecting EOS. It uses a 2 GiB source
pool plus a separate finite 32 GiB operation account capped at five million
callbacks. The measured sequence consumes 13,223,435,767 charged bytes across
4,079,607 callbacks in 18.52 seconds. No model execution was involved. The later
scoped-frame worker supersedes these cumulative frame charges; the released
request outcomes are recorded above and in the linked native validation.

```sh
cargo test --offline -p eredu --no-default-features --lib released_required_prose_prefix_under_finite_funding -- --ignored --nocapture
```

Log: `/private/tmp/tokenizer-released-grammar-prefix-regression2.log`.
