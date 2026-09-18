# Schema regular-expression source producers

Inventory before implementation (2026-09-18). Current selected
`jsonschema-regex` 0.52.1 is a registry dependency. Its ECMA translation parses
an AST, rewrites a `Cow<str>`, and traverses with a heap visitor. Its literal
optimization and witness producers allocate strings/vectors; its ECMA syntax
checker owns group/name vectors and decoded capture-name strings. These are
reached from the ordinary JSON-schema compiler before a compiled matcher exists.
The already funded tokenizer adapters do not cover this dependency.

Canonical implementation: retain this exact selected source/version and add
borrowed prospective allocation hooks through the same ordinary workers. Reuse
`regex-syntax`'s neutral allocation interface and its paid AST/HIR parser and
visitor stack. Ordinary entry points use explicit `Unenforced`. Fixed semantic
and allocation failures remain distinguishable; callers retain their original
funding owner and recover the original refusal. Preserve upstream licensing,
archive hashes, syntax/translation semantics, and optimization decisions.

At inventory time, schema validation selected registry fancy-regex 0.19.0
and regex 1.13.1 while the tokenizer fork used fancy-regex 0.17.0. The audit
and canonical 0.19 migration below resolve this version split. Prospective
dynamic compiler/runtime allocation hooks remain necessary; accounting alone
must not select a different matching algorithm.

## ECMA source implementation and validation

The exact selected 0.52.1 archive is now a portable workspace member with the
workspace unsafe-code prohibition. Archive SHA-256:
`f5d90ea83fa606c96f0b4737ecedf1fa6b624272022edc42039565f8d8af0b78`.
`third-party/parser-upstream.json` retains all six original file hashes and VCS
revision `94546ceb734c6076e73c4a6723de98804ad63ae6`. The archive declares MIT but
omits a license file; the same project's MIT notice is included from the
existing 0.52.1 value fork.

Public `*_with_allocations` entry points cover translation, syntax validation,
literal analysis, prefix extraction, and witness generation. They borrow the
existing `regex_syntax::allocation::Allocation` contract. Translation returns a
fixed `TranslationError::{Syntax, Allocation}`; other operations preserve their
existing semantic result inside `Result<_, AllocationError>`. The caller keeps
its actual funding owner alive through outputs and errors. Ordinary APIs invoke
these same workers with `Unenforced`.

The original AST/HIR parsers and heap visitor are prospectively funded. String
rewrites pay the complete new destination before replacement; syntax group and
name tables pay growth and owned decoded-name copies; optimization alternatives
and witness strings use the same prospective producers. Bell-escape rejection
now returns its fixed semantic cause directly instead of parsing a deliberately
invalid pattern to manufacture an error. No engine or optimization is selected
by whether an account is enforced.

Validation:

- **241 library tests passed**: all 237 upstream tests plus four refusal suites.
  `cargo test -p jsonschema-regex --lib --offline` with the real compiler.
- **10,645 exact independent comparisons passed** over 2,129 patterns, covering
  translation text and borrowed/owned result shape, ECMA validity, optimization
  variants, witness strings, and prefix results. The validation script verifies
  all six pristine archive file hashes before linking that original translator
  beside the modified implementation. Both consume the same previously
  independently validated syntax dependency, isolating this fork's behavior.
- Refusal tests stop at every reached allocation, including invalid inputs and
  nested decoded names. Representative translation `([\d\w\s]+|\cA){2}` reaches
  129 requests / 41,108 cumulative bytes; a nested named-group syntax example
  reaches 14 / 802; a four-alternative literal optimization reaches 16 / 204;
  the AST/HIR witness fixture reaches 189 / 31,466. These are cumulative
  reservations, not retained heap or process RSS.

Logs are `/private/tmp/eredu-ecma-final-tests.log`,
`/private/tmp/eredu-ecma-funding-tests.log`, and
`/private/tmp/eredu-ecma-reference.log`. Reproduce the independent comparison:

```sh
RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
bash third-party/jsonschema-regex-0.52.1/validation/run-reference.sh \
  /Users/jbg/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/jsonschema-regex-0.52.1 \
  /private/tmp/eredu-ecma-reference \
  /private/tmp/eredu-reference-validation/debug/deps \
  /private/tmp/eredu-reference-validation/debug/deps/libregex_syntax-80a102490ce0fe5b.rlib
```

The schema matcher engine, runtime scratch, enclosing compiler controls and
retained account lifetimes remain separate obligations. This source producer
slice alone is not a complete schema-compilation admission claim.

## Matcher version audit before migration

A downgrade to fancy-regex 0.17 is not semantically compatible with the selected
schema dependency. The 0.18/0.19 sources add subroutine calls, absent repeaters,
FAIL/general-newline constructs, byte/ranged inputs, and assertion overrides.
Version 0.19 also corrects Unicode backreference byte-window comparison,
inline-flag precedence, and `\G` handling. Its delegates compile directly from
HIR and its VM pools scratch for ordinary repeated searches. The 0.17 workspace
fork cannot stand in for those capabilities by changing a version requirement.

Canonical direction: port explicit workspace and capture-free DFA mechanisms
onto exact 0.19, preserving its one VM worker and current semantics; regenerate
the accepted tokenizer source recipes with that compiler; then migrate both
consumers and remove the older production fork. Explicit workspace versus a
shared scratch pool is a storage/execution ownership policy, not a payer-driven
algorithm choice. The newer VM's additional cut/subroutine/control storage must
have actual reached geometry or typed capability restrictions specific to that
resource; do not infer bounded general schema matching from the tokenizer's
closed, capture-free source recipes.

The later general dynamic compiler hooks also remain necessary: the existing
closed tokenizer source recipes do not fund arbitrary schema regex parsing,
analysis, HIR/delegate construction, or runtime scratch. This audit does not
claim those mechanisms complete.

## Canonical 0.19 implementation and evidence

Both production consumers now resolve the local 0.19.0 workspace member;
the older production fork is removed. Its archive provenance is retained.
The new archive SHA-256 is
`476de73bddf2ef8490aa4ee8f1cf40b430bf1d56c48c22080e5186952cd580e6`, revision
`e8d6986bccd57df06d6013221158cdb023f74d88`. All 58 original file hashes and
upstream licensing are retained in `parser-upstream.json` and the new fork.

One parsed-source helper owns ordinary parsing/optimization/analysis policy.
The original 0.19 VM loop and match-iterator progression each remain a single
worker used by both ordinary and explicit-workspace entry points. Generic
bytes/ranges, subroutines, absent repeaters, assertion options, Unicode
backreferences and current capture behavior are retained. The explicit closed
profile does not claim those additional allocating instruction families are
already prospectively funded.

The pinned compiler regenerated the five exact tokenizer recipes. Three
formerly delegated classes now use 0.19's original `CharClass` instruction;
15 distinct capture-free DFA sources remain. Every actual range vector and
DFA buffer is in the checked fresh source geometry. Exact-capacity range
vectors move into boxed slices without a shrinking allocation. Fixed-program
construction creates no ordinary scratch pool. Dynamic default and checked
construction select the same closed program for these accepted patterns.

Validation with the real Rust compiler and `CARGO_INCREMENTAL=0`:

- The staged complete upstream/library/integration suite passed **598 tests**,
  with the development recipe emitter ignored. After regenerated recipes,
  explicit-workspace tests passed **20**, source construction/partial failure
  tests **8**, and independent DFA/each-buffer refusal tests **2**. These counts
  overlap; they are not additional independent totals.
- Tokenizer library tests passed **282**, with **3** measurement tests ignored,
  using `fancy-regex,tokenizer-compiler-test-support` and the local syntax fork.
- `eredu-text --no-default-features` compiles. The engine compiles both with no
  default features and with `unicode,perf,variable-lookbehinds` without `std`.
- The independent comparison verifies all **58** pristine archive hashes,
  compiles that original 0.19 implementation beside this fork, and passes
  **15,507** matching, capture and diagnostic comparisons over **437** patterns.
  It includes all five closed tokenizer patterns with 530 multilingual/adverse
  inputs each. Both versions consume the same independently tested lower regex
  dependencies, isolating the fancy engine changes.
- Source geometry tests equal the planned heap to actual pattern/instruction/
  literal/class/table storage. Every reached class reserve and DFA buffer can
  refuse while retaining the complete constructed prefix.

Reproduce the independent comparison with
`third-party/fancy-regex-0.19.0/validation/run-reference.sh`, passing the pristine
0.19 source directory, output directory, Cargo dependency directory, and exact
`regex_automata`, `regex_syntax`, `bit_set` rlib paths. The script verifies the
provenance before compilation. Recorded logs:
`/private/tmp/eredu-fancy19-reference.log`,
`/private/tmp/eredu-fancy19-upstream-tests-1.log`,
`/private/tmp/eredu-fancy19-workspace-final.log`,
`/private/tmp/eredu-fancy19-construction-final.log`,
`/private/tmp/eredu-fancy19-dfa-tests.log`, and
`/private/tmp/eredu-tokenizer-fancy19-tests-2.log`.

Current release throughput compares complete `Split::pre_tokenize` operations
against the same 0.19 ordinary engine, checking independent output equality
before measurement. Each row runs for at least 300 milliseconds. Pattern
ordinals refer to the exact five-pattern inventory, not a model-family switch.

| Pattern | Input | Bytes | Ordinary µs | Closed source µs |
| --- | --- | ---: | ---: | ---: |
| 0 | ASCII chat repeated 512 times | 20,992 | 857.56 | 888.63 |
| 0 | Multilingual text repeated 512 times | 17,408 | 950.52 | 913.03 |
| 0 | 32,768 spaces then `x` | 32,769 | 323.85 | 333.97 |
| 2 | ASCII chat repeated 512 times | 20,992 | 699.39 | 656.86 |
| 2 | Multilingual text repeated 512 times | 17,408 | 582.77 | 526.14 |
| 2 | 32,768 spaces then `x` | 32,769 | 346.97 | 350.35 |

Source heap is **425,943 bytes** for pattern 0 and **2,553,938 bytes** for
pattern 2. Each operation's conservative full fixed workspace remains
**56,002,521 bytes**, chiefly the original million-branch VM bound and undo
records; this is declared storage, not measured RSS. The new ordinary engine
also improves the baseline: the prior 1.4–3.5× forced-workspace regression is
not present. Current relative differences range from 9.7% faster to 3.6% slower
and do not establish timing guarantees. Reproduce with the existing ignored
`canonical_regex_performance` tokenizer test in release mode. Exact input
strings, counts and output comparisons are in that test; the output log is
`/private/tmp/eredu-tokenizer-fancy19-performance.log`.

At this version-consolidation checkpoint, general schema funding was a separate
incomplete slice. Arbitrary source parsing, capture-name tables,
analysis/optimization, HIR/delegate automata, compiled control/error storage and
scoped runtime scratch subsequently received prospective producer hooks; see
[dynamic regex allocation evidence](bounded-followup-regex-engine-funding.md) and
[current schema compiler scope](bounded-followup-schema-compiler.md#current-scope).
The version and closed-tokenizer results above alone did not establish that
later funding closure. Optional logging and numeric-profile limits retain their
separate documented scope.
