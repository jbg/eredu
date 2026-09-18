# Regex syntax allocation inventory

Cold grammar construction reaches `regex_syntax::Parser::parse` from derivre's
`ExprSet::parse_expr` and llguidance's regex rewriting. The original registry
dependency was regex-syntax 0.8.11, archive SHA-256
`d6f6ff9a378485b298a5286656da665ba74413d36db0979633275d2e708145d4`.

The dependency owns the actual allocation mechanisms: AST parser group/class
stacks, capture-name and comment storage, scratch strings, boxed AST nodes and
error pattern copies; HIR translation adds visitor/translation stacks, literal
buffers, capture names, Unicode/byte interval sets, and optimized HIR node
construction. AST/HIR teardown also uses iterative traversal storage and must
be included. Quoting only a completed HIR or derivre's hash-cons table misses
these earlier allocations.

The canonical parser and translator workers accept a neutral allocation
callback with a fixed refusal marker. Existing ordinary entry points use an
unenforced policy on the same workers. The dependency must not depend on derivre
or host admission contracts. Derivre's `ParserAllocationFunding` supplies the
callback and retains the concrete first refusal plus account through the entire
grammar compiler and terminal failures; partial AST/HIR values retire while
that outer owner remains live. No separate restricted parser or ordinary
fallback will replace the general syntax implementation.

The initial audit identified HIR constructor/interval-set allocation propagation
and iterative teardown as necessary parts of the same bound. Source growth and
error allocations must be covered before they occur; quoting a finished tree is
insufficient.

The local fork is now a workspace member selected by the root patch, with the
workspace unsafe-code prohibition and complete original archive/file hashes.
The borrowed `Allocation::reserve` policy returns a fixed `AllocationError`;
`Allocator` supplies explicit geometric vector/string replacement and exact
box/slice producers. AST parsing, comments, capture-name copies, group/class
stacks, Unicode-name scratch, node boxes, visitor stacks and error pattern copies
use those producers. A refusal creates a fixed error without another pattern
allocation. Error formatting streams its at-most-two spans into the caller's
writer without intermediate vectors or strings.

AST and class-set destructors use capacities 1, 2, 4, ... for iterative stacks.
Their total requested replacement storage is less than four slots per visited
node. Node construction reserves that credit plus the possible AST empty-span
replacement box before the node exists; partial trees therefore also have their
retirement storage paid. HIR movement and retirement use a private empty
properties sentinel, eliminating the original replacement-property allocation.
HIR constructors reserve their new property box and four HIR stack slots. These
are facts of the actual construction/teardown algorithm, independent of pattern
length or semantic acceptance.

The first full Unicode test run passes all 147 upstream tests; the added AST
matrix passes as the 148th test. It refuses every reached callback for nested
groups/alternation/repetition, verbose comments, Unicode names, class-set
operations, duplicate names and incomplete syntax. Each refusal returns the
fixed marker on the first failed callback, without subsequent funding attempts,
and supports diagnostic output to a non-allocating writer. The direct compiler
commands use the real Rust 1.98 compiler and all default Unicode feature flags;
temporary output is `/private/tmp/regex-syntax-upstream-tests-2.log`.

HIR smart constructors, translator and Unicode class/interval producers now use
the same fallible workers. Case folding and Unicode queries preserve allocation
refusals separately from unavailable-data and syntax errors. Class normalization
sorts/merges in place; failures during append-based operations restore canonical
class state without allocating cleanup storage.

The full suite passes 157 tests with all default Unicode features and 152 with
minimal features, including the complete AST-to-HIR and individual Unicode/set
producer refusal matrices. A separate executable compares against the pristine
registry source after verifying all 42 original file hashes: all 5,285 deterministic patterns
match in AST, HIR, scalar properties and error diagnostics. Its corpus includes
all pairs of 13 literal/class/assertion atoms with seven quantifiers, 4,096
generated concrete-syntax cases (seed `0x92d68ca2`), and six focused error/prefix
cases. The reference implementation exists only in this validation executable.

Reproduce the independent comparison with the pinned archive extracted outside
the tracked tree (archive SHA-256 above):

```sh
bash third-party/regex-syntax-0.8.11/validation/run-reference.sh \
  /path/to/pristine/regex-syntax-0.8.11 /tmp/regex-syntax-reference
cargo test -p regex-syntax --lib
cargo test -p regex-syntax --no-default-features --lib
```

Temporary full-suite output is
`/private/tmp/regex-syntax-upstream-tests-4.log`; independent output is
`/private/tmp/regex-syntax-reference-final.log`. Derivre/llguidance integration
and whole cold-grammar accounting remain separate checks; dependency parser
coverage alone does not establish the complete grammar compiler bound.
