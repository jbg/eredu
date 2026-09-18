# Regex retained source storage inventory

The prospective dynamic constructor and scoped invocation workers are connected.
The schema source census additionally requires exact retained immutable storage;
cumulative constructor callbacks include temporary replacements and cannot supply
that answer. The initial inventory below is followed by the implemented traversal and its
validation.

Canonical owning groups and required traversal:

- Fancy `Regex`: original/delegated pattern string capacities, shared name map
  header/table/key storage, shared VM program header/instruction vector, per
  instruction literals/classes/delegate sources, initial scratch pool storage.
- Meta `Regex`: shared implementation, strategy and metadata property owners;
  uniquely owned pool closure and empty pool shell/rows. Metadata is shared with
  the selected strategy and must only be counted once.
- Thompson NFA: shared NFA header, state/start vectors and state-owned transition
  or alternative boxes; separately shared GroupInfo, including exact name hash
  tables, row vectors and shared name payloads.
- Dense and one-pass automata: actual retained transition/start/match/accelerator
  capacities. Lazy DFA source retains its NFA and configuration; dynamic state
  maps and rows belong to the invocation cache.
- Prefilters: actual retained pattern, searcher and automaton owners, including
  the pinned Aho and memchr backing. Shared prefilter owners can appear in
  multiple selected engines.

The borrowed callback reports each actual shared owner identity and its
header plus exclusively owned backing, then recursively visits separately shared
children. Returning false skips an already-seen or refused group. The enclosing
source collector retains its concrete failure and deduplication storage; the
regex dependency does not allocate a second census graph or select policy.

A fresh source's empty persistent pool shell is retained source storage. Scoped
searches leave it empty for both enforced and unenforced policy. A query must
explicitly distinguish a source whose ordinary persistent pool has been warmed;
it cannot silently describe mutable pooled caches as absent. Invocation-owned
workspaces and their caches remain separate from immutable source storage.


## Implemented ownership traversal

`fancy_regex::Regex::visit_source_storage` and the underlying automata owner
methods now use `regex_automata::util::source_storage::Visitor`. A group reports
the actual owner pointer, shell and exclusive backing; separately shared children
receive separate visits. Returning false skips that owner's children. The caller
owns any identity set and concrete refusal, so inspection itself allocates no
second graph inside the dependency.

The implementation covers each group above, including native prefilter adapters,
Aho and memchr owners. Actual Arc headers, table allocation layouts, Vec and
String capacities and boxed payloads are included. The HIR property summary now
counts its actual optional box instead of charging an absent box. Invocation-only
DFA rows, VM stacks and delegate caches are excluded from immutable source bytes.

Cold persistent pools are counted with their real shell, closure and fixed initial
rows. If an ordinary pooled search has used a source, inspection returns typed
`Error::WarmedPool`; it cannot silently omit mutable cache storage. Scoped searches
leave this source state cold under either allocation policy. Source-census tests
verify stable identities and bytes after scoped execution, alias deduplication,
actual pattern-only growth for a shared-program clone, and warmed-pool rejection.
The schema source inspector consumes these visits using its own paid dedupe set.

The independent whole-engine performance runner records exact retained source
bytes and owner-group counts: 74-byte tokenizer source 163,020 bytes/589 groups;
274-byte tokenizer source 558,278 bytes/2,030 groups; 41-byte lookahead source
55,518 bytes/160 groups; 21-byte Unicode backreference source 2,405 bytes/7 groups;
11-byte atomic-stack source 2,273 bytes/8 groups. These numbers are separate from
prospective construction totals, which also contain temporary destinations.
See `bounded-followup-regex-engine-funding.md` for the source definitions,
reproduction runner and timing methodology. Schema integration and its actual
invocation-scope cost are recorded in [schema compiler evidence](bounded-followup-schema-compiler.md).
