# Custom tool grammar source producers

Inventory before implementation (2026-09-18). Compared with parent `c513e17`,
the current shared declaration API already passes `ParserAllocationFunding` and
uses `grammar_text::Text` for the enclosing declarative grammar. The remaining
custom helpers still construct strings, vectors, and required-name trees before
that account sees the resulting text.

- `lfm2.rs`: Python grammar rule names, top-level alternatives, object fields,
  suffix rules, enum values, nested literals, and diagnostic strings use ordinary
  allocating producers. Canonical destination: the existing paid `Text` writer;
  owned rule strings and vector growth must use the same funding owner.
- `dialect.rs`: tagged parameter grammar copies alternatives, type annotations,
  schema JSON, and required names; structural object grammar repeats the Python
  optional-field sequence algorithm. Canonical destination: `Text`, borrowed
  schema membership, and one shared optional-field writer parameterized by the
  actual comma spelling. Tagged parameter nesting has different semantics and
  remains a distinct writer.
- Remove intermediate joins, serialized schema strings, required-name trees,
  and temporary schema objects where a borrowed traversal suffices. Preserve
  field order, rule numbering, exact escaping, optional-field behavior, and
  specialized value syntax. Incremental response parsers are outside this slice;
  shared identifier/type spelling helpers must still use one implementation.

Validation will exercise grammar behavior and refusal at reached producer
allocations. This slice does not establish a total cold compiler bound: schema
projection, declaration compilation, dependency internals, and retained source
ownership have their own producers and evidence.

## Implemented producers and evidence

Both custom builders now retain the original funding owner in their `Text`
destinations. Rule names and vectors reserve prospective backing; enum/nested
literal output streams directly. Required membership is computed once per
field and retained as a boolean, so suffix expansion does not repeatedly scan
the required array. Python and structural formats call one iterative optional
field writer. Tagged suffix nesting retains its distinct semantics and now
writes iteratively. A shared `Quoted<Display>` worker handles nested escaping;
ordinary borrowed literals delegate to that same escaping implementation.

Tagged local references also reached `Value::pointer`, whose prior worker made
two replacement strings per segment. `pointer_with_allocations` now owns the
same ordinary/enforced lookup worker: unescaped segments borrow, escaped
segments reserve one destination before unescaping. The facade forwards the
actual parser owner and recovers its original refusal. The existing 64-hop
local-reference policy is unchanged. Runtime lexical diagnostics and compact
type names still use the same spelling implementations as grammar construction.

The source snapshot portable facade run passed **335 tests**, with **4 existing
ignored tests** (12.54 seconds). Command, with the actual compiler selected:

```sh
CARGO_INCREMENTAL=0 \
RUSTC=/Users/jbg/.rustup/toolchains/1.98.0-aarch64-apple-darwin/bin/rustc \
cargo test -p eredu --no-default-features --lib --offline
```

Eight focused producer tests passed. Each successful funded result equals the
ordinary same-worker result; every reached refusal stops subsequent requests
and retains the original concrete error. Explicit expectations cover Python
nested values and escaping, tagged optional nesting, structural token IDs,
compact type annotations, and escaped local references. Independent JSON
serialization supplies the nested-escaping oracle.

| Fixture | Output bytes | Reservation requests | Cumulative reserved bytes |
| --- | ---: | ---: | ---: |
| Python nested tool | 2,008 | 62 | 22,346 |
| Structural nested tool | 1,547 | 61 | 22,431 |
| Tagged tool | 1,102 | 13 | 6,011 |
| Tagged escaped local reference | 355 | 9 | 2,625 |
| Invalid Python name diagnostic | 67 | 3 | 863 |

These are reached cumulative reservations, including shared writer/storage
controls; they are not retained heap sizes or process RSS. Full-facade evidence:
`/private/tmp/eredu-custom-grammar-facade-tests.log`; focused output:
`/private/tmp/eredu-custom-grammar-producer-final.log`.

The two pointer tests passed under both insertion-order and default sorted-map
profiles. They compare all 1,365 slash-free strings of length zero through five
over `~`, `0`, `1`, and a Unicode scalar against the prior two-pass replacement
semantics, check array index rejection and empty keys, and refuse each reached
escaped-component allocation. The direct test builds use the exact current
serde source/artifact because the upstream standalone dev dependency `automod`
is not cached. Logs: `/private/tmp/eredu-json-pointer-tests.log` and
`/private/tmp/eredu-json-pointer-sorted-tests.log`.

This closes these owning text/vector/tree-copy producers. It does not claim
complete cold admission: caller schema projection, recursive producer/control
storage, serializer controls, declaration compilation, and regex engine
construction need their own complete accounting evidence. Incremental response
parser catalog/state ownership remains outside this helper slice.
