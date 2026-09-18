# Prepared parser frame lifetime

The prepared parser holds one borrowed frame workspace for each actual chart
operation. Its exact borrowed allocation callback fixes the payer and error type.
The workspace pays its counter and `Rc` shell before construction, then reserves
additional capacity only when simultaneous fixed frames exceed its previous
peak. Derivre owns the shared mechanism; the sealed adapter accepts existing
allocation closures and the parser's allocation account. It provides no
execution permission, and its guards cannot outlive either scope or payer.

Each entered frame has an RAII guard. Sequential lexer/vector visits reuse the
same retained capacity. Recursive Earley `advance::run` and `hidden` calls retain
their parent guards while entering child frames, so overlapping frames remain
additive. Normal returns, typed failures and unwinds release live slots without
refunding the account. An independent operation or parser copy starts a fresh
workspace using its own callback. No workspace capacity is copied from the
historical source.

Cold grammar inspection, schema construction/copy and JSON/Lark emission now use
the same dependency-owned mechanism for their reached recursive controls. Their
producer coverage and current validation status are recorded in
[cold grammar compilation](bounded-followup-grammar.md); the measurements below
describe the earlier mutable-parser correction specifically.

This changes only the lifetime accounting of the existing worker's fixed
frames. Heap/table growth still calls the original cumulative allocation callback
before every producer. Source constructors, independent parser copies, emitted
masks and escaped errors retain their separate storage charges. A failed parser
keeps its existing sticky failure behavior and original partial state; the
facade's owning failure cells retain the account after the scope retires.
There is no capacity increase, grammar-language change, alternate activation
policy or retry with unenforced allocation.

The historical first correction separated borrowed mutable frames from owned
constructor frames. That removed false ownership charges but still paid all sequential stack
visits cumulatively: two released-tokenizer masks plus `I` commitment consumed
13,223,435,767 charged bytes over 4,079,607 callbacks. The scoped implementation
preserves the same full trie/grammar traversal while paying its actual frame
peak. With `I am unable to`, five full masks plus four commitments and completion
checks consume 845,584,155 charged bytes over 12,296 callbacks. This includes the
unchanged 794,322,049-byte source-copy request. The final finite-ceiling run
takes 24.86 seconds in the debug test profile.

The ignored regression uses the actual pinned tokenizer, canonical controller,
Required tool policy, and a `reading` tool whose integer `value` is restricted to
`17`. It checks each allowed prefix token, nonterminal state and rejection of
premature EOS. Tokenizer SHA-256:
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
The passing final fixture uses a finite 2 GiB source pool and separate 2 GiB
operation account capped at 200,000 callback requests. Its actual source-pool
charge is 534,979,461 bytes. This is tokenizer/parser evidence;
it executes no model. Five masks do not establish a bound for every possible
48-token sequence.

Later [released text and image runs](bounded-followup-released-tools.md) include
the scoped parser and cold-compiler corrections. The concrete sensor request
passes Required and Auto under the unchanged 64 GiB limit, producing one complete
`reading({"value":17})` call and `GrammarComplete`. The original generic-action
Required request now completes 48 tokens without a funding failure, but emits
prose and terminates at `MaxTokens`; the accounting failure is closed while that
tool request remains unsuccessful. These are recorded resident Metal runs with
an explicit 64 MiB debug stack. Current CPU/distributed validation and pending
reruns remain separately scoped in the [native guide](bounded-followup-native.md).

At the scoped-parser checkpoint, the preexisting 77 llguidance tests passed. Two additional
workspace tests cover nested overlap, sequential reuse, independent payers,
unchanged heap debits, exact refusal, and unwind restoration. All 47 facade
constraint tests passed, with the two external-tokenizer probes ignored in that
ordinary suite. The copied-parser regression refuses all 189 reached mask
requests individually while its historical source rejects further debits. Each
cut retains the exact typed failure and destination account until the failed
prefix retires, without another callback after refusal.

That refusal sweep also exposed transparent error forwarding which skipped a
leaf `HostMetadataFundingError` in the public source chain. The existing inline
funding variants now expose that exact leaf as their source. Their display,
storage layout, allocation behavior and account custody are unchanged.

```sh
cargo test --offline -p llguidance --lib
cargo test --offline -p eredu --no-default-features --lib scoped_mask_refusals_keep_exact_destination_payer_and_failed_prefix -- --nocapture
EREDU_RELEASED_TOKENIZER_JSON=/path/outside/repository/tokenizer.json \
  cargo test --offline -p eredu --no-default-features --lib released_required_prose_prefix_under_finite_funding -- --ignored --nocapture
```

Tests use the pinned Rust 1.98 compiler with `CARGO_INCREMENTAL=0`. Logs:
`/private/tmp/tokenizer-frame-workspace-tests1.log`,
`/private/tmp/tokenizer-frame-workspace-guards-tests.log`,
`/private/tmp/tokenizer-frame-workspace-custody-tests3.log`,
`/private/tmp/tokenizer-frame-workspace-constraints-tests.log`, and
`/private/tmp/tokenizer-released-grammar-scoped-prefix-final.log`.

The upstream package remains llguidance 1.8.0, archive SHA-256
`207e72ede15e79f1e7a7f0e2683387da031c1931f6de650020afb80c571efb67`.
The cached original archive, every original-file hash recorded in
`third-party/parser-upstream.json`, and the retained upstream license were
rechecked. Those provenance entries describe the unchanged original archive;
local source changes are intentionally not substituted into the original-file
hash manifest. No unsafe code or new dependency was added.
