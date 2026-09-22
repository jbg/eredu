# Bounded inference validation

Validation is scoped to the mechanisms, fixtures, profiles and hardware listed
here. Counts overlap and must not be summed into a whole-repository total.
Builds, test filters, artifact digests and results are in
[`bounded-native-results.json`](validation/bounded-native-results.json).

## Behavioral coverage

| Suite | Result | Contract |
| --- | --- | --- |
| Portable facade | 38 pass, 2 external fixtures ignored | Ordinary prepared chat, source custody and controlled parity. |
| Backend conformance | 126 pass | Neutral public behavior and ordinary-only tool support. |
| Facade library | 372 pass, 6 ignored | Preparation, controllers and semantic publication. |
| Residual admission | 244 pass | Actual destinations, refusal causes and retained payers. |
| Architecture | 772 pass, 1 ignored | Family geometry and portable construction. |
| Runtime integration | 206 pass | Selected execution and source contracts. |
| Neural contracts | 216 unit and 5 doctests pass | Borrowed parameter/state and paid destinations. |
| Grammar/schema compiler | 90 pass | Recursive overlap, source compilation/copy and refusal. |
| Facade controller | 47 pass | Tool policy, parser state and source ownership. |
| Public native prepared chat | 7 pass | Tools, media, TP/PP, child sampling and controlled parity. |
| Distributed CLI | 6 pass | Dense/Mova Resident, Host and Disk; branch exchange and reset. |
| Focused native backend | 9 pass | Completion quarantine, selected group accounting and CPU/Metal population. |

Separate scoped suites cover 66 CPU/Metal source cases, six paged-control cases,
25 numerical mechanisms and 15 custody/reset/copy cases. Observation coverage
includes 13 workspace cases, two nonzero numerical comparisons and three
TP2/intervention cases. These are mechanism-specific evidence, not assertions
that every feature cross-product runs on every device.

## Native reproduction

The native environment uses Rust 1.98.0, Apple M3 Ultra with 256 GiB RAM,
CommandLineTools and MacOSX26.5 SDK. Use a writable compiler module cache and
actual Metal access. Native tests run serially. Large debug executables may need
a stripped copy to fit the host loader's mapping limits.

```sh
export RUST_MIN_STACK=67108864
export CARGO_INCREMENTAL=0
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export SDKROOT="$DEVELOPER_DIR/SDKs/MacOSX26.5.sdk"
export CLANG_MODULE_CACHE_PATH="${TMPDIR:-/tmp}/eredu-clang-cache"
cargo +1.98.0 test --offline -p eredu --no-default-features \
  --features mlx,metal,image,audio --test prepared_chat_native -- --test-threads=1
```

The exact optimization overrides and target commands are retained in the result
record. Public and CLI runs optimize six Rust packages at level 2 with assertions
and overflow checks. Two readiness cases use backend optimization level 0 and
selected neutral dependencies at level 2. Native development guards are enabled.
The 64 MiB test stack is explicit; default 2 MiB debug-stack behavior is unqualified.

Public media tests use 8 GiB. The distributed CLI matrix uses a finite 16 TiB
shared-domain allowance and a 240-second per-case deadline. Its conservative
native graph and control allowances are distinct from actual memory residency.
Current ledger results and outstanding CLI validation are recorded in
[physical memory validation](physical-memory-validation.md). These cases test
functional behavior rather than isolated performance.

## Numerical, performance and memory scope

[Released prepared-chat validation](prepared-chat-validation.md) pins the official
checkpoint, artifact hashes, request and event assertions. Its independent
resident numerical comparison covers prompts of 5 and 3000 tokens, each with
three cached decode steps: eight finite 248,320-value rows, matching argmax and
zero violations of `abs(actual-reference) <= 0.25 + 0.02*abs(reference)`.
Ordinary numerical inference does not establish bounded tool admission.

[Tokenizer memory policy](bounded-text-processing.md) distinguishes estimated
dependency headroom from first-party destinations. Behavioral parity does not
establish tokenization throughput or a process-memory ceiling.

The four released text/image tool runs use 3.13–3.19 GB peak RSS and
5.61–5.68 GB peak process footprint. Framework capacity, retained buffer bounds,
native allocation counters, RSS and process footprint are different quantities.
Registered allocator backings remain charged while a cache retains them.
Application buffers and unrelated process memory require their own accounting;
these framework charges do not establish a process-memory ceiling.

CUDA/NCCL and unlisted hardware, fresh funded file opening outside Unix, arbitrary
custom callback bounds and downstream application memory/cache policy are not
established by these runs. Prediction-extension and distributed speculative-bank
limits are described in [native execution](bounded-native-execution.md#support-limits).
