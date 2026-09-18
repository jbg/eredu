# Shared quote diagnostics

`IncrementalInferenceQuote` and its typed `ResidualInferenceQuote` wrapper
retain one immutable paid `RuntimeStateEstimate`. Cloning a quote shares that
original report, including diagnostic strings and sliding-window rows. Native
startup, token steps, snapshots and resume already clone these quotes; those
operations no longer allocate uncharged diagnostic copies.

The report's shared shell is admitted through the same
`WorkspaceReportMetadata` destination as report construction. It retains that
destination's actual funding account. The closed owner exports neither raw Arc
nor Weak handles, and releases the Arc allocation before its report and account
retire. Ordinary and counted construction use the same owner and workers.

Cold span sealing and native-generation replacement can extend a candidate's
diagnostics. A unique candidate mutates its original report. If another quote
shares the report, the existing counted report copier creates a separate paid
report before mutation. Refusal preserves the earlier candidate and its source
custody. Numerical demand, source identity, reservation authority and the
existing span-seal comparisons are unchanged.

The five focused reservation-metadata tests pass, including allocation-free
quote aliases, paid report separation at sealing, unique-report reuse, refusal
before mutation, and final account retirement. The complete residual sweep now
passes 244 tests with zero failures (0.36 seconds). The new terminal refusal case
also exposed and verified a shared-planner fix: a zero-chunk terminal placement
returns its original refusal instead of attempting to shrink below zero.
Log: `/private/tmp/tokenizer-shared-quote-residual-complete.log`.
After the final fixed-control admission adjustment, all five focused quote
ownership/refusal tests also pass; log:
`/private/tmp/tokenizer-shared-quote-fixed-controls-final-tests.log`.

```sh
cargo test --offline -p eredu-runtime --lib working_memory::residual -- --test-threads=2
```
