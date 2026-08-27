# Evaluation and observation architecture

Eredu separates execution observation from evaluation policy. Backends expose
completed values through general diagnostics contracts; `eredu-evaluation`
turns those values into parity, quality, and performance evidence.

## Observation boundary

`eredu-core` owns the portable `ObservationSet`, `ObservationValue`, and
`TensorObservation` schemas as well as canonical protocol paths. Model output
logits use `MODEL_LOGITS_OBSERVATION_PATH` (`model.logits`) in every execution
mode rather than backend-local literals. Paths are stable semantic names, not
backend object names. Host materialization is always explicit:

- `BackendSession::observe_output` observes an already completed prefill or
  decode output;
- `InspectableBackendSession` runs an explicitly instrumented prefill or
  decode pass and returns selected named activations;
- `RealtimeBackend::observe_output` observes completed realtime tokens and
  requested decision logits; and
- `eredu-nn::Tensor::to_f32_vec` and `to_i32_vec` are the neutral host-transfer
  operations used by codec and architecture-level tools.

Ordinary inference does not enable these paths and therefore does not acquire
host-transfer or instrumentation overhead. The same records can be consumed by
evaluation, telemetry exporters, debuggers, model inspectors, and observability
tools.

Activation inspection is a property of the exact prepared session, not of a
device. Required inspection support is admitted from header-only architecture,
residency, and topology information before tensor payload materialization.
Instrumented execution wraps the production resident, bounded, expert-provider,
and distributed paths rather than running a second family implementation.
Distributed observations are rank-local; only the output-owning rank exposes
`model.logits`.

Architecture execution already emits named activation and routed-expert points
through `eredu-runtime::ActivationObserver`. A backend inspection adapter binds
those native tensor handles to the portable selected observations after the
instrumented operation completes. Evaluation code never imports a native
tensor, stream, or allocator type.

## Evaluation layer

`eredu-evaluation` owns reusable policy and reporting:

- `EvaluationEvidence` packages observations with kind and provenance;
- `compare_observations` applies exact, numeric, or vocabulary-logit parity
  rules independently of backend and modality;
- `compare_distributions` computes reference/candidate distribution metrics;
- `summarize_latencies` reports common latency percentiles and deadline misses;
  and
- adapters translate established external artifacts into observations without
  duplicating comparison algorithms.

The `eredu-parity` binary adapts the existing text checkpoint probe and
Transformers reference artifacts to this engine. Moshi and PersonaPlex fixture
parity use the same exact-value comparison. The PersonaPlex quantization driver
uses the shared distribution and latency metrics in addition to its
modality-specific audio preparation and listening-test reports.

## Adding evaluation coverage

New evaluation work should follow this order:

1. If the needed value is generally useful for diagnostics, add or reuse a
   neutral observation at the execution boundary that owns it.
2. Implement backend materialization without embedding thresholds, reference
   identities, or evaluation methodology in the backend.
3. Express comparisons and reports in `eredu-evaluation` using stable semantic
   paths.
4. Keep model- or modality-specific preparation in a narrow driver while
   reusing the general parity, distribution, timing, and evidence types.

Do not add evaluation-specific backend extension traits or reach through a
backend adapter to native implementation details.
