# eredu-evaluation

`eredu-evaluation` owns backend-independent evidence, comparison, statistics,
and evaluation drivers. It is not tied to one model family, execution mode, or
backend.

The crate currently provides:

- path-addressed portable evidence over the observation records in
  `eredu-core`;
- one parity engine for exact values, numeric tensors, and vocabulary logits;
- categorical-distribution metrics including KL divergence, target NLL delta,
  centered-logit RMSE, top-1 agreement, and top-k overlap;
- reusable latency and deadline summaries;
- a text-checkpoint artifact adapter and the `eredu-parity` command used by the
  cross-backend validation workflow; and
- the PersonaPlex dense-versus-quantized driver and blinded suite tooling.

Concrete backends do not implement evaluation policy. They implement ordinary
output observation and, where supported, named activation inspection through
the neutral core contracts. Backend-specific examples may select a device and
load artifacts, but comparisons and thresholds belong here.

See the [evaluation architecture guide](https://github.com/jbg/eredu/blob/main/doc/evaluation.md)
and the [PersonaPlex quantization evaluation guide](doc/personaplex-quantization.md).
