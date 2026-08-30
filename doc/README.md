# Eredu documentation

This directory contains Eredu's user-facing technical documentation. Most
applications should start with the [`eredu` crate](../eredu/) and follow the
relevant model, execution, or platform guide here.

## Models and applications

- [Model and checkpoint support](model-support.md): supported families,
  modalities, artifact formats, and inspection.
- [Model loading, quantization, and memory](model-loading.md): weight residency,
  cache residency, prompt caches, and memory accounting.
- [Parallel execution](parallel-execution.md): tensor, pipeline, and expert
  parallel support.
- [Native tool calling](tool-calling.md): structured chat preparation,
  constraints, semantic events, and cancellation.
- [Speculative decoding and MTP](speculative-decoding.md): external and embedded
  assistants, stream placement, and lookahead behavior.
- [Cancellation and bounded execution](cancellation.md): scheduler lifecycle and
  the boundary between cooperative cancellation and submitted backend work.
- [PersonaPlex quantization
  evaluation](../eredu-evaluation/doc/personaplex-quantization.md): the audio
  comparison and blinded-listening tools in `eredu-evaluation`.
- [Evaluation and observation architecture](evaluation.md): portable evidence,
  backend parity, activation inspection, distribution metrics, and performance
  reporting.

## Runtime and backends

- [Language-model backend architecture](backend-architecture.md): neutral
  contracts, ownership boundaries, and the path for another backend.
- [MLX backend documentation](../eredu-backend-mlx/doc/README.md): concrete
  adapter architecture, native platform setup, and low-level implementation
  references.

## Maintainer references

- [Adding a native tool protocol](tool-protocol-development.md): evidence,
  fixtures, recognizers, constraints, and validation for a new wire format.
- [Releasing workspace crates](releasing.md): package validation and the
  dependency-ordered publication sequence.

API-level documentation is provided by each crate's Rustdoc. The guides here
focus on contracts and choices that span multiple APIs or crates.
