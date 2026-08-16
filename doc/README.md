# SafeMLX documentation

This directory contains the user-facing technical documentation shared by the
SafeMLX crates. Start with the README for the crate you intend to use, then
follow the relevant guide here.

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
- [PersonaPlex quantization evaluation](personaplex-evaluation.md): the audio
  comparison and blinded-listening tools in `safemlx-codec`.

## Core runtime

- [Language-model backend architecture](backend-architecture.md): the neutral
  core contract, MLX adapter, ownership boundary, and future backend path.
- [Platform setup](platforms.md): native prerequisites for Apple, Linux, CUDA,
  and Windows builds.
- [Completion events](completion-events.md): graph submission, host observation,
  and same-device stream dependencies.
- [Asynchronous device timing](device-timing.md): execution-timeline timestamp
  boundaries, backend accuracy, and nonblocking profiling.
- [Host-transfer buffers](host-transfer-buffers.md): storage selection,
  ownership, and asynchronous copy rules.

## Maintainer references

- [Adding a native tool protocol](tool-protocol-development.md): evidence,
  fixtures, recognizers, constraints, and validation for a new wire format.

API-level documentation is provided by each crate's Rustdoc. The guides here
focus on contracts and choices that span multiple APIs or crates.
