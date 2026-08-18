# GGUF conversion fixtures

The oracle files validate GGUF tensor conversion independently of the Rust
implementation.

- `mlx-v0.32.0.oracle` covers dense and affine GGML types using raw blocks,
  logical metadata, packed weights, scales, biases, and dequantized F16 values.
- `llama-c0bc8591-iq.oracle` covers the canonical IQ encodings using raw blocks
  and scalar llama.cpp outputs rounded to F16.

Ordinary tests require neither MLX nor llama.cpp. Fixture changes require
outputs from an external implementation and review of both inputs and expected
values.
