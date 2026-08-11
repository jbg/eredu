# GGUF conversion fixtures

This directory contains deterministic oracle data for validating GGUF tensor
conversion independently of the Rust implementation.

- `mlx-v0.32.0.oracle` covers dense and affine GGML types. Each row records raw
  blocks, logical output metadata, packed MLX weights, scales and biases, and
  dequantized F16 values from MLX.
- `llama-c0bc8591-iq.oracle` covers the nine canonical IQ encodings. Each row
  records two deterministic raw blocks and scalar llama.cpp outputs rounded to
  F16.

Inputs exercise multiple blocks, both signs, extreme scales, zero and maximum
codes, and bit/codebook boundaries. The oracle files are fixed test inputs;
ordinary test runs do not require an MLX or llama.cpp checkout.

The fixtures are intentionally independent of the decoder under test. Changing
them requires an external implementation and a review of the resulting raw
blocks and expected values.
