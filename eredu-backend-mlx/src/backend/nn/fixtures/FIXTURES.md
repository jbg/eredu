# MLX native quantization fixtures

`llama-c0bc8591-iq.oracle` covers the canonical IQ encodings using raw blocks
and scalar llama.cpp outputs rounded to F16. The MLX backend owns this copy
because its unit tests consume the oracle directly and must compile from the
published crate archive without workspace siblings.

Ordinary tests require neither llama.cpp nor a Metal device. Fixture changes
require outputs from an external implementation and review of both inputs and
expected values.
