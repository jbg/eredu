# eredu-gguf

`eredu-gguf` is a bounded, framework-independent GGUF reader, writer, and
tensor converter written in safe Rust. It can inspect a complete sharded
checkpoint without loading tensor payloads, then materialize individual
tensors or selections on demand.

Use it for checkpoint tooling, validation, conversion, or as the storage layer
for a runtime. It does not depend on a concrete execution backend.

## Features

- GGUF v1-v3, little- or big-endian.
- Canonical shard discovery and whole-checkpoint descriptor validation.
- Configurable limits for metadata, arrays, tensors, ranks, and allocations.
- Dense, affine-quantized, IQ, and MXFP4-MoE tensor representations.
- Named, out-of-order materialization with bounded open-reader reuse.
- Native-axis selections for supported encodings and contiguous scalar spans
  for F32, F16, and BF16 tensors.
- Deterministic seekable writing with byte-preserving quantized payloads.

```rust,no_run
use eredu_gguf::Checkpoint;

let checkpoint = Checkpoint::open("model-00001-of-00004.gguf")?;
checkpoint.for_each_converted_tensor(|tensor| {
    println!("{} -> {:?}", tensor.descriptor().name, tensor.output_names());
    Ok(())
})?;
# Ok::<(), eredu_gguf::Error>(())
```

## Tensor encodings

Dense tensors support F32, F16, BF16, F64, I8, I16, I32, and I64. Affine
conversion supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
and Q6_K.

The canonical IQ encodings IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S,
IQ2_S, IQ4_XS, and IQ1_M are retained as explicit packed values because their
nonlinear codebooks cannot be represented as affine weights, scales, and
biases. Backends that execute those packed blocks directly obtain the canonical
typed values through `IQuantCodebook`; the generated table modules are private
implementation details. MXFP4-MoE type 39 is likewise represented explicitly.
Materialized tensor groups carry the logical output names established by the
validated catalog, so consumers do not reconstruct affine or MXFP4 companion
names from the physical tensor name.

Names such as `UD-Q2_K_XL` describe file-level mixed-precision recipes, not new
tensor encodings. A recipe is compatible when every tensor in the file uses a
supported encoding.

The reader rejects duplicate names, invalid alignment, impossible block
shapes, truncated or overlapping payloads, and arithmetic overflow before
exposing a checkpoint as valid. Reader and checkpoint errors retain structured
classifications across shard context; callers can query unsupported tensor
encodings with `Error::unsupported_tensor_type_code` without parsing diagnostic
text.

## License

Licensed under either Apache-2.0 or MIT.
