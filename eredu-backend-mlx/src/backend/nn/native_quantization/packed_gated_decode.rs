use super::{dispatch::native_execution_backend, storage::NativeExecutionBackend, *};
use crate::backend::nn::linear::PhysicalLinear;

/// Fuses two compatible packed Q4_K matrix-vector projections and ordered SiLU.
pub(crate) fn try_packed_gated_decode(
    gate: &PhysicalLinear,
    up: &PhysicalLinear,
    input: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if input.dtype() != Dtype::Float32
        || gate.input_dimensions <= 0
        || input.size() != gate.input_dimensions as usize
        || input.shape().last().copied() != Some(gate.input_dimensions)
        || gate.input_dimensions != up.input_dimensions
        || gate.output_dimensions < 1024
        || gate.output_dimensions != up.output_dimensions
        || gate.bias.as_ref().is_some()
        || up.bias.as_ref().is_some()
    {
        return Ok(None);
    }
    for operation in [gate, up] {
        if operation.gguf.and_then(|format| format.gguf_iquant())
            != Some((GgmlType::Q4K, GgufEndian::Little))
            || operation.global_scale.as_ref().is_some()
            || operation.weight.as_ref().dtype() != Dtype::Uint8
        {
            return Ok(None);
        }
    }

    if native_execution_backend(stream)? != NativeExecutionBackend::Metal {
        return Ok(None);
    }

    let columns = gate.input_dimensions;
    let outputs = gate.output_dimensions;
    let gate_weight = NativeQuantizedTensor::from_iq_array(
        gate.weight.as_ref().clone(),
        &[outputs, columns],
        GgmlType::Q4K,
        GgufEndian::Little,
    )?;
    let up_weight = NativeQuantizedTensor::from_iq_array(
        up.weight.as_ref().clone(),
        &[outputs, columns],
        GgmlType::Q4K,
        GgufEndian::Little,
    )?;
    let flat = input.reshape(&[1, columns], stream)?;
    const TILE: i32 = 2;
    const GROUPS: i32 = 4;
    let workgroups = (outputs - 1) / (TILE * (GROUPS / 2)) + 1;
    let config = CustomKernelConfig::new()
        .with_template_arg_dtype("T", Dtype::Float32)
        .with_template_arg_int("OUT_DIM", outputs)
        .with_template_arg_int("BLOCKS", columns / 256)
        .with_template_arg_int("ROW_START", 0)
        .with_template_arg_int("PHYSICAL_ROWS", outputs)
        .with_template_arg_int("ROWS_PER_SIMD", TILE)
        .with_template_arg_int("SIMD_GROUPS", GROUPS)
        .with_grid([32, workgroups * GROUPS, 1])
        .with_thread_group([32, GROUPS, 1])
        .with_output_arg([1, outputs], Dtype::Float32);
    thread_local! {
        static PAIRED: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    }
    let output = PAIRED.with(|cell| {
        let mut kernel = cell.borrow_mut();
        if kernel.is_none() {
            *kernel = Some(MetalKernel::new(
                "native_q4k_decode_gated_pair",
                ["input", "gate_weight", "up_weight"],
                ["out"],
                concat!(
                    include_str!("q4k_decode_pair_prologue.metal"),
                    include_str!("q4k_decode_dot.metal"),
                    include_str!("q4k_decode_gated_pair.metal"),
                ),
                super::super::ordered_f32::packed_gated_silu_header(),
                true,
                false,
            )?);
        }
        kernel
            .as_ref()
            .expect("paired packed decode kernel initialized")
            .apply_one_device(
                [
                    &flat,
                    gate_weight.storage.bytes(),
                    up_weight.storage.bytes(),
                ],
                &config,
                stream,
            )
    })?;
    let mut shape = input.shape().to_vec();
    *shape.last_mut().expect("input has feature dimension") = outputs;
    output.reshape(&shape, stream).map(Some)
}
