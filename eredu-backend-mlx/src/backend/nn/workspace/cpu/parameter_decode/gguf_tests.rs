use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxAcceleratorFamily, MlxBackend,
        MlxDeviceIdentity,
    },
    MlxTensor,
};
use eredu_checkpoint::LinearFormat;
use eredu_gguf::{Endian, GgmlType};
use eredu_nn::{
    parameter_values::{decode_parameter, ParameterDecoding, WorkspaceParameterDecoding},
    Tensor,
};
use safemlx::{Array, Device, DeviceType};

#[test]
fn original_cpu_gguf_decode_preserves_nonzero_values_and_custody() {
    if crate::tests::support::native_process::enter("cpu-standalone-gguf-decoding") {
        run(false);
    }
}
#[test]
fn original_metal_gguf_decode_preserves_nonzero_values_and_custody() {
    if crate::tests::support::native_process::enter("metal-standalone-gguf-decoding") {
        run(true);
    }
}
fn run(metal: bool) {
    let ledger = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
    let streams = if metal {
        PreparedExecutionStreams::for_factory(&ledger)
    } else {
        PreparedExecutionStreams::for_cpu_factory_with_matmul(&ledger, choice)
    }
    .unwrap()
    .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(
            &Device::new(
                if metal {
                    DeviceType::Gpu
                } else {
                    DeviceType::Cpu
                },
                0,
            ),
            metal.then_some(MlxAcceleratorFamily::Metal),
        )
        .unwrap(),
    );
    for shape in [&[2, 32][..], &[1, 2, 32][..], &[2, 2, 32][..]] {
        for endian in [Endian::Little, Endian::Big] {
            let count = shape.iter().map(|&n| n as usize).product::<usize>();
            let mut packed = Vec::new();
            let mut expected = Vec::new();
            for block in 0..count / 32 {
                let scale = half::f16::from_f32((block + 1) as f32 * 0.25);
                packed.extend_from_slice(&match endian {
                    Endian::Little => scale.to_bits().to_le_bytes(),
                    Endian::Big => scale.to_bits().to_be_bytes(),
                });
                for i in 0..32 {
                    let value = i as i8 - 15;
                    packed.push(value as u8);
                    expected.push(f32::from(value) * scale.to_f32());
                }
            }
            let source = MlxTensor::from_array(Array::from_slice(&packed, &[packed.len() as i32]));
            let context = if metal {
                WorkspaceContext::new(ordinary)
            } else {
                WorkspaceContext::new(cpu)
            };
            let input = WorkspaceTensor::existing(
                context
                    .layout(&[packed.len() as i32], WorkspaceDtype::Uint8)
                    .unwrap(),
                &context,
            )
            .unwrap();
            context.begin_state_span([&input]).unwrap();
            let decoding = ParameterDecoding {
                format: LinearFormat::GgufIQuant {
                    ggml_type: GgmlType::Q8_0,
                    endian,
                },
                row_layout: eredu_nn::LinearRowLayout::Contiguous,
            };
            let output = decode_parameter(
                WorkspaceParameterDecoding::new(&input, None, None, shape, &context).unwrap(),
                decoding,
            )
            .unwrap();
            let report = context.finish_report(&[output]).unwrap();
            assert!(
                report.unpriced_operations.is_empty(),
                "{shape:?} {endian:?}: {report:?}"
            );
            assert!(
                report.unpriced_host_operations.is_empty(),
                "{shape:?} {endian:?}: {report:?}"
            );
            assert!(report.host_workspace_bytes.unwrap() >= count as u64 * 4);
            let recipe = SpeculativeNumericalRecipe::inspect_completed_outputs(
                &report,
                1,
                if metal {
                    ResidentExecutionMechanisms::Metal(ordinary)
                } else {
                    ResidentExecutionMechanisms::Cpu { ordinary, cpu }
                },
                &context,
            )
            .unwrap();
            assert_eq!(recipe.completion.nested_completions, 2);
            super::super::test_execution::run(
                recipe,
                &backend,
                &[&source],
                |stream| {
                    let decoded = crate::native_quantization::NativeQuantizedTensor::from_iq_array(
                        source.as_array().clone(),
                        shape,
                        GgmlType::Q8_0,
                        endian,
                    )
                    .unwrap()
                    .dequantize(stream)
                    .unwrap();
                    MlxTensor::from_array(
                        decoded
                            .reshape(shape, stream)
                            .unwrap()
                            .as_dtype(Dtype::Float32, stream)
                            .unwrap(),
                    )
                },
                |actual| {
                    assert_eq!(actual.shape(), shape);
                    assert_eq!(
                        actual.as_array().evaluated().unwrap().as_slice::<f32>(),
                        expected
                    );
                },
            );
        }
    }
}
