use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxAcceleratorFamily, MlxBackend,
        MlxDeviceIdentity,
    },
    MlxTensor,
};
use eredu_checkpoint::{AffineQuantization, LinearFormat};
use eredu_nn::{
    parameter_values::{decode_parameter, ParameterDecoding, WorkspaceParameterDecoding},
    Tensor,
};
use safemlx::{Array, Device, DeviceType};

struct FailureContext(LinearFormat, bool);
impl Drop for FailureContext {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("standalone decoder source: {:?}, Metal={}", self.0, self.1);
        }
    }
}

#[test]
fn original_cpu_standalone_decoders_preserve_nonzero_packed_values_and_custody() {
    if !crate::tests::support::native_process::enter("cpu-standalone-packed-decoding") {
        return;
    }
    run(false);
}
#[test]
fn original_metal_standalone_decoders_preserve_nonzero_packed_values_and_custody() {
    if !crate::tests::support::native_process::enter("metal-standalone-packed-decoding") {
        return;
    }
    run(true);
}
fn run(metal: bool) {
    let startup = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
    let streams = if metal {
        PreparedExecutionStreams::for_factory(&startup)
    } else {
        PreparedExecutionStreams::for_cpu_factory_with_matmul(&startup, choice)
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
    for affine in [true, false] {
        for bits in if affine {
            &[2, 3, 4, 5, 6, 8][..]
        } else {
            &[4][..]
        } {
            let (rows, width, group) = (2usize, 96usize, 32usize);
            let codes = (0..rows * width)
                .map(|n| ((n * 13 + 3) & ((1 << bits) - 1)) as u32)
                .collect::<Vec<_>>();
            let mut words = vec![0u32; rows * width * bits / 32];
            for (i, &value) in codes.iter().enumerate() {
                let bit = i * bits;
                words[bit / 32] |= value << (bit % 32);
                if bit % 32 + bits > 32 {
                    words[bit / 32 + 1] |= value >> (32 - bit % 32);
                }
            }
            let scales = (0..rows * width / group)
                .map(|n| (n % 5 + 1) as f32 / 16.)
                .collect::<Vec<_>>();
            let biases = (0..scales.len())
                .map(|n| (n as i32 % 3 - 1) as f32 / 8.)
                .collect::<Vec<_>>();
            let exponents = (0..scales.len())
                .map(|n| 126 + (n % 3) as u8)
                .collect::<Vec<_>>();
            let lut = [
                0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
            ];
            let expected = codes
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    if affine {
                        n as f32 * scales[i / group] + biases[i / group]
                    } else {
                        lut[n as usize] * 2f32.powi(i32::from(exponents[i / group]) - 127)
                    }
                })
                .collect::<Vec<_>>();
            let shape = [rows as i32, width as i32];
            let packed_shape = [rows as i32, (width * bits / 32) as i32];
            let companion_shape = [rows as i32, (width / group) as i32];
            let weight = MlxTensor::from_array(Array::from_slice(&words, &packed_shape));
            let scale = MlxTensor::from_array(if affine {
                Array::from_slice(&scales, &companion_shape)
            } else {
                Array::from_slice(&exponents, &companion_shape)
            });
            let bias =
                affine.then(|| MlxTensor::from_array(Array::from_slice(&biases, &companion_shape)));
            for source in [&weight, &scale].into_iter().chain(bias.as_ref()) {
                source.as_array().evaluated().unwrap();
            }
            let context = if metal {
                WorkspaceContext::new(ordinary)
            } else {
                WorkspaceContext::new(cpu)
            };
            let symbolic_weight = WorkspaceTensor::existing(
                context
                    .layout(&packed_shape, WorkspaceDtype::Uint32)
                    .unwrap(),
                &context,
            )
            .unwrap();
            let symbolic_scale = WorkspaceTensor::existing(
                context
                    .layout(
                        &companion_shape,
                        if affine {
                            WorkspaceDtype::Float32
                        } else {
                            WorkspaceDtype::Uint8
                        },
                    )
                    .unwrap()
                    .with_representation(affine.then_some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &context,
            )
            .unwrap();
            let symbolic_bias = affine.then(|| {
                WorkspaceTensor::existing(
                    context
                        .layout(&companion_shape, WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap()
            });
            context
                .begin_state_span(
                    [&symbolic_weight, &symbolic_scale]
                        .into_iter()
                        .chain(symbolic_bias.as_ref()),
                )
                .unwrap();
            let format = if affine {
                LinearFormat::Affine(AffineQuantization::new(group as i32, *bits as i32).unwrap())
            } else {
                LinearFormat::MxFp4
            };
            let _failure_context = FailureContext(format, metal);
            let output = decode_parameter(
                WorkspaceParameterDecoding::new(
                    &symbolic_weight,
                    Some(&symbolic_scale),
                    symbolic_bias.as_ref(),
                    &shape,
                    &context,
                )
                .unwrap(),
                ParameterDecoding {
                    format,
                    row_layout: eredu_nn::LinearRowLayout::Contiguous,
                },
            );
            let output = output.unwrap();
            assert_eq!(
                output.layout().representation().unwrap().dtype(),
                WorkspaceFloatingType::Float32
            );
            let report = context.finish_report(&[output]).unwrap();
            assert!(
                report.unpriced_operations.is_empty(),
                "{format:?}: {report:?}"
            );
            assert!(
                report.unpriced_host_operations.is_empty(),
                "{format:?}: {report:?}"
            );
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
            let inputs = [&weight, &scale];
            let input_refs = inputs.into_iter().chain(bias.as_ref()).collect::<Vec<_>>();
            super::super::test_execution::run(
                recipe,
                &backend,
                &input_refs,
                |stream| {
                    let decoded = safemlx::ops::dequantize_with_mode(
                        weight.as_array(),
                        scale.as_array(),
                        bias.as_ref().map(MlxTensor::as_array),
                        group as i32,
                        *bits as i32,
                        if affine {
                            safemlx::ops::QuantizationMode::Affine
                        } else {
                            safemlx::ops::QuantizationMode::MxFp4
                        },
                        stream,
                    )
                    .unwrap();
                    MlxTensor::from_array(decoded.as_dtype(Dtype::Float32, stream).unwrap())
                },
                |actual| {
                    let actual = actual.as_array().evaluated().unwrap();
                    let values = actual.as_slice::<f32>();
                    assert_eq!(values.len(), expected.len());
                    for (i, (&actual, &expected)) in values.iter().zip(&expected).enumerate() {
                        assert!(
                            (actual - expected).abs() < 1e-6,
                            "{format:?} {metal} index{i}: {actual} != {expected}"
                        );
                    }
                },
            );
        }
    }
}
