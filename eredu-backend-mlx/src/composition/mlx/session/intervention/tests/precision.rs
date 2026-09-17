use super::*;
use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType, WorkspaceMechanisms,
    WorkspaceOperationKind, WorkspaceRepresentation, WorkspaceTensor,
};
use half::{bf16, f16, slice::HalfBitsSliceExt};

fn bits(dtype: InterventionDtype, value: f32) -> u16 {
    match dtype {
        InterventionDtype::Float16 => f16::from_f32(value).to_bits(),
        InterventionDtype::Bfloat16 => bf16::from_f32(value).to_bits(),
        _ => unreachable!(),
    }
}
fn number(dtype: InterventionDtype, value: u16) -> f32 {
    match dtype {
        InterventionDtype::Float16 => f16::from_bits(value).to_f32(),
        InterventionDtype::Bfloat16 => bf16::from_bits(value).to_f32(),
        _ => unreachable!(),
    }
}
fn values(dtype: InterventionDtype, values: Vec<u16>) -> InterventionValues {
    match dtype {
        InterventionDtype::Float16 => InterventionValues::Float16(values),
        InterventionDtype::Bfloat16 => InterventionValues::Bfloat16(values),
        _ => unreachable!(),
    }
}
fn floating(dtype: InterventionDtype) -> WorkspaceFloatingType {
    match dtype {
        InterventionDtype::Float16 => WorkspaceFloatingType::Float16,
        InterventionDtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
        _ => unreachable!(),
    }
}
fn read_bits(array: &Array) -> Vec<u16> {
    let evaluated = array.evaluated().unwrap();
    match array.dtype() {
        Dtype::Float16 => evaluated
            .try_iter::<f16>()
            .unwrap()
            .map(|v| v.to_bits())
            .collect(),
        Dtype::Bfloat16 => evaluated
            .try_iter::<bf16>()
            .unwrap()
            .map(|v| v.to_bits())
            .collect(),
        _ => unreachable!(),
    }
}

#[test]
fn prepared_half_activation_actions_preserve_bits_and_round_scalars_cpu() {
    verify(safemlx::DeviceType::Cpu);
}
#[cfg(feature = "metal")]
#[test]
fn prepared_half_activation_actions_preserve_bits_and_round_scalars_metal() {
    verify(safemlx::DeviceType::Gpu);
}

fn verify(device: safemlx::DeviceType) {
    let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
    for dtype in [InterventionDtype::Float16, InterventionDtype::Bfloat16] {
        let mut source_bits: Vec<u16> = (0..14)
            .map(|i| bits(dtype, i as f32 * 0.25 - 1.5))
            .collect();
        let nan = if dtype == InterventionDtype::Float16 {
            0x7e55
        } else {
            0x7fc5
        };
        source_bits[0] = nan; // Untouched source NaN payload must not be canonicalized.
        source_bits[1] = 0x8000;
        let transposed: Vec<u16> = (0..7)
            .flat_map(|column| [source_bits[column], source_bits[7 + column]])
            .collect();
        let array = match dtype {
            InterventionDtype::Float16 => {
                Array::from_slice(transposed.reinterpret_cast::<f16>(), &[7, 2])
            }
            InterventionDtype::Bfloat16 => {
                Array::from_slice(transposed.reinterpret_cast::<bf16>(), &[7, 2])
            }
            _ => unreachable!(),
        }
        .transpose_axes(&[1, 0], &stream)
        .unwrap();
        let input = MlxTensor::from_array(array);
        let factor = if dtype == InterventionDtype::Float16 {
            1.0004
        } else {
            1.003
        };
        let rounded_factor = number(dtype, bits(dtype, factor));
        assert_ne!(rounded_factor, factor);
        // Finite payload includes subnormal and signed zero. Replacement must
        // preserve these bit patterns, without a widen/narrow numerical pass.
        let replacement = [1, 0x8000, bits(dtype, 0.3333)];
        let bias = [bits(dtype, 0.125), bits(dtype, -0.5), bits(dtype, 1.75)];
        for case in 0..7 {
            let whole = case == 3 || case == 6;
            let slice = ResolvedCaptureSlice {
                starts: vec![1, if whole { 0 } else { 1 }],
                ends: vec![2, 7],
                strides: vec![1, if whole { 1 } else { 2 }],
                shape: vec![1, if whole { 7 } else { 3 }],
            };
            let tensor = |data: &[u16]| InterventionTensor {
                shape: vec![1, 3],
                values: values(dtype, data.to_vec()),
            };
            let action = match case {
                0 => InterventionAction::Zero { dtype },
                1 => InterventionAction::Scale { dtype, factor },
                2 => InterventionAction::Mask {
                    dtype,
                    shape: vec![1, 3],
                    keep: vec![true, false, true],
                },
                3 => InterventionAction::MaskComponents {
                    dtype,
                    indices: vec![1, 5],
                    keep_selected: true,
                },
                4 => InterventionAction::Replace {
                    tensor: tensor(&replacement),
                },
                5 => InterventionAction::Add {
                    tensor: tensor(&bias),
                },
                _ => InterventionAction::MaskLogits {
                    dtype,
                    token_ids: vec![0, 4],
                },
            };
            let mut expected = source_bits.clone();
            for column in 0..7 {
                if !whole && ![1, 3, 5].contains(&column) {
                    continue;
                }
                let index = 7 + column;
                let ordinal = column / 2;
                expected[index] = match case {
                    0 => 0,
                    1 => bits(dtype, number(dtype, source_bits[index]) * rounded_factor),
                    2 => {
                        if ordinal == 1 {
                            0
                        } else {
                            source_bits[index]
                        }
                    }
                    3 => {
                        if [1, 5].contains(&column) {
                            source_bits[index]
                        } else {
                            0
                        }
                    }
                    4 => replacement[ordinal],
                    5 => bits(
                        dtype,
                        number(dtype, source_bits[index]) + number(dtype, bias[ordinal]),
                    ),
                    _ => {
                        if [0, 4].contains(&column) {
                            bits(dtype, f32::NEG_INFINITY)
                        } else {
                            source_bits[index]
                        }
                    }
                };
            }
            if case == 1 {
                assert!(
                    [8, 10, 12]
                        .iter()
                        .any(|i| expected[*i]
                            != bits(dtype, number(dtype, source_bits[*i]) * factor)),
                    "fixture must distinguish rounded scalar from an F32 multiplication"
                );
            }
            let program = PreparedStaticActivation::new(&action, &slice, &[2, 7], dtype).unwrap();
            let population = program.population().unwrap();
            let mut native = NativeCapture {
                stream: &stream,
                domain: None,
                partition: None,
            };
            let output =
                eredu_runtime::intervention::apply_activation(&mut native, &input, &action, &slice)
                    .unwrap();
            assert_eq!(
                read_bits(output.as_array()),
                expected,
                "{dtype:?} action {case}"
            );
            assert_eq!(read_bits(input.as_array()), source_bits, "source changed");

            let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
            let context = WorkspaceContext::new(mechanism);
            let source = WorkspaceTensor::existing(
                context
                    .layout(&[2, 7], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        floating(dtype),
                        false,
                    ))),
                &context,
            )
            .unwrap();
            let wrong = WorkspaceTensor::existing(
                context
                    .layout(&[2, 7], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        false,
                    ))),
                &context,
            )
            .unwrap();
            context.begin_span();
            let mut retained = Vec::new();
            assert!(program.trace(&wrong, &context, &mut retained).is_err());
            assert!(
                retained.is_empty(),
                "mismatched source must fail before construction"
            );
            let traced = program.trace(&source, &context, &mut retained).unwrap();
            assert_eq!(
                traced.layout().representation().unwrap().dtype(),
                floating(dtype)
            );
            assert_eq!(traced.layout().dtype(), WorkspaceDtype::Float32);
            assert_eq!(retained.len(), population.retained_roots);
            assert_eq!(population.host_bytes, if whole { 7 } else { 0 });
            assert_eq!(population.completions, population.retained_roots);
            let report = context.report(&retained).unwrap();
            let cast = report
                .operations
                .iter()
                .find(|op| matches!(op.kind, WorkspaceOperationKind::CastFloating(_)));
            if [1, 2, 3, 6].contains(&case) {
                let cast = cast.expect("real scalar conversion must be recorded");
                assert!(
                    matches!(cast.kind, WorkspaceOperationKind::CastFloating(to) if to == floating(dtype))
                );
                assert_eq!(
                    cast.inputs[0].representation().unwrap().dtype(),
                    WorkspaceFloatingType::Float32
                );
                assert_eq!(
                    cast.outputs[0].representation().unwrap().dtype(),
                    floating(dtype)
                );
            } else {
                assert!(cast.is_none());
            }
            for operation in &report.operations {
                assert!(
                    mechanism.operation_bound(operation).unwrap().is_some(),
                    "unpriced {operation:?}"
                );
                assert!(
                    mechanism.host_workspace_bound(operation).unwrap().is_some(),
                    "unpriced host {operation:?}"
                );
            }
            #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
            {
                // The same report must lower every actual operation, including
                // the typed initialization and AsType, without native work.
                let plan =
                    crate::backend::nn::workspace::OriginalComponentTestPlan::from_report(report);
                assert_eq!(plan.completion.nested_completions, 0);
            }
        }
        let bad = InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![1, 3],
                values: values(dtype, vec![nan, 0, 0]),
            },
        };
        let slice = ResolvedCaptureSlice {
            starts: vec![1, 1],
            ends: vec![2, 7],
            strides: vec![1, 2],
            shape: vec![1, 3],
        };
        assert!(
            PreparedStaticActivation::new(&bad, &slice, &[2, 7], dtype).is_err(),
            "nonfinite payload policy must remain unchanged"
        );
        let mismatched_payload = InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![1, 3],
                values: InterventionValues::Float32(vec![0.5; 3]),
            },
        };
        assert!(
            PreparedStaticActivation::new(&mismatched_payload, &slice, &[2, 7], dtype).is_err()
        );
    }
}
