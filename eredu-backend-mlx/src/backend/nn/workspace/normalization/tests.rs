use super::*;
use eredu_nn::{
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, NormalizationScale,
    ParameterMetadata, ParameterSpec, ParameterVisitorMut, Parameterized, Tensor,
};

#[derive(Clone, Copy, Debug)]
enum Case {
    Rms,
    WeightedRms,
    L2,
    Layer { weight: bool, bias: bool },
    Constructed { scale: u8, grouped: bool },
    Gated { after: bool },
}
fn cases() -> Vec<Case> {
    let mut cases = vec![Case::Rms, Case::WeightedRms, Case::L2];
    for weight in [false, true] {
        for bias in [false, true] {
            cases.push(Case::Layer { weight, bias });
        }
    }
    for scale in 0..3 {
        for grouped in [false, true] {
            cases.push(Case::Constructed { scale, grouped });
        }
    }
    cases.extend([Case::Gated { after: false }, Case::Gated { after: true }]);
    cases
}
struct Bind<'w, T>(&'w T);
impl<'a, T: Tensor + 'a> ParameterVisitorMut<'a, T> for Bind<'_, T> {
    fn visit_mut(&mut self, _: ParameterMetadata, value: &'a mut T) {
        *value = self.0.clone();
    }
}
fn equation<B: NeuralBackend>(
    case: Case,
    input: &B::Tensor,
    gate: &B::Tensor,
    weight: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let width = *input.shape().last().unwrap();
    let groups = if width % 4 == 0 {
        4
    } else if width % 3 == 0 {
        3
    } else {
        1
    };
    let epsilon = 1e-5;
    match case {
        Case::Rms => B::rms_norm_without_weight(input, epsilon, context),
        Case::WeightedRms => B::rms_norm_with_weight(input, weight, epsilon, context),
        Case::L2 => B::l2_normalize(input, epsilon, context),
        Case::Layer { weight: w, bias: b } => B::Tensor::layer_norm(
            input,
            w.then_some(weight),
            b.then_some(weight),
            epsilon,
            context,
        ),
        Case::Constructed { scale, grouped } => {
            let spec = NormalizationConstructionSpec {
                dimensions: width,
                epsilon,
                groups: grouped.then_some(groups),
                scale: match scale {
                    0 => NormalizationScale::Unit,
                    1 => NormalizationScale::Learned(
                        ParameterSpec::trainable("norm.weight").unwrap(),
                    ),
                    _ => NormalizationScale::LearnedOffset {
                        weight: ParameterSpec::trainable("norm.weight").unwrap(),
                        offset: 1.0,
                    },
                },
            };
            let mut norm = B::normalization(spec, context)?;
            norm.visit_parameters_mut(&mut Bind(weight));
            norm.forward(input, context)
        }
        Case::Gated { after: false } => {
            B::gated_group_rms_norm(input, gate, weight, groups, epsilon, context)
        }
        Case::Gated { after: true } => {
            B::silu_gated_group_rms_norm(input, gate, weight, groups, epsilon, context)
        }
    }
}
fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn meta_inputs(
    shape: &[i32],
    context: &WorkspaceContext,
) -> (WorkspaceTensor, WorkspaceTensor, WorkspaceTensor) {
    let input = || {
        WorkspaceTensor::existing(
            WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
            context,
        )
        .unwrap()
    };
    (
        input(),
        input(),
        WorkspaceTensor::existing(
            WorkspaceLayout::new(&[*shape.last().unwrap()], WorkspaceDtype::Float32).unwrap(),
            context,
        )
        .unwrap(),
    )
}

#[test]
fn all_normalization_policies_produce_cold_bounds_including_empty_rows() {
    for shape in [[1, 1], [7, 32], [2, 768], [3, 8193], [0, 32]] {
        for case in cases() {
            let context = WorkspaceContext::new(mechanisms());
            let (input, gate, weight) = meta_inputs(&shape, &context);
            let output =
                equation::<WorkspaceBackend>(case, &input, &gate, &weight, &context).unwrap();
            assert_eq!(output.shape(), shape);
            let report = context.report(&[output]).unwrap();
            assert!(
                report.tensor_buffers.total_bytes.is_some(),
                "missing {case:?}"
            );
            assert!(
                report.tensor_buffers.total_bytes.unwrap()
                    > report.tensor_buffers.retained_bytes.unwrap()
            );
        }
    }
}

#[test]
fn invalid_group_or_scale_geometry_cannot_acquire_normalization_authority() {
    let input = WorkspaceLayout::new(&[3, 32], WorkspaceDtype::Float32).unwrap();
    let mut operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::Normalization("rms", None),
        inputs: vec![
            input.clone(),
            WorkspaceLayout::new(&[31], WorkspaceDtype::Float32).unwrap(),
        ],
        outputs: vec![input.clone()],
    };
    assert!(mechanisms().operation_bound(&operation).is_err());
    operation.kind = WorkspaceOperationKind::Normalization("gated_group_rms_norm", Some(3));
    operation.inputs = vec![
        input.clone(),
        input,
        WorkspaceLayout::new(&[32], WorkspaceDtype::Float32).unwrap(),
    ];
    assert!(mechanisms().operation_bound(&operation).is_err());
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_normalization_peaks_fit_bounds_for_all_selected_policies() {
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for (dtype, weight_dtype) in [
        (Dtype::Float32, Dtype::Float32),
        (Dtype::Float16, Dtype::Float16),
        (Dtype::Bfloat16, Dtype::Bfloat16),
        (Dtype::Float16, Dtype::Bfloat16),
        (Dtype::Bfloat16, Dtype::Float32),
    ] {
        for shape in [[1, 1], [7, 32], [2, 768], [3, 8193]] {
            for case in cases() {
                let context = WorkspaceContext::new(selected);
                let (a, g, w) = meta_inputs(&shape, &context);
                let declared = equation::<WorkspaceBackend>(case, &a, &g, &w, &context).unwrap();
                let allowed = context
                    .report(&[])
                    .unwrap()
                    .tensor_buffers
                    .total_bytes
                    .unwrap();
                let count = (shape[0] * shape[1]) as usize;
                let input = |offset: f32| {
                    let values = (0..count)
                        .map(|n| offset + (n % 17) as f32 / 32.0)
                        .collect::<Vec<_>>();
                    MlxTensor::from_array(
                        Array::from_slice(&values, &[shape[1], shape[0]])
                            .as_dtype(dtype, &stream)
                            .unwrap()
                            .transpose(&stream)
                            .unwrap(),
                    )
                };
                let a = input(-0.1);
                let g = input(0.2);
                // A stride-two learned scale exercises parameter strides too.
                use safemlx::ops::indexing::{IntoStrideBy, TryIndexOp};
                let values = (0..shape[1] * 2)
                    .map(|n| 0.7 + (n % 13) as f32 / 64.0)
                    .collect::<Vec<_>>();
                let w = MlxTensor::from_array(
                    Array::from_slice(&values, &[shape[1] * 2])
                        .as_dtype(weight_dtype, &stream)
                        .unwrap()
                        .try_index_device((..).stride_by(2), &stream)
                        .unwrap(),
                );
                safemlx::transforms::eval([a.as_array(), g.as_array(), w.as_array()]).unwrap();
                let before = safemlx::memory::active_memory().unwrap();
                safemlx::memory::reset_peak_memory().unwrap();
                let output = equation::<MlxNeuralBackend>(case, &a, &g, &w, &stream).unwrap();
                safemlx::transforms::eval([output.as_array()]).unwrap();
                let observed = safemlx::memory::peak_memory()
                    .unwrap()
                    .saturating_sub(before) as u64;
                assert_eq!(output.shape(), declared.shape());
                assert!(
                    observed <= allowed,
                    "{case:?}: native peak {observed} exceeds {allowed}"
                );
                assert!(output
                    .to_f32_vec(&stream)
                    .unwrap()
                    .iter()
                    .all(|n| n.is_finite()));
                eprintln!("normalization dtype={dtype:?}/{weight_dtype:?} shape={shape:?} case={case:?} observed={observed} bound={allowed}");
            }
        }
    }
}
