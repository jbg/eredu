use super::*;
use eredu_nn::{GatedProductActivation, GatedProductPolicy, Index, NeuralBackend, Tensor};

#[derive(Clone, Copy, Debug)]
enum Case {
    Initialize,
    HostInitialize,
    ZerosLike,
    Add,
    Divide,
    Square,
    Tanh,
    Exp,
    MultiplyScalar,
    MaximumScalar,
    Equal,
    Clip,
    Where,
    LogicalOr,
    Silu,
    Sigmoid,
    Softplus,
    Gelu,
    GeluApproximate,
    Elu,
    Gated(GatedProductPolicy),
    Views,
    StaticIndex,
    Reshape,
    Concatenate,
    Mask(Option<i32>),
}
fn cases() -> Vec<Case> {
    vec![
        Case::Initialize,
        Case::HostInitialize,
        Case::ZerosLike,
        Case::Add,
        Case::Divide,
        Case::Square,
        Case::Tanh,
        Case::Exp,
        Case::MultiplyScalar,
        Case::MaximumScalar,
        Case::Equal,
        Case::Clip,
        Case::Where,
        Case::LogicalOr,
        Case::Silu,
        Case::Sigmoid,
        Case::Softplus,
        Case::Gelu,
        Case::GeluApproximate,
        Case::Elu,
        Case::Gated(GatedProductPolicy::ordinary_silu()),
        Case::Gated(GatedProductPolicy::ordinary_gelu_approximate()),
        Case::Gated(
            GatedProductPolicy::new(GatedProductActivation::Silu, Some(0.5), Some(0.3), 1.7, 0.2)
                .unwrap(),
        ),
        Case::Views,
        Case::StaticIndex,
        Case::Reshape,
        Case::Concatenate,
        Case::Mask(None),
        Case::Mask(Some(3)),
    ]
}
fn equation<B: NeuralBackend>(
    case: Case,
    a: B::Tensor,
    b: B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    Ok(match case {
        Case::Initialize => B::Tensor::full_f32(0.2, a.shape(), context)?,
        Case::HostInitialize => {
            let count = a.shape().iter().map(|n| *n as usize).product();
            B::Tensor::from_f32_slice(&vec![0.2; count], a.shape(), context)?
        }
        Case::ZerosLike => a.zeros_like(context)?,
        Case::Add => B::add_residual(&a, &b, true, context)?,
        Case::Divide => a.divide(&b, context)?,
        Case::Square => a.square(context)?,
        Case::Tanh => a.tanh(context)?,
        Case::Exp => B::exp(a, context)?,
        Case::MultiplyScalar => a.multiply_scalar(0.7, context)?,
        Case::MaximumScalar => a.maximum_scalar(0.2, context)?,
        Case::Equal => a.equal_i32(0, context)?,
        Case::Clip => a.clip(&b.multiply_scalar(-1.0, context)?, &b, context)?,
        Case::Where => B::Tensor::where_condition(&a.equal_i32(0, context)?, &a, &b, context)?,
        Case::LogicalOr => a
            .equal_i32(0, context)?
            .logical_or(&b.equal_i32(1, context)?, context)?,
        Case::Silu => B::silu(a, context)?,
        Case::Sigmoid => B::sigmoid(a, context)?,
        Case::Softplus => B::softplus(a, 1.3, context)?,
        Case::Gelu => B::Tensor::gelu(&a, context)?,
        Case::GeluApproximate => B::gelu_approximate(a, context)?,
        Case::Elu => B::Tensor::elu(&a, 1.1, context)?,
        Case::Gated(policy) => B::gated_product(a, b, policy, context)?,
        Case::Views => {
            let tail = a.index(&[Index::Range(0, 1), Index::Full], context)?;
            let expanded = tail
                .broadcast_to(a.shape(), context)?
                .transpose(context)?
                .expand_dims(0, context)?;
            expanded.squeeze_axes(&[0], context)?
        }
        Case::StaticIndex => a.index(&[Index::At(-1), Index::Full], context)?,
        Case::Reshape => a.reshape(&[-1], context)?,
        Case::Concatenate => B::Tensor::concatenate(&[a, b], 0, context)?,
        Case::Mask(window) => B::causal_mask(a.shape()[0], 5, window, context)?,
    })
}
fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts { page_size: 16384, cpu_header: false },
        sdpa_blocks: None,
    }
}

#[test]
fn padding_rejects_shrinking_and_empty_edge_extension_and_counts_each_axis() {
    let layout = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
    for mode in [eredu_nn::PadMode::Constant, eredu_nn::PadMode::Edge] {
        let mut op = WorkspaceOperation {
            kind: WorkspaceOperationKind::Pad(mode),
            inputs: vec![layout(&[2, 3])],
            outputs: vec![layout(&[4, 7])],
        };
        assert!(mechanisms().operation_bound(&op).unwrap().is_some());
        assert_eq!(
            mechanisms()
                .host_workspace_bound(&op)
                .unwrap()
                .unwrap()
                .bytes,
            0
        );
        op.outputs[0] = layout(&[1, 7]);
        assert!(mechanisms().operation_bound(&op).is_err());
        assert!(mechanisms().host_workspace_bound(&op).is_err());
        op.inputs[0] = layout(&[0, 3]);
        op.outputs[0] = layout(&[2, 7]);
        assert_eq!(
            mechanisms().operation_bound(&op).is_err(),
            mode == eredu_nn::PadMode::Edge
        );
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_padding_peaks_fit_bounds_and_preserve_constant_and_edge_values() {
    use crate::MlxTensor;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let shapes: &[(&[i32], &[(i32, i32)])] = &[
        (&[2, 3], &[(0, 0), (0, 0)]),
        (&[2, 3], &[(1, 2), (0, 1)]),
        (&[2, 3, 4], &[(1, 1), (3, 2), (2, 1)]),
        (&[1, 3, 129], &[(0, 0), (2, 0), (0, 0)]),
        (&[2, 65, 257], &[(0, 1), (2, 3), (3, 1)]),
        (&[0, 3], &[(1, 2), (1, 1)]),
    ];
    let mut cases = 0;
    for (shape, widths) in shapes {
        let count = shape.iter().product::<i32>() as usize;
        let values = (0..count).map(|i| (i % 7 + 1) as f32).collect::<Vec<_>>();
        for (dtype, metadata_dtype) in [
            (Dtype::Float32, WorkspaceDtype::Float32),
            (Dtype::Float16, WorkspaceDtype::Float32),
            (Dtype::Bfloat16, WorkspaceDtype::Float32),
            (Dtype::Int32, WorkspaceDtype::Int32),
            (Dtype::Uint32, WorkspaceDtype::Uint32),
            (Dtype::Uint8, WorkspaceDtype::Uint8),
            (Dtype::Bool, WorkspaceDtype::Bool),
        ] {
            for strided in [false, true] {
                let physical = if strided {
                    shape.iter().rev().copied().collect::<Vec<_>>()
                } else {
                    shape.to_vec()
                };
                let source = Array::from_slice(&values, &physical)
                    .as_dtype(dtype, &stream)
                    .unwrap();
                let source = if strided {
                    source.transpose(&stream).unwrap()
                } else {
                    source
                };
                let source = MlxTensor::from_array(source);
                let original = source.to_f32_vec(&stream).unwrap();
                for mode in [eredu_nn::PadMode::Constant, eredu_nn::PadMode::Edge] {
                    if count == 0 && mode == eredu_nn::PadMode::Edge {
                        continue;
                    }
                    let context = WorkspaceContext::new(selected);
                    let meta = WorkspaceTensor::existing(
                        WorkspaceLayout::new(shape, metadata_dtype).unwrap(),
                        &context,
                    )
                    .unwrap();
                    let padded = WorkspaceTensor::pad(&meta, widths, mode, &context).unwrap();
                    let output_shape = padded.shape().to_vec();
                    let report = context.report(&[padded]).unwrap();
                    let bound = report.tensor_buffers.total_bytes.unwrap();
                    assert_eq!(report.host_workspace_bytes, Some(0));
                    stream.synchronize().unwrap();
                    let before = safemlx::memory::active_memory().unwrap();
                    safemlx::memory::reset_peak_memory().unwrap();
                    let output = MlxTensor::pad(&source, widths, mode, &stream).unwrap();
                    safemlx::transforms::eval([output.as_array()]).unwrap();
                    stream.synchronize().unwrap();
                    let observed = safemlx::memory::peak_memory()
                        .unwrap()
                        .saturating_sub(before) as u64;
                    assert!(
                        observed <= bound,
                        "{shape:?} {mode:?} {dtype:?} strided={strided}: {observed} > {bound}"
                    );
                    assert_eq!(output.shape(), output_shape);
                    assert_eq!(output.as_array().dtype(), dtype);
                    let actual = output.to_f32_vec(&stream).unwrap();
                    for (index, actual) in actual.iter().enumerate() {
                        let mut remaining = index;
                        let mut source_index = 0;
                        let mut stride = 1;
                        let mut inside = true;
                        for axis in (0..shape.len()).rev() {
                            let coordinate =
                                (remaining % output_shape[axis] as usize) as i32 - widths[axis].0;
                            remaining /= output_shape[axis] as usize;
                            inside &= coordinate >= 0 && coordinate < shape[axis];
                            if shape[axis] > 0 {
                                source_index +=
                                    coordinate.clamp(0, shape[axis] - 1) as usize * stride;
                            }
                            stride *= shape[axis] as usize;
                        }
                        let expected = if inside || mode == eredu_nn::PadMode::Edge {
                            original[source_index]
                        } else {
                            0.0
                        };
                        assert_eq!(*actual, expected, "{shape:?} {mode:?} index={index}");
                    }
                    eprintln!(
                        "PAD shape={shape:?} mode={mode:?} dtype={dtype:?} strided={strided} peak={observed} bound={bound} values={}",
                        actual.len()
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 154);
}

#[test]
fn common_equations_have_cold_bounds_and_views_preserve_paid_roots() {
    for shape in [[1, 1], [7, 13], [65, 257]] {
        for case in cases() {
            let context = WorkspaceContext::new(mechanisms());
            let layout = WorkspaceLayout::new(&shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                )));
            let a = WorkspaceTensor::existing(layout.clone(), &context).unwrap();
            let b = WorkspaceTensor::existing(layout, &context).unwrap();
            let output = equation::<WorkspaceBackend>(case, a, b, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert!(
                report.tensor_buffers.total_bytes.is_some(),
                "missing bound for {case:?}: {report:?}"
            );
            if matches!(case, Case::Views) {
                assert_eq!(report.tensor_buffers.total_bytes, Some(0));
                assert_eq!(report.tensor_buffers.retained_bytes, Some(0));
            } else {
                assert!(report.tensor_buffers.total_bytes.unwrap() > 0);
            }
        }
    }
}

#[test]
fn unknown_and_wider_integer_operations_cannot_acquire_a_bound() {
    let float = WorkspaceLayout::new(&[3, 7], WorkspaceDtype::Float32).unwrap();
    let mut operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::Elementwise("unknown_test_operation"),
        inputs: vec![float.clone()],
        outputs: vec![float.clone()],
    };
    assert!(mechanisms().operation_bound(&operation).unwrap().is_none());
    operation.kind = WorkspaceOperationKind::Elementwise("add");
    operation.inputs = vec![
        WorkspaceLayout::new(&[3, 7], WorkspaceDtype::Int32).unwrap(),
        WorkspaceLayout::new(&[3, 7], WorkspaceDtype::Uint32).unwrap(),
    ];
    assert!(mechanisms().operation_bound(&operation).unwrap().is_none());
    operation.inputs = vec![float.clone(), float];
    operation.outputs = vec![WorkspaceLayout::new(&[2, 7], WorkspaceDtype::Float32).unwrap()];
    assert!(mechanisms().operation_bound(&operation).is_err());
}

#[test]
fn empty_broadcasts_and_scalar_backing_are_still_charged() {
    let context = WorkspaceContext::new(mechanisms());
    let a = WorkspaceTensor::full_f32(1.0, &[0, 257], &context).unwrap();
    let scalar = WorkspaceTensor::full_f32(0.5, &[], &context).unwrap();
    let result = a.add(&scalar, &context).unwrap();
    let report = context.report(&[result]).unwrap();
    assert!(report.tensor_buffers.total_bytes.unwrap() > 0);
    assert!(report.tensor_buffers.retained_bytes.unwrap() > 0);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires an exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_basic_observed_peaks_fit_cold_equation_bounds() {
    use crate::{MlxTensor, backend::nn::shared::MlxNeuralBackend};
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for (dtype, other_dtype) in [
        (Dtype::Float32, Dtype::Float32),
        (Dtype::Float16, Dtype::Float16),
        (Dtype::Bfloat16, Dtype::Bfloat16),
        (Dtype::Float16, Dtype::Bfloat16),
        (Dtype::Bfloat16, Dtype::Float32),
    ] {
        for shape in [[1, 1], [7, 13], [65, 257]] {
            for case in cases() {
                let context = WorkspaceContext::new(selected);
                let layout = WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap();
                let a = WorkspaceTensor::existing(layout.clone(), &context).unwrap();
                let b = WorkspaceTensor::existing(layout, &context).unwrap();
                let declared = equation::<WorkspaceBackend>(case, a, b, &context).unwrap();
                let allowed = context
                    .report(&[])
                    .unwrap()
                    .tensor_buffers
                    .total_bytes
                    .unwrap();
                // Unequal dimensions transpose genuinely strided nonzero data.
                let count = (shape[0] * shape[1]) as usize;
                let input = |offset: f32, dtype: Dtype| {
                    let values = (0..count)
                        .map(|n| offset + (n % 23) as f32 * 0.015)
                        .collect::<Vec<_>>();
                    MlxTensor::from_array(
                        Array::from_slice(&values, &[shape[1], shape[0]])
                            .as_dtype(dtype, &stream)
                            .unwrap()
                            .transpose(&stream)
                            .unwrap(),
                    )
                };
                let a = input(-0.1, dtype);
                let b = input(0.2, other_dtype);
                safemlx::transforms::eval([a.as_array(), b.as_array()]).unwrap();
                let before = safemlx::memory::active_memory().unwrap();
                safemlx::memory::reset_peak_memory().unwrap();
                let output =
                    equation::<MlxNeuralBackend>(case, a.clone(), b.clone(), &stream).unwrap();
                safemlx::transforms::eval([output.as_array()]).unwrap();
                let observed = safemlx::memory::peak_memory()
                    .unwrap()
                    .saturating_sub(before) as u64;
                assert_eq!(output.shape(), declared.shape());
                eprintln!(
                    "basic dtype={dtype:?}/{other_dtype:?} shape={shape:?} case={case:?} observed={observed} bound={allowed}"
                );
                assert!(
                    observed <= allowed,
                    "{case:?}: native peak {observed} exceeds cold bound {allowed}"
                );
            }
        }
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires an exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_pooling_mask_peaks_fit_cold_bounds_and_integer_visibility() {
    use crate::backend::runtime::cache::kv::PoolingCache;
    use safemlx::{Device, DeviceType, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for (queries, offset, ratio) in [(7, 3, 4), (2, 9, 6), (7, 131, 128), (2, (1 << 24) + 1, 128)] {
        let pooled = (offset + queries) / ratio;
        let mut cache = PoolingCache::new(ratio).unwrap();
        let values =
            safemlx::ops::zeros_dtype(&[1, pooled, 1], safemlx::Dtype::Float32, &stream).unwrap();
        let retained = cache.update_and_fetch(values, &stream).unwrap();
        retained.evaluated().unwrap();
        let geometry =
            eredu_nn::operation_geometry::PoolingMaskGeometry::new(queries, pooled, offset, ratio)
                .unwrap();
        let operation = WorkspaceOperation {
            kind: WorkspaceOperationKind::PoolingMask(geometry),
            inputs: vec![],
            outputs: vec![WorkspaceLayout::new(&[queries, pooled], WorkspaceDtype::Bool).unwrap()],
        };
        let quote = selected.operation_bound(&operation).unwrap().unwrap();
        let WorkspaceOutputStorage::Allocate(output_bytes) = quote.outputs[0] else {
            unreachable!()
        };
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        let mask = cache.make_mask(queries, offset, &stream).unwrap().unwrap();
        let evaluated = mask.evaluated().unwrap();
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        assert!(
            observed <= output_bytes + quote.scratch_bytes,
            "pooling mask peak {observed} exceeds bound {}",
            output_bytes + quote.scratch_bytes
        );
        let actual = evaluated.try_to_vec::<bool>().unwrap();
        let expected = (0..queries)
            .flat_map(|query| {
                (0..pooled).map(move |column| {
                    (i64::from(column) + 1) * i64::from(ratio)
                        <= i64::from(offset) + i64::from(query) + 1
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires an exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_causal_mask_integer_positions_remain_exact_and_bounded() {
    use crate::backend::nn::tensor::create_causal_mask;
    use safemlx::{Device, DeviceType, Dtype, Stream, ops::indexing::TryIndexOp};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    // F32 coordinates round adjacent positions together at this boundary.
    let offset = (1 << 24) + 1;
    for window in [None, Some(0), Some(3)] {
        let geometry =
            eredu_nn::operation_geometry::CausalMaskGeometry::new(2, offset, window).unwrap();
        let operation = WorkspaceOperation {
            kind: WorkspaceOperationKind::CausalMask(geometry),
            inputs: vec![],
            outputs: vec![
                WorkspaceLayout::new(&[2, geometry.keys()], WorkspaceDtype::Bool).unwrap(),
            ],
        };
        let quote = selected.operation_bound(&operation).unwrap().unwrap();
        let allowed = quote.scratch_bytes
            + match quote.outputs[0] {
                WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, .. } => bytes,
                _ => unreachable!(),
            };
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        let mask = create_causal_mask(2, Some(offset), window, None, &stream).unwrap();
        safemlx::transforms::eval([&mask]).unwrap();
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        assert!(
            observed <= allowed,
            "mask peak {observed} exceeds {allowed}"
        );
        let tail = mask
            .try_index_device((.., offset - 3..geometry.keys()), &stream)
            .unwrap()
            .as_dtype(Dtype::Int32, &stream)
            .unwrap()
            .contiguous(false, &stream)
            .unwrap();
        let actual = tail.evaluated().unwrap().try_to_vec::<i32>().unwrap();
        let expected = (0..2)
            .flat_map(|query| {
                (offset - 3..geometry.keys()).map(move |key| {
                    i32::from(
                        key <= offset + query
                            && window.is_none_or(|distance| key >= offset + query - distance),
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "window {window:?}");
        eprintln!(
            "large-mask offset={offset} window={window:?} observed={observed} bound={allowed}"
        );
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires an exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_empty_broadcast_activations_keep_scalar_storage_within_bounds() {
    use crate::{MlxTensor, backend::nn::shared::MlxNeuralBackend};
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for case in [
            Case::Silu,
            Case::Sigmoid,
            Case::Softplus,
            Case::Gelu,
            Case::GeluApproximate,
            Case::Gated(GatedProductPolicy::ordinary_silu()),
        ] {
            let context = WorkspaceContext::new(selected);
            let layout = WorkspaceLayout::new(&[0, 13], WorkspaceDtype::Float32).unwrap();
            let a = WorkspaceTensor::existing(layout.clone(), &context).unwrap();
            let b = WorkspaceTensor::existing(layout, &context).unwrap();
            let declared = equation::<WorkspaceBackend>(case, a, b, &context).unwrap();
            let allowed = context
                .report(&[])
                .unwrap()
                .tensor_buffers
                .total_bytes
                .unwrap();
            let input = MlxTensor::from_array(
                Array::full::<f32>(&[0, 13], Array::from_f32(0.2), &stream)
                    .unwrap()
                    .as_dtype(dtype, &stream)
                    .unwrap(),
            );
            safemlx::transforms::eval([input.as_array()]).unwrap();
            let before = safemlx::memory::active_memory().unwrap();
            safemlx::memory::reset_peak_memory().unwrap();
            let output =
                equation::<MlxNeuralBackend>(case, input.clone(), input.clone(), &stream).unwrap();
            safemlx::transforms::eval([output.as_array()]).unwrap();
            let observed = safemlx::memory::peak_memory()
                .unwrap()
                .saturating_sub(before) as u64;
            assert_eq!(output.shape(), declared.shape());
            assert!(
                observed <= allowed,
                "empty {case:?}: peak {observed} exceeds {allowed}"
            );
            eprintln!(
                "empty-activation dtype={dtype:?} case={case:?} observed={observed} bound={allowed}"
            );
        }
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn unsigned_token_equality_funds_i64_promotion_and_matches_signed_values() {
    use crate::MlxTensor;
    use safemlx::{Array, Device, DeviceType, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let values = [0u32, 1, i32::MAX as u32, 1u32 << 31, u32::MAX, 7];
    let source = MlxTensor::from_array(Array::from_slice(&values, &[2, 3]));
    for scalar in [-1, 0, 1, i32::MAX] {
        let context = WorkspaceContext::new(selected);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 3], WorkspaceDtype::Uint32).unwrap(),
            &context,
        ).unwrap();
        let output = input.equal_i32(scalar, &context).unwrap();
        let report = context.report(&[output]).unwrap();
        let bound = report.tensor_buffers.total_bytes.unwrap();
        stream.synchronize().unwrap();
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        let actual = source.equal_i32(scalar, &stream).unwrap();
        safemlx::transforms::eval([actual.as_array()]).unwrap();
        stream.synchronize().unwrap();
        let observed = safemlx::memory::peak_memory().unwrap().saturating_sub(before) as u64;
        assert!(observed <= bound, "{scalar}: observed {observed}, bound {bound}");
        let expected: Vec<f32> = values.iter().map(|&value| {
            if i64::from(value) == i64::from(scalar) { 1.0 } else { 0.0 }
        }).collect();
        assert_eq!(actual.to_f32_vec(&stream).unwrap(), expected);
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn unsigned_media_placeholders_preserve_full_token_range_and_concatenation_bound() {
    use crate::MlxTensor;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let tail = [7u32, u32::MAX];
    let native_tail = MlxTensor::from_array(Array::from_slice(&tail, &[1, 2]));
    for scalar in [0u32, i32::MAX as u32 + 1, u32::MAX] {
        for count in [0, 1, 9] {
            let context = WorkspaceContext::new(selected);
            let input = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[1, 2], WorkspaceDtype::Uint32).unwrap(), &context,
            ).unwrap();
            let placeholder = WorkspaceTensor::full_u32(scalar, &[1, count], &context).unwrap();
            let output = WorkspaceTensor::concatenate(&[placeholder, input], 1, &context).unwrap();
            assert_eq!(output.layout().dtype(), WorkspaceDtype::Uint32);
            let bound = context.report(&[output]).unwrap().tensor_buffers.total_bytes.unwrap();
            stream.synchronize().unwrap();
            let before = safemlx::memory::active_memory().unwrap();
            safemlx::memory::reset_peak_memory().unwrap();
            let placeholder = MlxTensor::full_u32(scalar, &[1, count], &stream).unwrap();
            let output = MlxTensor::concatenate(&[placeholder, native_tail.clone()], 1, &stream).unwrap();
            safemlx::transforms::eval([output.as_array()]).unwrap();
            stream.synchronize().unwrap();
            let observed = safemlx::memory::peak_memory().unwrap().saturating_sub(before) as u64;
            assert!(observed <= bound, "scalar={scalar}, count={count}: {observed} > {bound}");
            assert_eq!(output.as_array().dtype(), Dtype::Uint32);
            let actual = output.as_array().evaluated().unwrap().try_to_vec::<u32>().unwrap();
            let mut expected = vec![scalar; count as usize];
            expected.extend_from_slice(&tail);
            assert_eq!(actual, expected);
        }
    }
}
