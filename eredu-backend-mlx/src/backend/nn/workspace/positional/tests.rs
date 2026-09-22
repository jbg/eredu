use super::*;
use eredu_nn::{multimodal::RotaryAxisSpec, NeuralBackend, RelativeAttentionInput, Tensor};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 16384,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    }
}
fn layout(shape: &[i32], dtype: WorkspaceDtype) -> WorkspaceLayout {
    WorkspaceLayout::new(shape, dtype).unwrap()
}
fn relative_op(
    dims: [i32; 7],
    origin: i32,
    window: Option<i32>,
    floor: Option<i32>,
) -> WorkspaceOperation {
    let [b, h, kv, q, k, d, p] = dims;
    let f = |s: &[i32]| layout(s, WorkspaceDtype::Float32);
    WorkspaceOperation {
        kind: WorkspaceOperationKind::RelativeAttention {
            query_offset: origin + k - q,
            key_offset: origin,
            window,
            log_scaling_floor: floor,
            log_scaling_alpha: 0.7,
        },
        inputs: vec![
            f(&[b, h, q, d]),
            f(&[b, kv, k, d]),
            f(&[b, kv, k, d]),
            f(&[b, h, q, p]),
        ],
        outputs: vec![f(&[b, h, q, d])],
    }
}
fn relative_execute<B: NeuralBackend>(
    op: &WorkspaceOperation,
    x: &[B::Tensor],
    c: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let WorkspaceOperationKind::RelativeAttention {
        query_offset,
        key_offset,
        window,
        log_scaling_floor,
        log_scaling_alpha,
    } = op.kind
    else {
        unreachable!()
    };
    B::relative_attention(
        RelativeAttentionInput {
            queries: &x[0],
            keys: &x[1],
            values: &x[2],
            profiles: &x[3],
            query_offset,
            key_offset,
            window,
            log_scaling_floor,
            log_scaling_alpha,
        },
        c,
    )
}
fn spec(dimensions: &[i32], kind: MultiAxisRotaryLayout) -> MultiAxisRotarySpec {
    MultiAxisRotarySpec {
        axes: dimensions
            .iter()
            .enumerate()
            .map(|(i, &dimensions)| RotaryAxisSpec {
                dimensions,
                position_offset: if i % 2 == 0 { -2 } else { 3 },
            })
            .collect(),
        base: 10000.,
        minimum_position: 0,
        layout: kind,
    }
}
fn rotary_op(
    leading: &[i32],
    spec: MultiAxisRotarySpec,
    dtype: WorkspaceDtype,
) -> WorkspaceOperation {
    let mut input = leading.to_vec();
    input.push(spec.axes.len() as i32);
    let mut output = leading.to_vec();
    output.push(spec.dimensions().unwrap());
    WorkspaceOperation {
        kind: WorkspaceOperationKind::MultiAxisRotary(spec),
        inputs: vec![layout(&input, dtype)],
        outputs: vec![layout(&output, WorkspaceDtype::Float32); 2],
    }
}

#[test]
fn prepared_rotary_preserves_native_buffers_and_removes_only_generated_host_vector() {
    for kind in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        let policy = spec(&[4, 8], kind);
        let ordinary = rotary_op(&[2, 3], policy.clone(), WorkspaceDtype::Int32);
        let mut prepared = ordinary.clone();
        prepared.kind = WorkspaceOperationKind::PreparedMultiAxisRotary(policy.clone());
        let mechanisms = mechanisms();
        let before = mechanisms.operation_bound(&ordinary).unwrap().unwrap();
        let after = mechanisms.operation_bound(&prepared).unwrap().unwrap();
        for operation in [&ordinary, &prepared] {
            for output in 0..2 {
                assert_eq!(
                    mechanisms.output_representation(operation.as_view(), output),
                    Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        false
                    ))
                );
            }
            assert!(mechanisms
                .output_representation(operation.as_view(), 2)
                .is_none());
        }
        assert_eq!(before.outputs, after.outputs);
        assert_eq!(before.scratch_bytes, after.scratch_bytes);
        assert!(after.scratch_bytes > 0);
        let old_host = super::super::host::operation_bound(&ordinary, &mechanisms)
            .unwrap()
            .unwrap();
        let new_host = super::super::host::operation_bound(&prepared, &mechanisms)
            .unwrap()
            .unwrap();
        assert_eq!(
            old_host.bytes,
            if kind == MultiAxisRotaryLayout::RoundRobinSections {
                24
            } else {
                16
            }
        );
        assert_eq!(new_host.bytes, 0);
    }
}

#[test]
fn positional_workspace_prices_relative_profiles_and_all_rotary_layouts() {
    for dims in [
        [1, 2, 1, 1, 17, 8, 5],
        [2, 4, 2, 3, 13, 5, 7],
        [1, 3, 1, 9, 21, 16, 4],
        [1, 2, 2, 1, 4097, 64, 17],
    ] {
        for window in [None, Some(1), Some(5)] {
            for floor in [None, Some(2)] {
                let op = relative_op(dims, 7, window, floor);
                let context = WorkspaceContext::new(mechanisms());
                let x = op
                    .inputs
                    .iter()
                    .map(|l| WorkspaceTensor::existing(l.clone(), &context).unwrap())
                    .collect::<Vec<_>>();
                let out = relative_execute::<WorkspaceBackend>(&op, &x, &context).unwrap();
                let report = context.report(&[out]).unwrap();
                assert!(report.tensor_buffers.total_bytes.is_some());
                assert_eq!(report.host_workspace_bytes, Some(0));
            }
        }
    }
    for kind in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        for axes in [&[2][..], &[4, 8], &[8, 4, 2]] {
            for leading in [&[7][..], &[2, 3], &[0], &[2, 0]] {
                for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Uint32] {
                    let spec = spec(axes, kind);
                    let op = rotary_op(leading, spec.clone(), dtype);
                    let context = WorkspaceContext::new(mechanisms());
                    let x = WorkspaceTensor::existing(op.inputs[0].clone(), &context).unwrap();
                    let (cos, sin) =
                        WorkspaceTensor::multi_axis_rotary_embeddings(&x, &spec, &context).unwrap();
                    let report = context.report(&[cos, sin]).unwrap();
                    let frequencies = if kind == MultiAxisRotaryLayout::RoundRobinSections {
                        axes.iter().sum::<i32>() / 2
                    } else {
                        *axes.iter().max().unwrap() / 2
                    };
                    assert_eq!(report.host_workspace_bytes, Some(frequencies as u64 * 4));
                    assert_eq!(
                        report.tensor_buffers.retained_bytes,
                        Some(
                            2 * capacity(
                                mechanisms().allocation,
                                op.outputs[0].elements().unwrap()
                            )
                            .unwrap()
                        )
                    );
                    assert!(report.tensor_buffers.transient_bytes.unwrap() > 0);
                }
            }
        }
    }
}

#[test]
fn positional_workspace_rejects_invalid_geometry_and_unknown_input_domains() {
    let mut op = relative_op([2, 4, 2, 3, 13, 5, 7], 7, None, Some(2));
    op.inputs[3] = layout(&[2, 3, 3, 7], WorkspaceDtype::Float32);
    assert!(mechanisms().operation_bound(&op).is_err());
    let mut op = relative_op([2, 4, 2, 3, 13, 5, 7], 7, None, Some(2));
    op.kind = WorkspaceOperationKind::RelativeAttention {
        query_offset: i32::MAX - 2,
        key_offset: 0,
        window: None,
        log_scaling_floor: None,
        log_scaling_alpha: 0.,
    };
    assert!(mechanisms().operation_bound(&op).is_err());
    let mut op = relative_op([2, 4, 2, 3, 13, 5, 7], 7, None, Some(2));
    op.inputs[0] = layout(op.inputs[0].shape(), WorkspaceDtype::Int32);
    assert!(mechanisms().operation_bound(&op).unwrap().is_none());
    assert!(mechanisms().host_workspace_bound(&op).unwrap().is_none());
    let rotary_spec = spec(&[4, 8], MultiAxisRotaryLayout::RoundRobinSections);
    for shape in [vec![2], vec![1, 3], vec![i32::MAX, 2, 2]] {
        let mut op = rotary_op(&[1], rotary_spec.clone(), WorkspaceDtype::Int32);
        op.inputs[0] = layout(&shape, WorkspaceDtype::Int32);
        assert!(mechanisms().operation_bound(&op).is_err());
        assert!(mechanisms()
            .output_representation(op.as_view(), 0)
            .is_none());
    }
    let op = rotary_op(&[2], rotary_spec, WorkspaceDtype::Float32);
    assert!(mechanisms().operation_bound(&op).unwrap().is_none());
    assert!(mechanisms()
        .output_representation(op.as_view(), 0)
        .is_none());
    let mut wrong = rotary_op(
        &[2],
        spec(&[4, 8], MultiAxisRotaryLayout::SplitHalves),
        WorkspaceDtype::Int32,
    );
    wrong.outputs[1] = layout(&[2, 11], WorkspaceDtype::Float32);
    assert!(mechanisms().operation_bound(&wrong).is_err());
    assert!(mechanisms()
        .output_representation(wrong.as_view(), 0)
        .is_none());
    let large = spec(
        &[2, 2, 2, i32::MAX - 7],
        MultiAxisRotaryLayout::RoundRobinSections,
    );
    let op = rotary_op(&[1], large, WorkspaceDtype::Int32);
    // Section*axis_count exceeds I32, but metadata and native selection use
    // widened arithmetic. Cold pricing does not iterate billions of columns.
    assert!(mechanisms().operation_bound(&op).unwrap().is_some());
}

#[test]
fn actual_vision_block_preserves_rotary_scalar_facts_through_attention_and_projection() {
    use eredu_architectures::qwen::vision::{VisionBlock, VisionConfigSource};
    use eredu_nn::{multimodal::PreparedMultiAxisRotary, ParameterVisitorMut, Parameterized};

    struct Bind<'c>(&'c WorkspaceContext);
    impl<'a> ParameterVisitorMut<'a, WorkspaceTensor> for Bind<'_> {
        fn visit_mut(
            &mut self,
            _: eredu_nn::ParameterMetadataView<'_>,
            value: &'a mut WorkspaceTensor,
        ) {
            *value = WorkspaceTensor::existing(
                layout(value.shape(), WorkspaceDtype::Float32).with_representation(Some(
                    WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false),
                )),
                self.0,
            )
            .unwrap();
        }
    }
    for activation in ["gelu", "gelu_pytorch_tanh", "silu"] {
        let config: VisionConfigSource = serde_json::from_value(serde_json::json!({
            "depth": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "num_position_embeddings": 16, "in_channels": 3,
            "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 1,
            "out_hidden_size": 16, "deepstack_visual_indexes": [],
            "hidden_act": activation,
        }))
        .unwrap();
        let config = config.normalize_qwen3_vl().unwrap();
        for prepared in [false, true] {
            let context = WorkspaceContext::new(mechanisms());
            let mut block = VisionBlock::<WorkspaceBackend>::new(&config, 0, &context).unwrap();
            block.visit_parameters_mut(&mut Bind(&context));
            let hidden = WorkspaceTensor::existing(
                layout(&[4, 16], WorkspaceDtype::Float32).with_representation(Some(
                    WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false),
                )),
                &context,
            )
            .unwrap();
            let positions =
                WorkspaceTensor::existing(layout(&[4, 2], WorkspaceDtype::Int32), &context)
                    .unwrap();
            let policy = spec(&[4, 4], MultiAxisRotaryLayout::SplitHalves);
            let mut frequencies = vec![0.; policy.as_ref().frequency_count().unwrap()];
            policy.as_ref().fill_frequencies(&mut frequencies).unwrap();
            context.begin_span();
            let (cosine, sine) = if prepared {
                WorkspaceTensor::multi_axis_rotary_embeddings_prepared(
                    &positions,
                    PreparedMultiAxisRotary::new(policy.as_ref(), &frequencies).unwrap(),
                    &context,
                )
            } else {
                WorkspaceTensor::multi_axis_rotary_embeddings(&positions, &policy, &context)
            }
            .unwrap();
            let output = block
                .forward(&hidden, &[2, 2], &cosine, &sine, &context)
                .unwrap();
            assert_eq!(output.shape(), [4, 16]);
            assert_eq!(
                output
                    .layout()
                    .representation()
                    .map(WorkspaceRepresentation::dtype),
                Some(WorkspaceFloatingType::Float32),
                "{activation}, prepared={prepared}"
            );
            let report = context.report(&[output]).unwrap();
            assert!(report.tensor_buffers.total_bytes.is_some());
            let mut projections = 0;
            let mut attention = 0;
            for operation in &report.operations {
                match operation.kind {
                    WorkspaceOperationKind::Projection(_) => projections += 1,
                    WorkspaceOperationKind::Attention { .. } => attention += 1,
                    _ => continue,
                }
                assert_eq!(
                    mechanisms()
                        .output_representation(operation.as_view(), 0)
                        .map(WorkspaceRepresentation::dtype),
                    Some(WorkspaceFloatingType::Float32),
                    "{activation}, prepared={prepared}: {:?}",
                    operation.kind
                );
            }
            assert_eq!(projections, 4);
            assert_eq!(attention, 2);

            // Logical F32 geometry alone still cannot authorize the downstream
            // collective. Use the same family worker with an unknown table.
            let unknown = WorkspaceTensor::existing(
                layout(cosine.shape(), WorkspaceDtype::Float32),
                &context,
            )
            .unwrap();
            let output = block
                .forward(&hidden, &[2, 2], &unknown, &sine, &context)
                .unwrap();
            assert!(output.layout().representation().is_none());
        }
    }
}

#[test]
fn zero_rotary_sections_preserve_global_cold_allocation_and_host_bounds() {
    for leading in [&[2][..], &[2, 3], &[0]] {
        let mut expected = None;
        for dimensions in [
            [2, 2, 4],
            [8, 0, 0],
            [0, 8, 0],
            [0, 0, 8],
            [0, 4, 4],
            [4, 0, 4],
            [4, 4, 0],
        ] {
            let policy = spec(&dimensions, MultiAxisRotaryLayout::RoundRobinSections);
            let op = rotary_op(leading, policy.clone(), WorkspaceDtype::Int32);
            let context = WorkspaceContext::new(mechanisms());
            let positions = WorkspaceTensor::existing(op.inputs[0].clone(), &context).unwrap();
            let (cos, sin) =
                WorkspaceTensor::multi_axis_rotary_embeddings(&positions, &policy, &context)
                    .unwrap();
            let report = context.report(&[cos, sin]).unwrap();
            assert_eq!(report.host_workspace_bytes, Some(4 * 4));
            let actual = (
                report.tensor_buffers.total_bytes,
                report.tensor_buffers.retained_bytes,
                report.tensor_buffers.transient_bytes,
                report.host_workspace_bytes,
            );
            if let Some(expected) = expected {
                assert_eq!(actual, expected);
            } else {
                expected = Some(actual);
            }
            let mut wrong = op;
            let mut shape = leading.to_vec();
            shape.push(2);
            wrong.inputs[0] = layout(&shape, WorkspaceDtype::Int32);
            assert!(mechanisms().operation_bound(&wrong).is_err());
        }
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };

    fn floating(
        shape: &[i32],
        slot: usize,
        dtype: Dtype,
        strided: bool,
        stream: &Stream,
    ) -> MlxTensor {
        let data = (0..count(shape).unwrap() as usize)
            .map(|i| ((i * 11 + slot * 5) % 73) as f32 / 128. - 0.25)
            .collect::<Vec<_>>();
        let data = if strided {
            data.iter().flat_map(|x| [*x, 0.125]).collect()
        } else {
            data
        };
        let mut a = Array::from_slice(&data, &[data.len() as i32])
            .as_dtype(dtype, stream)
            .unwrap();
        if strided {
            a = a.try_index_device((..).stride_by(2), stream).unwrap();
        }
        MlxTensor::from_array(a.reshape(shape, stream).unwrap())
    }
    fn input_positions(
        shape: &[i32],
        values: &[i32],
        unsigned: bool,
        strided: bool,
        stream: &Stream,
    ) -> MlxTensor {
        let data = if strided {
            values.iter().flat_map(|x| [*x, 0]).collect::<Vec<_>>()
        } else {
            values.to_vec()
        };
        let mut a = Array::from_slice(&data, &[data.len() as i32]);
        if unsigned {
            a = a.as_dtype(Dtype::Uint32, stream).unwrap();
        }
        if strided {
            a = a.try_index_device((..).stride_by(2), stream).unwrap();
        }
        MlxTensor::from_array(a.reshape(shape, stream).unwrap())
    }
    fn allowance(op: &WorkspaceOperation, m: MlxMetalWorkspaceMechanisms) -> u64 {
        let bound = m.operation_bound(op).unwrap().unwrap();
        bound.scratch_bytes
            + bound
                .outputs
                .iter()
                .map(|s| match s {
                    WorkspaceOutputStorage::Allocate(n)
                    | WorkspaceOutputStorage::AllocateOrAliasInputs { bytes: n, .. } => *n,
                    _ => panic!("unexpected positional output storage"),
                })
                .sum::<u64>()
    }
    fn relative_reference(
        op: &WorkspaceOperation,
        x: &[eredu_core::HostTensorBuffer<f32>],
    ) -> Vec<f64> {
        let WorkspaceOperationKind::RelativeAttention {
            query_offset,
            key_offset,
            window,
            log_scaling_floor,
            log_scaling_alpha,
        } = op.kind
        else {
            unreachable!()
        };
        let qs = op.inputs[0].shape();
        let ks = op.inputs[1].shape();
        let extent = op.inputs[3].shape()[3] as usize;
        let (b, h, q, d, kv, k) = (
            qs[0] as usize,
            qs[1] as usize,
            qs[2] as usize,
            qs[3] as usize,
            ks[1] as usize,
            ks[2] as usize,
        );
        let mut output = Vec::new();
        for batch in 0..b {
            for head in 0..h {
                for query in 0..q {
                    let position = i64::from(query_offset) + query as i64;
                    let tau = if window.is_none() {
                        log_scaling_floor.map_or(1., |floor| {
                            1. + log_scaling_alpha as f64
                                * ((position + 1) as f64 / floor as f64).max(1.).ln()
                        })
                    } else {
                        1.
                    };
                    let key_head = head / (h / kv);
                    let scores = (0..k)
                        .map(|key| {
                            let distance = position - (i64::from(key_offset) + key as i64);
                            if distance < 0 || window.is_some_and(|w| distance >= i64::from(w)) {
                                return f64::NEG_INFINITY;
                            }
                            let dot = (0..d)
                                .map(|dim| {
                                    x[0][((batch * h + head) * q + query) * d + dim] as f64
                                        * x[1][((batch * kv + key_head) * k + key) * d + dim] as f64
                                })
                                .sum::<f64>();
                            let bias = if (distance as usize) < extent {
                                x[3][((batch * h + head) * q + query) * extent + distance as usize]
                                    as f64
                            } else {
                                0.
                            };
                            tau * (dot / d as f64 + bias)
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights = scores.iter().map(|x| (*x - max).exp()).collect::<Vec<_>>();
                    let denominator = weights.iter().sum::<f64>();
                    for dim in 0..d {
                        output.push(
                            (0..k)
                                .map(|key| {
                                    weights[key]
                                        * x[2][((batch * kv + key_head) * k + key) * d + dim] as f64
                                })
                                .sum::<f64>()
                                / denominator,
                        );
                    }
                }
            }
        }
        output
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_relative_workspace_bounds_cover_profiles_windows_and_large_integer_coordinates() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mechanisms = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = 0;
        for dims in [
            [1, 2, 1, 1, 17, 8, 5],
            [2, 4, 2, 3, 13, 5, 7],
            [1, 3, 1, 9, 21, 16, 4],
            [1, 2, 2, 1, 4097, 64, 17],
        ] {
            for origin in [7, (1 << 24) + 3, i32::MAX - dims[4]] {
                for window in [None, Some(1), Some(5)] {
                    for floor in [None, Some(2)] {
                        let op = relative_op(dims, origin, window, floor);
                        let allowed = allowance(&op, mechanisms);
                        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                            for strided in [false, true] {
                                let input = op
                                    .inputs
                                    .iter()
                                    .enumerate()
                                    .map(|(slot, l)| {
                                        floating(l.shape(), slot, dtype, strided, &stream)
                                    })
                                    .collect::<Vec<_>>();
                                safemlx::transforms::eval(
                                    input.iter().map(MlxTensor::as_array).collect::<Vec<_>>(),
                                )
                                .unwrap();
                                stream.synchronize().unwrap();
                                let before = safemlx::memory::active_memory().unwrap();
                                safemlx::memory::reset_peak_memory().unwrap();
                                let output =
                                    relative_execute::<MlxNeuralBackend>(&op, &input, &stream)
                                        .unwrap();
                                output.as_array().evaluated().unwrap();
                                stream.synchronize().unwrap();
                                let observed = safemlx::memory::peak_memory()
                                    .unwrap()
                                    .saturating_sub(before)
                                    as u64;
                                assert!(observed<=allowed,"{dims:?} origin={origin} window={window:?} floor={floor:?} {dtype:?} strided={strided}: {observed}>{allowed}");
                                let values = input
                                    .iter()
                                    .map(|x| x.to_f32_vec(&stream).unwrap())
                                    .collect::<Vec<_>>();
                                let actual = output.to_f32_vec(&stream).unwrap();
                                let expected = relative_reference(&op, &values);
                                let (atol, rtol) = if dtype == Dtype::Float32 {
                                    (3e-5, 3e-4)
                                } else {
                                    (3e-3, 0.04)
                                };
                                for (i, (actual, expected)) in
                                    actual.iter().zip(expected).enumerate()
                                {
                                    assert!((*actual as f64-expected).abs()<=atol+rtol*expected.abs(),"{dims:?} origin={origin} window={window:?} floor={floor:?} {dtype:?} strided={strided} element {i}: {actual}!={expected}");
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!("RELATIVE_WORKSPACE_NATIVE_CASES={cases}");
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_multi_axis_workspace_bounds_cover_layouts_empty_rows_and_saturated_positions() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mechanisms = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = 0;
        for kind in [
            MultiAxisRotaryLayout::IndependentAxes,
            MultiAxisRotaryLayout::SplitHalves,
            MultiAxisRotaryLayout::RoundRobinSections,
        ] {
            for axes in [&[2][..], &[4, 8], &[8, 4, 2]] {
                for leading in [&[5][..], &[2, 3], &[1, 9], &[0]] {
                    for unsigned in [false, true] {
                        for strided in [false, true] {
                            let spec = spec(axes, kind);
                            let op = rotary_op(
                                leading,
                                spec.clone(),
                                if unsigned {
                                    WorkspaceDtype::Uint32
                                } else {
                                    WorkspaceDtype::Int32
                                },
                            );
                            let data = (0..op.inputs[0].elements().unwrap())
                                .map(|i| ((i * 7) % 37) as i32 - if unsigned { 0 } else { 8 })
                                .collect::<Vec<_>>();
                            check_rotary(&op, &spec, &data, unsigned, strided, &stream, mechanisms);
                            cases += 1;
                        }
                    }
                }
            }
            let mut spec = spec(&[2, 2], kind);
            spec.base = 1.;
            spec.minimum_position = i32::MIN;
            spec.axes[0].position_offset = 13;
            spec.axes[1].position_offset = -13;
            let op = rotary_op(&[2], spec.clone(), WorkspaceDtype::Int32);
            check_rotary(
                &op,
                &spec,
                &[i32::MAX, i32::MIN, -17, 19],
                false,
                true,
                &stream,
                mechanisms,
            );
            cases += 1;
        }
        eprintln!("MULTI_AXIS_WORKSPACE_NATIVE_CASES={cases}");
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_zero_sections_preserve_distinct_axes_and_original_cold_bound() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mechanisms = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for dimensions in [
            [8, 0, 0],
            [0, 8, 0],
            [0, 0, 8],
            [0, 4, 4],
            [4, 0, 4],
            [4, 4, 0],
            [2, 2, 4],
            [4, 0, 14],
        ] {
            let policy = spec(&dimensions, MultiAxisRotaryLayout::RoundRobinSections);
            for unsigned in [false, true] {
                for strided in [false, true] {
                    let op = rotary_op(
                        &[2],
                        policy.clone(),
                        if unsigned {
                            WorkspaceDtype::Uint32
                        } else {
                            WorkspaceDtype::Int32
                        },
                    );
                    check_rotary(
                        &op,
                        &policy,
                        &[2, 5, 9, 3, 7, 11],
                        unsigned,
                        strided,
                        &stream,
                        mechanisms,
                    );
                }
            }
        }
    }

    fn check_rotary(
        op: &WorkspaceOperation,
        spec: &MultiAxisRotarySpec,
        data: &[i32],
        unsigned: bool,
        strided: bool,
        stream: &Stream,
        mechanisms: MlxMetalWorkspaceMechanisms,
    ) {
        let allowed = allowance(op, mechanisms);
        let input = input_positions(op.inputs[0].shape(), data, unsigned, strided, stream);
        input.as_array().evaluated().unwrap();
        stream.synchronize().unwrap();
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        let (cos, sin) = MlxTensor::multi_axis_rotary_embeddings(&input, spec, stream).unwrap();
        safemlx::transforms::eval([cos.as_array(), sin.as_array()]).unwrap();
        stream.synchronize().unwrap();
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        assert!(
            observed <= allowed,
            "{spec:?} shape={:?} unsigned={unsigned} strided={strided}: {observed}>{allowed}",
            input.shape()
        );
        let rows = data.len() / spec.axes.len();
        let expected = if rows == 0 {
            (vec![], vec![])
        } else {
            eredu_nn::multimodal::reference_multi_axis_rotary_embeddings(data, rows, spec).unwrap()
        };
        for (actual, expected) in [(&cos, expected.0), (&sin, expected.1)] {
            assert_eq!(actual.shape(), op.outputs[0].shape());
            if let Some(storage) = actual.as_array().allocation_info().unwrap() {
                let retained =
                    capacity(mechanisms.allocation(), op.outputs[0].elements().unwrap()).unwrap();
                assert!(
                    storage.bytes() as u64 <= retained,
                    "{spec:?} shape={:?}: output backing {} exceeds retained allowance {retained}",
                    actual.shape(),
                    storage.bytes()
                );
            }
            for (i, (a, b)) in actual
                .to_f32_vec(stream)
                .unwrap()
                .iter()
                .zip(expected)
                .enumerate()
            {
                assert!((*a-b).abs()<=2e-5,"{spec:?} shape={:?} unsigned={unsigned} strided={strided} element {i}: {a}!={b}",input.shape());
            }
        }
    }
}
