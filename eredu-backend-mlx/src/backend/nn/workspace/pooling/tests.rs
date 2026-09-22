use super::*;
use eredu_nn::{NeuralBackend, Tensor};

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

#[derive(Clone, Copy, Debug)]
enum Case {
    Pooled,
    Indexed,
    Positions,
    Gather,
}

fn operation(case: Case, dims: [i32; 7], masks: u32) -> WorkspaceOperation {
    let [b, h, q, l, p, d, s] = dims;
    let f = |shape: &[i32]| layout(shape, WorkspaceDtype::Float32);
    let local_mask = masks & 1 != 0;
    let pooled_mask = masks & 2 != 0;
    let sinks = masks & 4 != 0;
    let lm = layout(&[1, h, 1, l], WorkspaceDtype::Bool);
    let pm = layout(
        &[b, 1, q, if matches!(case, Case::Indexed) { s } else { p }],
        if masks & 8 != 0 {
            WorkspaceDtype::Float32
        } else {
            WorkspaceDtype::Bool
        },
    );
    let (kind, mut inputs, out) = match case {
        Case::Pooled => (
            WorkspaceOperationKind::PooledAttention {
                scale: 0.7,
                local_mask,
                pooled_mask,
                sinks,
            },
            vec![f(&[b, h, q, d]), f(&[b, l, d]), f(&[b, p, d])],
            f(&[b, h, q, d]),
        ),
        Case::Indexed => (
            WorkspaceOperationKind::IndexedAttention {
                scale: 0.7,
                local_mask,
                pooled_mask,
                sinks,
            },
            vec![
                f(&[b, h, q, d]),
                f(&[b, l, d]),
                f(&[b, l, d + 2]),
                f(&[b, p, d]),
                f(&[b, p, d + 2]),
                layout(&[b, q, s], WorkspaceDtype::Uint32),
            ],
            f(&[b, h, q, d + 2]),
        ),
        Case::Positions => (
            WorkspaceOperationKind::PooledPositions {
                scale: 0.7,
                head_scale: 1.2,
                top_k: s,
                masked: pooled_mask,
            },
            vec![f(&[b, h, q, d]), f(&[b, p, d]), f(&[b, q, h])],
            layout(&[b, q, s.min(p)], WorkspaceDtype::Uint32),
        ),
        Case::Gather => (
            WorkspaceOperationKind::GatherPooledMask,
            vec![
                layout(
                    &[q, p],
                    if masks & 8 != 0 {
                        WorkspaceDtype::Float32
                    } else {
                        WorkspaceDtype::Bool
                    },
                ),
                layout(&[b, q, s], WorkspaceDtype::Int32),
            ],
            layout(
                &[b, 1, q, s],
                if masks & 8 != 0 {
                    WorkspaceDtype::Float32
                } else {
                    WorkspaceDtype::Bool
                },
            ),
        ),
    };
    match case {
        Case::Pooled | Case::Indexed => {
            if local_mask {
                inputs.push(lm);
            }
            if pooled_mask {
                inputs.push(pm);
            }
            if sinks {
                inputs.push(f(&[h]));
            }
        }
        Case::Positions if pooled_mask => inputs.push(layout(&[b, q, p], WorkspaceDtype::Bool)),
        _ => {}
    }
    WorkspaceOperation {
        kind,
        inputs,
        outputs: vec![out],
    }
}

fn execute<B: NeuralBackend>(
    op: &WorkspaceOperation,
    values: &[B::Tensor],
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    use WorkspaceOperationKind as K;
    match op.kind {
        K::PooledAttention {
            scale,
            local_mask,
            pooled_mask,
            sinks,
        } => B::pooled_attention(
            eredu_nn::PooledAttentionInput {
                queries: &values[0],
                local: &values[1],
                pooled: &values[2],
                scale,
                local_mask: local_mask.then(|| &values[3]),
                pooled_mask: pooled_mask.then(|| &values[3 + usize::from(local_mask)]),
                sinks: sinks.then(|| values.last().unwrap()),
            },
            context,
        ),
        K::IndexedAttention {
            scale,
            local_mask,
            pooled_mask,
            sinks,
        } => B::indexed_attention(
            eredu_nn::IndexedAttentionInput {
                queries: &values[0],
                local_keys: &values[1],
                local_values: &values[2],
                pooled_keys: &values[3],
                pooled_values: &values[4],
                selected_positions: &values[5],
                scale,
                local_mask: local_mask.then(|| &values[6]),
                pooled_mask: pooled_mask.then(|| &values[6 + usize::from(local_mask)]),
                sinks: sinks.then(|| values.last().unwrap()),
            },
            context,
        ),
        K::PooledPositions {
            top_k,
            scale,
            head_scale,
            masked,
        } => B::select_pooled_positions(
            eredu_nn::PooledPositionInput {
                queries: &values[0],
                pooled_keys: &values[1],
                head_weights: &values[2],
                top_k,
                scale,
                head_scale,
                mask: masked.then(|| &values[3]),
            },
            context,
        ),
        K::GatherPooledMask => B::gather_pooled_mask(&values[0], &values[1], context),
        _ => unreachable!(),
    }
}

#[test]
fn pooled_workspace_prices_compound_paths_and_retains_full_partition_storage() {
    for case in [Case::Pooled, Case::Indexed, Case::Positions, Case::Gather] {
        for dims in [
            [2, 3, 5, 7, 13, 5, 3],
            [1, 2, 1, 31, 4097, 64, 7],
            [2, 4, 9, 13, 29, 64, 11],
        ] {
            for masks in [0, 1, 2, 3, 7, 10, 15] {
                let op = operation(case, dims, masks);
                let context = WorkspaceContext::new(mechanisms());
                let values = op
                    .inputs
                    .iter()
                    .map(|l| WorkspaceTensor::existing(l.clone(), &context).unwrap())
                    .collect::<Vec<_>>();
                let out = execute::<WorkspaceBackend>(&op, &values, &context).unwrap();
                let report = context.report(&[out]).unwrap();
                assert!(report.tensor_buffers.total_bytes.is_some(), "{op:?}");
                assert_eq!(report.host_workspace_bytes, Some(0));
                if matches!(case, Case::Positions) {
                    let expected = capacity(
                        mechanisms().allocation,
                        product(&[dims[0], dims[2], dims[4]]).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(report.tensor_buffers.retained_bytes, Some(expected));
                    assert!(expected > op.outputs[0].bytes().unwrap());
                }
            }
        }
    }
    let op = operation(Case::Positions, [2, 3, 5, 7, 0, 5, 3], 2);
    assert!(mechanisms().operation_bound(&op).unwrap().is_some());
}

#[test]
fn pooled_workspace_rejects_invalid_descriptors_and_preserves_unknown_dtypes() {
    for case in [Case::Pooled, Case::Indexed, Case::Positions, Case::Gather] {
        let op = operation(case, [2, 3, 5, 7, 13, 5, 3], 3);
        let mut bad = op.clone();
        bad.outputs[0] = layout(&[1], WorkspaceDtype::Float32);
        assert!(mechanisms().operation_bound(&bad).is_err());
        let mut bad = op.clone();
        bad.inputs.pop();
        assert!(mechanisms().operation_bound(&bad).is_err());
        if !matches!(case, Case::Gather) {
            let mut unsupported = op.clone();
            unsupported.inputs[0] = layout(op.inputs[0].shape(), WorkspaceDtype::Int32);
            assert!(
                mechanisms()
                    .operation_bound(&unsupported)
                    .unwrap()
                    .is_none()
            );
            assert!(
                mechanisms()
                    .host_workspace_bound(&unsupported)
                    .unwrap()
                    .is_none()
            );
        }
    }
    let mut op = operation(Case::Pooled, [2, 3, 5, 7, 13, 5, 3], 3);
    op.inputs[3] = layout(&[2, 3, 5, 8], WorkspaceDtype::Bool);
    assert!(mechanisms().operation_bound(&op).is_err());
    let mut op = operation(Case::Indexed, [2, 3, 5, 7, 13, 5, 3], 0);
    op.inputs[5] = layout(&[2, 5, 3], WorkspaceDtype::Float32);
    assert!(mechanisms().operation_bound(&op).is_err());
    let mut op = operation(Case::Positions, [2, 3, 5, 7, 13, 5, 3], 0);
    op.kind = WorkspaceOperationKind::PooledPositions {
        top_k: 3,
        scale: f32::NAN,
        head_scale: 1.,
        masked: false,
    };
    assert!(mechanisms().operation_bound(&op).is_err());
    let op = operation(Case::Indexed, [1, i32::MAX, 2, 1, 1, 1, 1], 0);
    assert!(mechanisms().operation_bound(&op).is_err());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{MlxTensor, backend::nn::shared::MlxNeuralBackend};
    use safemlx::{
        Array, Device, DeviceType, Dtype, Stream,
        ops::indexing::{IntoStrideBy, TryIndexOp},
    };

    fn fixtures(
        op: &WorkspaceOperation,
        dims: [i32; 7],
        dtype: Dtype,
        strided: bool,
        stream: &Stream,
    ) -> Vec<MlxTensor> {
        let p = dims[4];
        op.inputs
            .iter()
            .enumerate()
            .map(|(slot, l)| {
                let values = (0..l.elements().unwrap() as usize)
                    .map(|i| match l.dtype() {
                        WorkspaceDtype::Bool => {
                            if i % 5 != 3 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32 => {
                            ((i * 7 + 2) % p.max(1) as usize) as f32
                        }
                        _ => ((i * 13 + slot * 7) % 97) as f32 / 128. - 0.3,
                    })
                    .collect::<Vec<_>>();
                let stored = if strided {
                    values.iter().flat_map(|v| [*v, 0.25]).collect::<Vec<_>>()
                } else {
                    values
                };
                let n = stored.len() as i32;
                let mut array = match l.dtype() {
                    WorkspaceDtype::Bool => Array::from_slice(
                        &stored.iter().map(|v| *v != 0.).collect::<Vec<_>>(),
                        &[n],
                    ),
                    WorkspaceDtype::Int32 => Array::from_slice(
                        &stored.iter().map(|v| *v as i32).collect::<Vec<_>>(),
                        &[n],
                    ),
                    WorkspaceDtype::Uint32 => Array::from_slice(
                        &stored.iter().map(|v| *v as u32).collect::<Vec<_>>(),
                        &[n],
                    ),
                    _ => Array::from_slice(&stored, &[n])
                        .as_dtype(dtype, stream)
                        .unwrap(),
                };
                if strided {
                    array = array.try_index_device((..).stride_by(2), stream).unwrap();
                }
                MlxTensor::from_array(array.reshape(l.shape(), stream).unwrap())
            })
            .collect()
    }
    fn mask_value(layout: &WorkspaceLayout, values: &[f32], at: &[usize]) -> f64 {
        let leading = at.len() - layout.shape().len();
        let mut index = 0;
        for (axis, &extent) in layout.shape().iter().enumerate() {
            index = index * extent as usize + if extent == 1 { 0 } else { at[leading + axis] };
        }
        if layout.dtype() == WorkspaceDtype::Bool {
            if values[index] != 0. {
                0.
            } else {
                f64::NEG_INFINITY
            }
        } else {
            values[index] as f64
        }
    }
    fn attention_reference(
        op: &WorkspaceOperation,
        x: &[eredu_core::HostTensorBuffer<f32>],
    ) -> Vec<f64> {
        let (indexed, scale, lm, pm, sinks) = match op.kind {
            WorkspaceOperationKind::PooledAttention {
                scale,
                local_mask,
                pooled_mask,
                sinks,
            } => (false, scale, local_mask, pooled_mask, sinks),
            WorkspaceOperationKind::IndexedAttention {
                scale,
                local_mask,
                pooled_mask,
                sinks,
            } => (true, scale, local_mask, pooled_mask, sinks),
            _ => unreachable!(),
        };
        let qshape = op.inputs[0].shape();
        let (b, h, q, d) = (
            qshape[0] as usize,
            qshape[1] as usize,
            qshape[2] as usize,
            qshape[3] as usize,
        );
        let pk = if indexed { 3 } else { 2 };
        let lv = if indexed { 2 } else { 1 };
        let pv = if indexed { 4 } else { 2 };
        let l = op.inputs[1].shape()[1] as usize;
        let p = op.inputs[pk].shape()[1] as usize;
        let s = if indexed {
            op.inputs[5].shape()[2] as usize
        } else {
            p
        };
        let v = op.inputs[lv].shape()[2] as usize;
        let mask_start = if indexed { 6 } else { 3 };
        let mut output = Vec::new();
        for batch in 0..b {
            for head in 0..h {
                for query in 0..q {
                    let mut scores = Vec::new();
                    let mut values = Vec::new();
                    for token in 0..l + s {
                        let (key_slot, value_slot, index, extent, mask) = if token < l {
                            (1, lv, token, l, lm.then_some(mask_start))
                        } else {
                            let index = if indexed {
                                x[5][(batch * q + query) * s + token - l] as usize
                            } else {
                                token - l
                            };
                            (pk, pv, index, p, pm.then_some(mask_start + usize::from(lm)))
                        };
                        let mut score = (0..d)
                            .map(|dim| {
                                x[0][((batch * h + head) * q + query) * d + dim] as f64
                                    * x[key_slot][(batch * extent + index) * d + dim] as f64
                            })
                            .sum::<f64>()
                            * scale as f64;
                        if let Some(slot) = mask {
                            score += mask_value(
                                &op.inputs[slot],
                                &x[slot],
                                &[
                                    batch,
                                    head,
                                    query,
                                    if token < l { token } else { token - l },
                                ],
                            );
                        }
                        scores.push(score);
                        values.push(
                            &x[value_slot]
                                [(batch * extent + index) * v..(batch * extent + index + 1) * v],
                        );
                    }
                    if sinks {
                        scores.push(x.last().unwrap()[head] as f64);
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exp = scores.iter().map(|s| (*s - max).exp()).collect::<Vec<_>>();
                    let denominator = exp.iter().sum::<f64>();
                    for dim in 0..v {
                        output.push(
                            values
                                .iter()
                                .zip(&exp)
                                .map(|(value, p)| value[dim] as f64 * p)
                                .sum::<f64>()
                                / denominator,
                        );
                    }
                }
            }
        }
        output
    }
    fn check_positions(
        op: &WorkspaceOperation,
        x: &[eredu_core::HostTensorBuffer<f32>],
        actual: &[f32],
    ) {
        let WorkspaceOperationKind::PooledPositions {
            top_k,
            scale,
            head_scale,
            masked,
        } = op.kind
        else {
            unreachable!()
        };
        let qshape = op.inputs[0].shape();
        let (b, h, q, d) = (
            qshape[0] as usize,
            qshape[1] as usize,
            qshape[2] as usize,
            qshape[3] as usize,
        );
        let p = op.inputs[1].shape()[1] as usize;
        let k = (top_k as usize).min(p);
        if p == 0 {
            assert!(actual.is_empty());
            return;
        }
        for batch in 0..b {
            for query in 0..q {
                let scores = (0..p)
                    .map(|pooled| {
                        if masked
                            && mask_value(&op.inputs[3], &x[3], &[batch, query, pooled])
                                .is_infinite()
                        {
                            return f64::NEG_INFINITY;
                        }
                        (0..h)
                            .map(|head| {
                                let dot = (0..d)
                                    .map(|dim| {
                                        x[0][((batch * h + head) * q + query) * d + dim] as f64
                                            * x[1][(batch * p + pooled) * d + dim] as f64
                                    })
                                    .sum::<f64>();
                                dot.max(0.)
                                    * scale as f64
                                    * x[2][(batch * q + query) * h + head] as f64
                                    * head_scale as f64
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let mut sorted = scores.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let chosen = &actual[(batch * q + query) * k..(batch * q + query + 1) * k];
                let unique = chosen
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(unique.len(), k);
                for index in unique {
                    assert!(
                        index < p && scores[index] + 1e-5 >= sorted[k - 1],
                        "selected {index} below top-k threshold"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_pooled_mechanisms_fit_cold_bounds_and_independent_equations() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut checked = 0;
        for case in [Case::Pooled, Case::Indexed, Case::Positions, Case::Gather] {
            for dims in [
                [2, 3, 5, 7, 13, 5, 3],
                [1, 2, 1, 31, 4097, 64, 7],
                [2, 4, 9, 13, 29, 64, 11],
                [1, 2, 3, 0, 17, 8, 2],
                [2, 2, 3, 7, 0, 8, 3],
            ] {
                if dims[4] == 0 && !matches!(case, Case::Pooled | Case::Positions) {
                    continue;
                }
                for masks in [0, 1, 2, 7, 10, 15] {
                    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                        for strided in [false, true] {
                            let op = operation(case, dims, masks);
                            let bound = selected.operation_bound(&op).unwrap().unwrap();
                            if matches!(case, Case::Pooled) {
                                assert!(
                                    selected
                                        .ordinary_call_controls(op.as_view())
                                        .unwrap()
                                        .is_some()
                                );
                            }
                            assert_eq!(
                                selected.host_workspace_bound(&op).unwrap().unwrap().bytes,
                                0
                            );
                            let retained = match bound.outputs[0] {
                                WorkspaceOutputStorage::Allocate(n)
                                | WorkspaceOutputStorage::AllocateOrAliasInputs {
                                    bytes: n, ..
                                } => n,
                                _ => unreachable!(),
                            };
                            let allowed = retained + bound.scratch_bytes;
                            let input = fixtures(&op, dims, dtype, strided, &stream);
                            safemlx::transforms::eval(
                                input.iter().map(MlxTensor::as_array).collect::<Vec<_>>(),
                            )
                            .unwrap();
                            stream.synchronize().unwrap();
                            let before = safemlx::memory::active_memory().unwrap();
                            safemlx::memory::reset_peak_memory().unwrap();
                            let out = execute::<MlxNeuralBackend>(&op, &input, &stream)
                                .unwrap_or_else(|error| panic!("{case:?} {dims:?} masks={masks} {dtype:?} strided={strided}: {error}"));
                            out.as_array().evaluated().unwrap();
                            stream.synchronize().unwrap();
                            let observed = safemlx::memory::peak_memory()
                                .unwrap()
                                .saturating_sub(before)
                                as u64;
                            assert!(
                                observed <= allowed,
                                "{case:?} {dims:?} masks={masks} {dtype:?} strided={strided}: {observed} > {allowed}"
                            );
                            assert_eq!(out.shape(), op.outputs[0].shape());
                            if matches!(case, Case::Positions) && dims[4] > 0 {
                                let backing = out.as_array().allocation_info().unwrap().unwrap();
                                let logical_partition_bytes = u64::try_from(dims[0]).unwrap()
                                    * dims[2] as u64
                                    * dims[4] as u64
                                    * 4;
                                assert!(backing.bytes() as u64 >= logical_partition_bytes);
                                assert!(backing.bytes() as u64 <= retained);
                            }
                            let actual = out.to_f32_vec(&stream).unwrap();
                            let values = input
                                .iter()
                                .map(|x| x.to_f32_vec(&stream).unwrap())
                                .collect::<Vec<_>>();
                            match case {
                                Case::Pooled | Case::Indexed => {
                                    let expected = attention_reference(&op, &values);
                                    let (atol, rtol) = if dtype == Dtype::Float32 {
                                        (2e-5, 2e-4)
                                    } else {
                                        (4e-3, 0.05)
                                    };
                                    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                                        assert!(
                                            (*a as f64 - e).abs() <= atol + rtol * e.abs(),
                                            "{case:?} {dims:?} masks={masks} {dtype:?} strided={strided} element {i}: {a} != {e}"
                                        );
                                    }
                                }
                                Case::Positions => check_positions(&op, &values, &actual),
                                Case::Gather => {
                                    let [b, _, q, _, p, _, s] = dims;
                                    let expected = (0..b * q * s)
                                        .map(|i| {
                                            values[0][((i / s) % q * p) as usize
                                                + values[1][i as usize] as usize]
                                        })
                                        .collect::<Vec<_>>();
                                    assert_eq!(actual, expected);
                                }
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
        eprintln!("POOLED_WORKSPACE_NATIVE_CASES={checked}");
    }
}

#[test]
fn ordinary_metal_pooled_positions_preserves_mask_empty_and_precision_sources() {
    use eredu_nn::workspace::{WorkspaceFloatingType, WorkspaceRepresentation};
    let mechanism = mechanisms();
    for pooled in [0, 7, 4097] {
        for masked in [false, true] {
            let mut unknown = operation(
                Case::Positions,
                [2, 3, 5, 7, pooled, 8, 2],
                if masked { 2 } else { 0 },
            );
            for value in &mut unknown.inputs[..3] {
                *value = value.clone().with_representation(None);
            }
            let quoted = mechanism
                .ordinary_call_controls(unknown.as_view())
                .unwrap()
                .unwrap();
            assert!(quoted.metadata_bytes > 0);
            for dtype in [
                WorkspaceFloatingType::Float32,
                WorkspaceFloatingType::Float16,
                WorkspaceFloatingType::Bfloat16,
            ] {
                let mut known = unknown.clone();
                for value in &mut known.inputs[..3] {
                    *value = value
                        .clone()
                        .with_representation(Some(WorkspaceRepresentation::new(dtype, false)));
                }
                assert_eq!(
                    mechanism.ordinary_call_controls(known.as_view()).unwrap(),
                    Some(quoted)
                );
            }
            assert!(
                unknown.inputs[..3]
                    .iter()
                    .all(|value| value.representation().is_none())
            );
            let mut changed = unknown.clone();
            changed.inputs[0] = layout(unknown.inputs[0].shape(), WorkspaceDtype::Int32);
            assert!(
                mechanism
                    .ordinary_call_controls(changed.as_view())
                    .unwrap()
                    .is_none()
            );
            if masked {
                changed = unknown;
                changed.inputs[3] = layout(changed.inputs[3].shape(), WorkspaceDtype::Float32);
                assert!(
                    mechanism
                        .ordinary_call_controls(changed.as_view())
                        .unwrap()
                        .is_none()
                );
            }
        }
    }
}
