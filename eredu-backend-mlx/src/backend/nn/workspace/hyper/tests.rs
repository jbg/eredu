use super::*;
use eredu_nn::{
    HyperConnectionOperator, HyperConnectionSpec, HyperHeadOperator, HyperHeadSpec,
    HyperNeuralBackend, ParameterSpec, Tensor,
};

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
fn connection(s: i32, h: i32, iterations: usize) -> HyperConnectionSpec {
    HyperConnectionSpec {
        streams: s,
        hidden_size: h,
        sinkhorn_iterations: iterations,
        epsilon: 1e-6,
        function: ParameterSpec::trainable("mix.function").unwrap(),
        base: ParameterSpec::trainable("mix.base").unwrap(),
        scale: ParameterSpec::trainable("mix.scale").unwrap(),
    }
}
fn head(s: i32, h: i32) -> HyperHeadSpec {
    HyperHeadSpec {
        streams: s,
        hidden_size: h,
        norm_epsilon: 1e-5,
        epsilon: 1e-6,
        function: ParameterSpec::trainable("head.function").unwrap(),
        base: ParameterSpec::trainable("head.base").unwrap(),
        scale: ParameterSpec::trainable("head.scale").unwrap(),
    }
}
fn layout(s: &[i32]) -> WorkspaceLayout {
    WorkspaceLayout::new(s, WorkspaceDtype::Float32).unwrap()
}
fn operations([b, t, s, h]: [i32; 4], iterations: usize) -> [WorkspaceOperation; 4] {
    let residual = layout(&[b, t, s, h]);
    let collapsed = layout(&[b, t, h]);
    let vector = layout(&[b, t, s]);
    let matrix = layout(&[b, t, s, s]);
    let width = (s + 2) * s;
    [
        WorkspaceOperation {
            kind: WorkspaceOperationKind::HyperCollapse(
                Box::new(connection(s, h, iterations)),
                1e-5,
            ),
            inputs: vec![
                residual.clone(),
                layout(&[width, s * h]),
                layout(&[width]),
                layout(&[3]),
            ],
            outputs: vec![
                collapsed.clone(),
                vector.clone(),
                vector.clone(),
                matrix.clone(),
            ],
        },
        WorkspaceOperation {
            kind: WorkspaceOperationKind::HyperExpand,
            inputs: vec![collapsed.clone(), residual.clone(), vector.clone(), matrix],
            outputs: vec![residual.clone()],
        },
        WorkspaceOperation {
            kind: WorkspaceOperationKind::HyperHeadCoefficients(Box::new(head(s, h))),
            inputs: vec![
                residual.clone(),
                layout(&[s, s * h]),
                layout(&[s]),
                layout(&[1]),
            ],
            outputs: vec![vector.clone()],
        },
        WorkspaceOperation {
            kind: WorkspaceOperationKind::HyperHeadSum,
            inputs: vec![residual, vector],
            outputs: vec![collapsed],
        },
    ]
}
fn total(op: &WorkspaceOperation, m: MlxMetalWorkspaceMechanisms) -> u64 {
    let bound = m.operation_bound(op).unwrap().unwrap();
    bound.scratch_bytes
        + bound
            .outputs
            .iter()
            .map(|o| match o {
                WorkspaceOutputStorage::Allocate(b) => *b,
                _ => panic!("unexpected hyper backing"),
            })
            .sum::<u64>()
}

#[test]
fn hyper_workspace_prices_residual_cycles_and_retained_coefficients() {
    for dims in [
        [1, 1, 1, 7],
        [2, 3, 3, 5],
        [1, 9, 4, 16],
        [1, 1, 8, 1025],
        [2, 0, 3, 7],
    ] {
        let [b, t, s, h] = dims;
        let mut previous = 0;
        for iterations in [1, 3, 20] {
            let ops = operations(dims, iterations);
            for operation in &ops {
                let population = super::structure(operation.as_view()).unwrap().unwrap();
                assert!(population.controls > 0);
                if t == 0 && matches!(operation.kind, WorkspaceOperationKind::HyperCollapse(..)) {
                    assert!(
                        population.aliases > 0,
                        "empty normalization retains exact aliases"
                    );
                }
            }
            let current = total(&ops[0], mechanisms());
            assert!(current > previous);
            previous = current;
            let context = WorkspaceContext::new(mechanisms());
            let mut mix =
                WorkspaceBackend::hyper_connection(connection(s, h, iterations), &context).unwrap();
            let mut output = WorkspaceBackend::hyper_head(head(s, h), &context).unwrap();
            let mut residual = WorkspaceTensor::existing(layout(&dims), &context).unwrap();
            for _ in 0..2 {
                let state = mix.collapse(&residual, 1e-5, &context).unwrap();
                let retained = context
                    .report(&[
                        state.collapsed.clone(),
                        state.pre.clone(),
                        state.post.clone(),
                        state.combination.clone(),
                    ])
                    .unwrap();
                assert_eq!(
                    retained.tensor_buffers.retained_bytes,
                    Some(
                        ops[0]
                            .outputs
                            .iter()
                            .map(|l| capacity(mechanisms().allocation, l.elements().unwrap())
                                .unwrap())
                            .sum()
                    )
                );
                assert_eq!(retained.host_workspace_bytes, Some(0));
                residual = mix
                    .expand(
                        &state.collapsed.square(&context).unwrap(),
                        &residual,
                        &state,
                        &context,
                    )
                    .unwrap();
            }
            let result = output.forward(&residual, &context).unwrap();
            assert_eq!(result.shape(), [b, t, h]);
            let report = context.report(&[result]).unwrap();
            assert!(report.total_bytes.is_some());
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
        }
    }
}

#[test]
fn hyper_workspace_rejects_invalid_descriptors_and_closes_large_pass_counts() {
    for mut op in operations([2, 3, 4, 5], 3) {
        let saved = op.outputs[0].clone();
        op.outputs[0] = layout(&[1]);
        assert!(mechanisms().operation_bound(&op).is_err());
        op.outputs[0] = saved;
        op.inputs[0] = WorkspaceLayout::new(op.inputs[0].shape(), WorkspaceDtype::Int32).unwrap();
        assert!(mechanisms().operation_bound(&op).unwrap().is_none());
        assert!(mechanisms().host_workspace_bound(&op).unwrap().is_none());
    }
    let mut op = operations([2, 3, 4, 5], 3)[0].clone();
    let WorkspaceOperationKind::HyperCollapse(ref mut spec, _) = op.kind else {
        unreachable!()
    };
    spec.sinkhorn_iterations = usize::MAX;
    assert!(mechanisms().operation_bound(&op).is_err());
    assert!(super::structure(op.as_view()).is_err());
    let WorkspaceOperationKind::HyperCollapse(ref mut spec, ref mut epsilon) = op.kind else {
        unreachable!()
    };
    spec.sinkhorn_iterations = 1 << 30;
    *epsilon = 1e-5;
    assert!(mechanisms().operation_bound(&op).unwrap().is_some());
    let population = super::structure(op.as_view()).unwrap().unwrap();
    assert!(population.primitives > 1 << 30);
    assert!(population.controls > 0);
    let WorkspaceOperationKind::HyperCollapse(_, ref mut epsilon) = op.kind else {
        unreachable!()
    };
    *epsilon = f32::NAN;
    assert!(mechanisms().operation_bound(&op).is_err());
    let mut op = operations([2, 3, 4, 5], 3)[0].clone();
    op.inputs[1] = layout(&[24, 19]);
    assert!(mechanisms().operation_bound(&op).is_err());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{
        backend::nn::hyper_connections::{expand, HyperConnection, HyperHead},
        MlxTensor,
    };
    use safemlx::{
        error::Exception,
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };

    fn floating(
        l: &WorkspaceLayout,
        slot: usize,
        dtype: Dtype,
        strided: bool,
        stream: &Stream,
    ) -> Array {
        let data = (0..l.elements().unwrap() as usize)
            .map(|i| ((i * 11 + slot * 5) % 37) as f32 / 64. - 0.25)
            .collect::<Vec<_>>();
        let data = if strided {
            data.iter().flat_map(|x| [*x, 0.125]).collect()
        } else {
            data
        };
        let mut array = Array::from_slice(&data, &[data.len() as i32])
            .as_dtype(dtype, stream)
            .unwrap();
        if strided {
            array = array.try_index_device((..).stride_by(2), stream).unwrap();
        }
        array.reshape(l.shape(), stream).unwrap()
    }
    fn values(a: &Array, stream: &Stream) -> eredu_core::HostTensorBuffer<f32> {
        MlxTensor::from_array(a.clone()).to_f32_vec(stream).unwrap()
    }
    fn measure(
        ops: &[WorkspaceOperation],
        inputs: &[&Array],
        stream: &Stream,
        m: MlxMetalWorkspaceMechanisms,
        run: impl FnOnce() -> Result<Vec<Array>, Exception>,
    ) -> Vec<Array> {
        eprintln!(
            "HYPER_NATIVE_PHASE={}",
            match ops[0].kind {
                WorkspaceOperationKind::HyperCollapse(..) => "collapse",
                WorkspaceOperationKind::HyperExpand => "expand",
                _ => "head",
            }
        );
        safemlx::transforms::eval(inputs.iter().copied()).unwrap();
        stream.synchronize().unwrap();
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        let result = run().unwrap();
        safemlx::transforms::eval(result.iter()).unwrap();
        stream.synchronize().unwrap();
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        let allowed = ops.iter().map(|op| total(op, m)).sum::<u64>();
        assert!(
            observed <= allowed,
            "{:?}: {observed}>{allowed}",
            ops.iter().map(|o| &o.kind).collect::<Vec<_>>()
        );
        let outputs = ops
            .iter()
            .flat_map(|op| op.outputs.iter())
            .collect::<Vec<_>>();
        assert_eq!(result.len(), outputs.len());
        for (value, layout) in result.iter().zip(outputs) {
            assert_eq!(value.shape(), layout.shape());
            let info = value.allocation_info().unwrap();
            assert!(info.is_some() || value.size() == 0);
            if let Some(info) = info {
                let capacity = capacity(m.allocation(), layout.elements().unwrap()).unwrap();
                assert!(
                    info.bytes() as u64 <= capacity,
                    "{:?} backing {}>{capacity}",
                    value.shape(),
                    info.bytes()
                );
            }
        }
        result
    }
    fn compare(actual: &Array, expected: &[f64], dtype: Dtype, stream: &Stream) {
        let actual = values(actual, stream);
        assert_eq!(actual.len(), expected.len());
        let (atol, rtol) = if dtype == Dtype::Float32 {
            (5e-5, 5e-4)
        } else {
            (3e-3, 0.03)
        };
        for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual as f64 - expected).abs() <= atol + rtol * expected.abs(),
                "{dtype:?} element {i}: {actual}!={expected}"
            );
        }
    }
    fn logits(
        x: &[f32],
        weight: &[f32],
        base: &[f32],
        rows: usize,
        width: usize,
        outputs: usize,
    ) -> Vec<f64> {
        assert_eq!(base.len(), outputs);
        (0..rows)
            .flat_map(|row| {
                let input = &x[row * width..(row + 1) * width];
                let inverse =
                    (input.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / width as f64 + 1e-5)
                        .sqrt()
                        .recip();
                (0..outputs).map(move |output| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| v as f64 * inverse * weight[output * width + i] as f64)
                        .sum()
                })
            })
            .collect()
    }
    fn split_reference(
        x: &[f32],
        weight: &[f32],
        base: &[f32],
        scale: &[f32],
        rows: usize,
        s: usize,
        h: usize,
        iterations: usize,
    ) -> [Vec<f64>; 4] {
        let width = s * (s + 2);
        let logit = logits(x, weight, base, rows, s * h, width);
        let sigmoid = |x: f64| 1. / (1. + (-x).exp());
        let mut pre = vec![];
        let mut post = vec![];
        let mut combination = vec![];
        let mut collapsed = vec![];
        for row in 0..rows {
            let v = &logit[row * width..(row + 1) * width];
            let p = (0..s)
                .map(|i| sigmoid(v[i] * scale[0] as f64 + base[i] as f64) + 1e-6)
                .collect::<Vec<_>>();
            pre.extend(&p);
            post.extend(
                (0..s).map(|i| 2. * sigmoid(v[s + i] * scale[1] as f64 + base[s + i] as f64)),
            );
            let mut matrix = (0..s * s)
                .map(|i| v[2 * s + i] * scale[2] as f64 + base[2 * s + i] as f64)
                .collect::<Vec<_>>();
            for i in 0..s {
                let max = matrix[i * s..(i + 1) * s]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum = matrix[i * s..(i + 1) * s]
                    .iter()
                    .map(|x| (*x - max).exp())
                    .sum::<f64>();
                for j in 0..s {
                    matrix[i * s + j] = (matrix[i * s + j] - max).exp() / sum + 1e-6;
                }
            }
            for pass in 0..2 * iterations - 1 {
                for i in 0..s {
                    let index = |j| if pass % 2 == 0 { j * s + i } else { i * s + j };
                    let divisor = (0..s).map(|j| matrix[index(j)]).sum::<f64>() + 1e-6;
                    for j in 0..s {
                        matrix[index(j)] /= divisor;
                    }
                }
            }
            combination.extend(matrix);
            collapsed.extend((0..h).map(|d| {
                (0..s)
                    .map(|i| p[i] * x[(row * s + i) * h + d] as f64)
                    .sum::<f64>()
            }));
        }
        [collapsed, pre, post, combination]
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_hyper_workspace_bounds_match_scalar_residual_cycles() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let m = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = 0;
        for dims in [
            [1, 1, 1, 7],
            [2, 3, 3, 5],
            [1, 9, 4, 16],
            [1, 1, 8, 1025],
            [2, 0, 3, 7],
        ] {
            let [b, t, s, h] = dims;
            let (rows, su, hu) = ((b * t) as usize, s as usize, h as usize);
            for iterations in [1, 3, 20] {
                let ops = operations(dims, iterations);
                for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                    for strided in [false, true] {
                        for low_parameters in [false, true] {
                            eprintln!("HYPER_NATIVE_CASE={cases} dims={dims:?} iterations={iterations} dtype={dtype:?} strided={strided} low_parameters={low_parameters}");
                            let ptype = if low_parameters {
                                dtype
                            } else {
                                Dtype::Float32
                            };
                            let inputs = ops[0]
                                .inputs
                                .iter()
                                .enumerate()
                                .map(|(i, l)| {
                                    floating(
                                        l,
                                        i,
                                        if i == 0 { dtype } else { ptype },
                                        strided,
                                        &stream,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let mut module = HyperConnection {
                                streams: s,
                                hidden_size: h,
                                iterations,
                                epsilon: 1e-6,
                                function: inputs[1].clone().into(),
                                base: inputs[2].clone().into(),
                                scale: inputs[3].clone().into(),
                            };
                            let split = measure(
                                &ops[..1],
                                &inputs.iter().collect::<Vec<_>>(),
                                &stream,
                                m,
                                || {
                                    let (out, split) =
                                        module.collapse_split(&inputs[0], 1e-5, &stream)?;
                                    Ok(vec![out, split.pre, split.post, split.combination])
                                },
                            );
                            let data = inputs
                                .iter()
                                .map(|a| values(a, &stream))
                                .collect::<Vec<_>>();
                            let expected = split_reference(
                                &data[0], &data[1], &data[2], &data[3], rows, su, hu, iterations,
                            );
                            for (i, (actual, expected)) in split.iter().zip(&expected).enumerate() {
                                compare(
                                    actual,
                                    expected,
                                    if i == 0 { dtype } else { Dtype::Float32 },
                                    &stream,
                                );
                            }
                            let sublayer = floating(&ops[1].inputs[0], 7, dtype, strided, &stream);
                            let expanded = measure(
                                &ops[1..2],
                                &[&sublayer, &inputs[0], &split[2], &split[3]],
                                &stream,
                                m,
                                || {
                                    Ok(vec![expand(
                                        &sublayer, &inputs[0], &split[2], &split[3], &stream,
                                    )?])
                                },
                            );
                            let (sub, post, comb) = (
                                values(&sublayer, &stream),
                                values(&split[2], &stream),
                                values(&split[3], &stream),
                            );
                            let mut expected = vec![];
                            for row in 0..rows {
                                for i in 0..su {
                                    for d in 0..hu {
                                        expected.push(
                                            post[row * su + i] as f64 * sub[row * hu + d] as f64
                                                + (0..su)
                                                    .map(|j| {
                                                        comb[(row * su + j) * su + i] as f64
                                                            * data[0][(row * su + j) * hu + d]
                                                                as f64
                                                    })
                                                    .sum::<f64>(),
                                        );
                                    }
                                }
                            }
                            compare(&expanded[0], &expected, dtype, &stream);
                            let head_input = ops[2]
                                .inputs
                                .iter()
                                .enumerate()
                                .map(|(i, l)| {
                                    floating(
                                        l,
                                        i + 11,
                                        if i == 0 { dtype } else { ptype },
                                        strided,
                                        &stream,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let mut head = HyperHead {
                                streams: s,
                                hidden_size: h,
                                norm_epsilon: 1e-5,
                                epsilon: 1e-6,
                                function: head_input[1].clone().into(),
                                base: head_input[2].clone().into(),
                                scale: head_input[3].clone().into(),
                            };
                            let result = measure(
                                &ops[2..],
                                &head_input.iter().collect::<Vec<_>>(),
                                &stream,
                                m,
                                || {
                                    let mut coefficients = None;
                                    let out = head.forward_with_coefficients_observer(
                                        &head_input[0],
                                        &stream,
                                        Some(&mut |value| {
                                            coefficients = Some(value.clone());
                                            Ok(())
                                        }),
                                    )?;
                                    Ok(vec![coefficients.unwrap(), out])
                                },
                            );
                            let data = head_input
                                .iter()
                                .map(|a| values(a, &stream))
                                .collect::<Vec<_>>();
                            let logit = logits(&data[0], &data[1], &data[2], rows, su * hu, su);
                            let coeff = logit
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| {
                                    1. / (1.
                                        + (-(v * data[3][0] as f64 + data[2][i % su] as f64)).exp())
                                        + 1e-6
                                })
                                .collect::<Vec<_>>();
                            compare(&result[0], &coeff, Dtype::Float32, &stream);
                            let mut expected = vec![];
                            for row in 0..rows {
                                for d in 0..hu {
                                    expected.push(
                                        (0..su)
                                            .map(|i| {
                                                coeff[row * su + i]
                                                    * data[0][(row * su + i) * hu + d] as f64
                                            })
                                            .sum::<f64>(),
                                    );
                                }
                            }
                            compare(&result[1], &expected, dtype, &stream);
                            cases += 1;
                        }
                    }
                }
            }
        }
        eprintln!("HYPER_WORKSPACE_NATIVE_CASES={cases}");
    }
}
