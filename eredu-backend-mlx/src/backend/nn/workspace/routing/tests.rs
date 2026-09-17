use super::*;
use eredu_nn::{
    GroupedNeuralBackend, JointGroupSelection, JointGroupSelectionInput, JointGroupSelectionSpec,
    Tensor,
};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn operation(
    leading: &[i32],
    d: i32,
    p: i32,
    shared: i32,
    k: i32,
    scale: f32,
) -> WorkspaceOperation {
    let layout = |shape: &[i32], dtype| WorkspaceLayout::new(shape, dtype).unwrap();
    let mut hidden = leading.to_vec();
    hidden.push(d);
    let rows = leading.iter().product::<i32>();
    WorkspaceOperation {
        kind: WorkspaceOperationKind::JointGroupSelection(
            JointGroupSelectionSpec::new(p, shared, k, scale).unwrap(),
        ),
        inputs: vec![
            layout(&hidden, WorkspaceDtype::Float32),
            layout(&[p + shared, d], WorkspaceDtype::Float32),
            layout(&[p], WorkspaceDtype::Float32),
            layout(&[1], WorkspaceDtype::Float32),
        ],
        outputs: vec![
            layout(&[rows, k], WorkspaceDtype::Uint32),
            layout(&[rows, k], WorkspaceDtype::Float32),
            layout(&[rows, shared], WorkspaceDtype::Float32),
        ],
    }
}
fn execute<B: GroupedNeuralBackend>(
    op: &WorkspaceOperation,
    x: &[B::Tensor],
    context: &<B::Tensor as Tensor>::Context,
) -> Result<JointGroupSelection<B::Tensor>, Error> {
    let WorkspaceOperationKind::JointGroupSelection(spec) = op.kind else {
        unreachable!()
    };
    B::joint_group_selection(
        JointGroupSelectionInput::new(&x[0], &x[1], &x[2], &x[3], spec)?,
        context,
    )
}

#[test]
fn joint_routing_workspace_retains_partition_and_shared_coefficient_backing_once() {
    for (leading, d, p, shared, k) in [
        (&[1][..], 7, 4, 1, 2),
        (&[2, 3][..], 8, 8, 3, 3),
        (&[2, 1, 2][..], 5, 7, 2, 7),
        (&[1][..], 64, 4097, 5, 1),
        (&[0][..], 3, 4, 2, 2),
    ] {
        let op = operation(leading, d, p, shared, k, 1.7);
        let context = WorkspaceContext::new(mechanisms());
        let x = op
            .inputs
            .iter()
            .map(|l| WorkspaceTensor::existing(l.clone(), &context).unwrap())
            .collect::<Vec<_>>();
        let result = execute::<WorkspaceBackend>(&op, &x, &context).unwrap();
        let retained = context
            .report(&[
                result.primary_indices().clone(),
                result.primary_coefficients().clone(),
                result.always_on_coefficients().clone(),
            ])
            .unwrap();
        let rows = op.outputs[0].shape()[0] as u64;
        let indices = capacity(mechanisms().allocation, mul(rows, p as u64).unwrap()).unwrap();
        let coefficients = capacity(
            mechanisms().allocation,
            mul(rows, (k + shared) as u64).unwrap(),
        )
        .unwrap();
        assert_eq!(
            retained.tensor_buffers.retained_bytes,
            Some(indices + coefficients)
        );
        assert_eq!(retained.host_workspace_bytes, Some(0));
        let shared = result.always_on_coefficients().clone();
        drop(result);
        let report = context.report(&[shared]).unwrap();
        assert_eq!(report.tensor_buffers.retained_bytes, Some(coefficients));
        assert!(report.total_bytes.is_some());
    }
}

#[test]
fn joint_routing_workspace_validates_shapes_domains_and_native_dimension_limits() {
    assert!(JointGroupSelectionSpec::new(i32::MAX, 1, 1, 1.).is_err());
    let mut op = operation(&[2, 3], 8, 8, 3, 3, 1.);
    op.inputs[2] = WorkspaceLayout::new(&[7], WorkspaceDtype::Float32).unwrap();
    assert!(mechanisms().operation_bound(&op).is_err());
    let mut op = operation(&[2, 3], 8, 8, 3, 3, 1.);
    op.inputs[0] = WorkspaceLayout::new(&[2, 3, 8], WorkspaceDtype::Int32).unwrap();
    assert!(mechanisms().operation_bound(&op).unwrap().is_none());
    assert!(mechanisms().host_workspace_bound(&op).unwrap().is_none());
    let mut op = operation(&[1], 8, 8, 3, 3, 1.);
    op.inputs[0] = WorkspaceLayout::new(&[i32::MAX, 2, 8], WorkspaceDtype::Float32).unwrap();
    assert!(mechanisms().operation_bound(&op).is_err());
    let context = WorkspaceContext::new(mechanisms());
    let x = [&[1, 0][..], &[11, 0][..], &[8][..], &[1][..]]
        .into_iter()
        .map(|s| {
            WorkspaceTensor::existing(
                WorkspaceLayout::new(s, WorkspaceDtype::Float32).unwrap(),
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(JointGroupSelectionInput::new(
        &x[0],
        &x[1],
        &x[2],
        &x[3],
        JointGroupSelectionSpec::new(8, 3, 3, 1.).unwrap()
    )
    .is_err());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };
    fn values(a: &MlxTensor, s: &Stream) -> Vec<f32> {
        a.to_f32_vec(s).unwrap()
    }
    fn inputs(
        op: &WorkspaceOperation,
        dtype: Dtype,
        low_parameters: bool,
        strided: bool,
        ties: bool,
        s: &Stream,
    ) -> Vec<MlxTensor> {
        let d = op.inputs[1].shape()[1] as usize;
        op.inputs
            .iter()
            .enumerate()
            .map(|(slot, l)| {
                let data = (0..l.elements().unwrap() as usize)
                    .map(|i| match slot {
                        1 if ties => ((i % d * 7) % 19) as f32 / 128. - 0.05,
                        2 if ties => 0.125,
                        2 => ((i * 13) % 71) as f32 / 128.,
                        3 => 0.625,
                        _ => ((i * 11 + slot * 7) % 37) as f32 / 128. - 0.1,
                    })
                    .collect::<Vec<_>>();
                let data = if strided {
                    data.iter().flat_map(|x| [*x, 0.125]).collect()
                } else {
                    data
                };
                let mut array = Array::from_slice(&data, &[data.len() as i32])
                    .as_dtype(
                        if slot == 0 || low_parameters {
                            dtype
                        } else {
                            Dtype::Float32
                        },
                        s,
                    )
                    .unwrap();
                if strided {
                    array = array.try_index_device((..).stride_by(2), s).unwrap();
                }
                MlxTensor::from_array(array.reshape(l.shape(), s).unwrap())
            })
            .collect()
    }
    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_joint_routing_workspace_bounds_cover_shared_storage_and_unbiased_weights() {
        let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let m = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = 0;
        for (leading, d, p, shared, k) in [
            (&[1][..], 7, 4, 1, 2),
            (&[2, 3][..], 8, 8, 3, 3),
            (&[2, 1, 2][..], 5, 7, 2, 7),
            (&[1][..], 64, 4097, 5, 1),
            (&[0][..], 3, 4, 2, 2),
        ] {
            for scale in [1., 1.7] {
                for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                    for low_parameters in [false, true] {
                        for strided in [false, true] {
                            for ties in [false, true] {
                                let op = operation(leading, d, p, shared, k, scale);
                                let x = inputs(&op, dtype, low_parameters, strided, ties, &s);
                                safemlx::transforms::eval(x.iter().map(MlxTensor::as_array))
                                    .unwrap();
                                s.synchronize().unwrap();
                                let before = safemlx::memory::active_memory().unwrap();
                                safemlx::memory::reset_peak_memory().unwrap();
                                let output = execute::<MlxNeuralBackend>(&op, &x, &s).unwrap();
                                safemlx::transforms::eval([
                                    output.primary_indices().as_array(),
                                    output.primary_coefficients().as_array(),
                                    output.always_on_coefficients().as_array(),
                                ])
                                .unwrap();
                                s.synchronize().unwrap();
                                let observed = safemlx::memory::peak_memory()
                                    .unwrap()
                                    .saturating_sub(before)
                                    as u64;
                                let bound = m.operation_bound(&op).unwrap().unwrap();
                                let allowed = bound.scratch_bytes
                                    + bound
                                        .outputs
                                        .iter()
                                        .filter_map(|o| match o {
                                            WorkspaceOutputStorage::Allocate(b) => Some(*b),
                                            _ => None,
                                        })
                                        .sum::<u64>();
                                assert!(observed<=allowed,"{leading:?} d={d} p={p} shared={shared} k={k} {dtype:?} low={low_parameters} strided={strided} ties={ties}: {observed}>{allowed}");
                                let rows = op.outputs[0].shape()[0] as usize;
                                let primary = output
                                    .primary_coefficients()
                                    .as_array()
                                    .allocation_info()
                                    .unwrap();
                                let always = output
                                    .always_on_coefficients()
                                    .as_array()
                                    .allocation_info()
                                    .unwrap();
                                assert_eq!(primary, always);
                                assert!(primary.is_some() || rows == 0);
                                if let Some(info) = primary {
                                    assert!(
                                        info.bytes() as u64
                                            <= capacity(
                                                m.allocation(),
                                                rows as u64 * (k + shared) as u64
                                            )
                                            .unwrap()
                                    );
                                }
                                if let Some(info) = output
                                    .primary_indices()
                                    .as_array()
                                    .allocation_info()
                                    .unwrap()
                                {
                                    assert!(
                                        info.bytes() as u64
                                            <= capacity(m.allocation(), rows as u64 * p as u64)
                                                .unwrap()
                                    );
                                }
                                let ids = output
                                    .primary_indices()
                                    .as_array()
                                    .contiguous(false, &s)
                                    .unwrap()
                                    .into_evaluated()
                                    .unwrap()
                                    .try_to_vec::<u32>()
                                    .unwrap();
                                let actual = values(output.primary_coefficients(), &s);
                                let shared_actual = values(output.always_on_coefficients(), &s);
                                let data = x.iter().map(|a| values(a, &s)).collect::<Vec<_>>();
                                let (d, p, shared, k) =
                                    (d as usize, p as usize, shared as usize, k as usize);
                                let (atol, rtol) = if dtype == Dtype::Float32 {
                                    (3e-5, 3e-4)
                                } else {
                                    (4e-3, 0.04)
                                };
                                for row in 0..rows {
                                    let logits = (0..p + shared)
                                        .map(|g| {
                                            (0..d)
                                                .map(|i| {
                                                    data[0][row * d + i] as f64
                                                        * data[1][g * d + i] as f64
                                                })
                                                .sum::<f64>()
                                        })
                                        .collect::<Vec<_>>();
                                    let sigmoid = |v: f64| 1. / (1. + (-v).exp());
                                    let ranking = (0..p)
                                        .map(|g| sigmoid(logits[g]) + data[2][g] as f64)
                                        .collect::<Vec<_>>();
                                    let chosen = &ids[row * k..(row + 1) * k];
                                    let mut unique = std::collections::BTreeSet::new();
                                    let mut scores = Vec::new();
                                    for &id in chosen {
                                        assert!((id as usize) < p && unique.insert(id));
                                        scores.push(sigmoid(logits[id as usize]));
                                    }
                                    let min = chosen
                                        .iter()
                                        .map(|&id| ranking[id as usize])
                                        .fold(f64::INFINITY, f64::min);
                                    for (g, &score) in ranking.iter().enumerate() {
                                        if !unique.contains(&(g as u32)) {
                                            assert!(
                                                score <= min + atol + rtol * min.abs(),
                                                "ranking cutoff {score}>{min}"
                                            );
                                        }
                                    }
                                    scores.extend((p..p + shared).map(|g| sigmoid(logits[g])));
                                    let denominator = scores.iter().sum::<f64>();
                                    for (i, &score) in scores.iter().enumerate() {
                                        let expected =
                                            score / denominator * scale as f64 * data[3][0] as f64;
                                        let actual = if i < k {
                                            actual[row * k + i]
                                        } else {
                                            shared_actual[row * shared + i - k]
                                        };
                                        assert!(
                                            (actual as f64 - expected).abs()
                                                <= atol + rtol * expected.abs(),
                                            "{dtype:?} {actual}!={expected}"
                                        );
                                    }
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!("JOINT_ROUTING_WORKSPACE_NATIVE_CASES={cases}");
    }
}
