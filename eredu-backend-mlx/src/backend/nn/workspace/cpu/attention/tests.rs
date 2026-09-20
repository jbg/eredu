use super::*;
use eredu_nn::{AttentionRequest, NeuralBackend, Tensor};
const DTYPES: [WorkspaceFloatingType; 3] = [
    WorkspaceFloatingType::Float32,
    WorkspaceFloatingType::Float16,
    WorkspaceFloatingType::Bfloat16,
];
const SCALE: f32 = 0.37;
fn native_array(values: &[f32], shape: &[i32], dtype: WorkspaceFloatingType) -> safemlx::Array {
    match dtype {
        WorkspaceFloatingType::Float32 => safemlx::Array::from_slice(values, shape),
        WorkspaceFloatingType::Float16 => safemlx::Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        WorkspaceFloatingType::Bfloat16 => safemlx::Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
    }
}

fn rounded(value: f32, dtype: WorkspaceFloatingType) -> f32 {
    match dtype {
        WorkspaceFloatingType::Float32 => value,
        WorkspaceFloatingType::Float16 => half::f16::from_f32(value).to_f32(),
        WorkspaceFloatingType::Bfloat16 => half::bf16::from_f32(value).to_f32(),
    }
}

fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: [usize; 6],
    qdtype: WorkspaceFloatingType,
    vdtype: WorkspaceFloatingType,
    arithmetic: AttentionArithmetic,
    cap: Option<f32>,
    mask: u8,
    sinks: bool,
) -> Vec<f32> {
    let [heads, kv, queries, keys, width, dv] = shape;
    let score = if arithmetic == AttentionArithmetic::InputScores {
        qdtype
    } else {
        WorkspaceFloatingType::Float32
    };
    let output = super::super::super::representation::promote(qdtype, vdtype);
    let mut result = vec![0.; heads * queries * dv];
    for h in 0..heads {
        for row in 0..queries {
            let mut scores = Vec::new();
            for col in 0..keys {
                let mut dot = 0f32;
                for lane in 0..width {
                    dot += rounded(q[(h * queries + row) * width + lane], score)
                        * rounded(k[((h / (heads / kv)) * keys + col) * width + lane], score);
                }
                let mut x = rounded(rounded(dot, score) * SCALE, score);
                if let Some(cap) = cap {
                    x = rounded((x * cap.recip()).tanh() * cap, score);
                }
                if mask == 1 && col % 3 == 0 && col + 1 < keys {
                    x = f32::NEG_INFINITY;
                }
                if mask == 2 {
                    x = rounded(x + rounded(-0.13 * (col % 3) as f32, score), score);
                }
                scores.push(x);
            }
            if sinks {
                scores.push(rounded(-0.17 + 0.11 * h as f32, score));
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total = scores.iter().map(|x| (x - max).exp()).sum::<f32>();
            for lane in 0..dv {
                let mut sum = 0.;
                for col in 0..keys {
                    let p = rounded(rounded((scores[col] - max).exp() / total, qdtype), output);
                    sum += p * rounded(v[((h / (heads / kv)) * keys + col) * dv + lane], output);
                }
                result[(h * queries + row) * dv + lane] = rounded(sum, output);
            }
        }
    }
    result
}

#[test]
fn explicit_cpu_attention_matches_independent_scores_and_retires_resources() {
    use crate::{
        MlxTensor,
        backend::{
            MlxBackend, MlxDeviceIdentity, managed_memory::gpu_stream::PreparedExecutionStreams,
            nn::shared::MlxNeuralBackend,
        },
    };
    use safemlx::{
        Device, DeviceType, OriginalBufferBudget, OriginalScopeObserver, PrefillRoots,
        PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
        PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
        SubmissionScope,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    #[derive(Debug)]
    struct Lifetime(Arc<AtomicBool>);
    impl Drop for Lifetime {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32AndFloat16Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let allocator = environment.input_runtime().unwrap();

    for shape in [[2, 2, 1, 1, 3, 2], [4, 2, 3, 5, 8, 6], [2, 1, 2, 17, 4, 3]] {
        let [h, kv, queries, keys, width, dv] = shape;
        for qdtype in DTYPES {
            for kdtype in DTYPES {
                for vdtype in DTYPES {
                    for (arithmetic, cap) in [
                        (AttentionArithmetic::Fused, Some(1.7)),
                        (AttentionArithmetic::InputScores, None),
                        (AttentionArithmetic::InputScores, Some(1.7)),
                    ] {
                        for mask_kind in 0..3 {
                            for sinks in [false, true] {
                                let data = |count: usize, dtype, seed: usize| {
                                    (0..count)
                                        .map(|i| {
                                            rounded(
                                                ((i * seed + 3) % 29) as f32 * 0.073 - 0.9,
                                                dtype,
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                };
                                let qdata = data(h * queries * width, qdtype, 7);
                                let kdata = data(kv * keys * width, kdtype, 11);
                                let vdata = data(kv * keys * dv, vdtype, 13);
                                let qshape = [1, h as i32, queries as i32, width as i32];
                                let kshape = [1, kv as i32, keys as i32, width as i32];
                                let vshape = [1, kv as i32, keys as i32, dv as i32];
                                let query =
                                    MlxTensor::from_array(native_array(&qdata, &qshape, qdtype));
                                let key =
                                    MlxTensor::from_array(native_array(&kdata, &kshape, kdtype));
                                let value =
                                    MlxTensor::from_array(native_array(&vdata, &vshape, vdtype));
                                let mask = match mask_kind {
                                    1 => Some(MlxTensor::from_array(safemlx::Array::from_slice(
                                        &(0..keys)
                                            .map(|col| col % 3 != 0 || col + 1 == keys)
                                            .collect::<Vec<_>>(),
                                        &[1, keys as i32],
                                    ))),
                                    2 => Some(MlxTensor::from_array(safemlx::Array::from_slice(
                                        &(0..keys)
                                            .map(|col| -0.13 * (col % 3) as f32)
                                            .collect::<Vec<_>>(),
                                        &[1, keys as i32],
                                    ))),
                                    _ => None,
                                };
                                let sink = sinks.then(|| {
                                    MlxTensor::from_array(safemlx::Array::from_slice(
                                        &(0..h)
                                            .map(|i| -0.17 + 0.11 * i as f32)
                                            .collect::<Vec<_>>(),
                                        &[h as i32],
                                    ))
                                });
                                let context = WorkspaceContext::new(cpu);
                                let represented = |shape: &[i32], dtype| {
                                    WorkspaceTensor::existing(
                                        context
                                            .layout(shape, WorkspaceDtype::Float32)
                                            .unwrap()
                                            .with_representation(Some(
                                                WorkspaceRepresentation::new(dtype, true),
                                            )),
                                        &context,
                                    )
                                    .unwrap()
                                };
                                let q = represented(&qshape, qdtype);
                                let k = represented(&kshape, kdtype);
                                let v = represented(&vshape, vdtype);
                                let wm = match mask_kind {
                                    1 => Some(
                                        WorkspaceTensor::existing(
                                            context
                                                .layout(&[1, keys as i32], WorkspaceDtype::Bool)
                                                .unwrap(),
                                            &context,
                                        )
                                        .unwrap(),
                                    ),
                                    2 => Some(represented(
                                        &[1, keys as i32],
                                        WorkspaceFloatingType::Float32,
                                    )),
                                    _ => None,
                                };
                                let ws = sinks.then(|| {
                                    represented(&[h as i32], WorkspaceFloatingType::Float32)
                                });
                                context.begin_span();
                                let output = WorkspaceBackend::attention_with_sinks(
                                    AttentionRequest {
                                        queries: q,
                                        keys: k,
                                        values: v,
                                        scale: SCALE,
                                        mask: wm.as_ref(),
                                        sinks: ws.as_ref(),
                                        softcap: cap,
                                        arithmetic,
                                    },
                                    &context,
                                )
                                .unwrap();
                                let report = context.finish_report(&[output]).unwrap();
                                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                                    &report, ordinary, cpu, &context,
                                )
                                .unwrap();
                                let completion = recipe.completion;
                                let physical = OriginalBufferBudget::population_layout(
                                    &allocator,
                                    usize::try_from(recipe.storage.mutable_bytes()).unwrap(),
                                    recipe.storage.maximum_births(),
                                )
                                .unwrap()
                                .capacity();
                                let retired = Arc::new(AtomicBool::new(false));
                                let owner = Arc::new(Lifetime(retired.clone()));
                                let graph = PreparedSubmissionGraphQuota::try_new(
                                    recipe.graph_capacity,
                                    owner.clone(),
                                )
                                .unwrap()
                                .try_allocate()
                                .unwrap();
                                let records = PreparedSubmissionRecordQuota::try_new(
                                    recipe.record_capacity,
                                    owner.clone(),
                                )
                                .unwrap()
                                .try_allocate()
                                .unwrap();
                                let budget = PreparedOriginalBufferBudget::try_new(
                                    &allocator,
                                    physical,
                                    owner.clone(),
                                )
                                .unwrap()
                                .try_allocate()
                                .unwrap();
                                let failure = PreparedPrefillFailure::try_new(owner.clone())
                                    .unwrap()
                                    .try_allocate()
                                    .unwrap();
                                let mut roots =
                                    PrefillRoots::new_retained(&runtime, 1, &graph, &failure)
                                        .unwrap();
                                let mut scope = SubmissionScope::try_begin_retaining(
                                    PreparedSubmissionScopeOwner::try_new(owner.clone())
                                        .unwrap()
                                        .with_graph_quota(graph.clone())
                                        .with_record_quota(records.clone()),
                                )
                                .unwrap();
                                scope.enable_scoped_observation().unwrap();
                                scope.require_original_native_controls().unwrap();
                                roots.bind_scope(&scope).unwrap();
                                scope.enable_original_native_controls().unwrap();
                                scope.bind_original_buffer_budget(&budget).unwrap();
                                let observer = OriginalScopeObserver::require_current().unwrap();
                                for input in [&query, &key, &value]
                                    .into_iter()
                                    .chain(mask.iter())
                                    .chain(sink.iter())
                                {
                                    OperationEvent::validate_traversal_leaf(
                                        input.as_array(),
                                        &observer,
                                    )
                                    .unwrap();
                                }
                                let bank = OperationEvent::prepare_resident_graph(
                                    completion.graph,
                                    &observer,
                                )
                                .unwrap();
                                let actual = MlxNeuralBackend::attention_with_sinks(
                                    AttentionRequest {
                                        queries: query.clone(),
                                        keys: key.clone(),
                                        values: value.clone(),
                                        scale: SCALE,
                                        mask: mask.as_ref(),
                                        sinks: sink.as_ref(),
                                        softcap: cap,
                                        arithmetic,
                                    },
                                    stream,
                                )
                                .unwrap();
                                drop(bank);
                                roots.append(actual.as_array()).unwrap();
                                roots
                                    .complete_current_scope_on_stream_prepared(
                                        stream,
                                        &completion.traversal,
                                    )
                                    .unwrap();
                                assert!(!observer.status().failed());
                                assert!(
                                    budget.occupied_bytes() > 0
                                        && budget.occupied_bytes() <= physical
                                );
                                let output_dtype =
                                    super::super::super::representation::promote(qdtype, vdtype);
                                assert_eq!(actual.as_array().dtype(), native_dtype(output_dtype));
                                let output = match output_dtype {
                                    WorkspaceFloatingType::Float32 => actual
                                        .as_array()
                                        .evaluated()
                                        .unwrap()
                                        .try_to_vec::<f32>()
                                        .unwrap(),
                                    WorkspaceFloatingType::Float16 => actual
                                        .as_array()
                                        .evaluated()
                                        .unwrap()
                                        .try_to_vec::<half::f16>()
                                        .unwrap()
                                        .into_iter()
                                        .map(|x| x.to_f32())
                                        .collect(),
                                    WorkspaceFloatingType::Bfloat16 => actual
                                        .as_array()
                                        .evaluated()
                                        .unwrap()
                                        .try_to_vec::<half::bf16>()
                                        .unwrap()
                                        .into_iter()
                                        .map(|x| x.to_f32())
                                        .collect(),
                                };
                                let expected = reference(
                                    &qdata, &kdata, &vdata, shape, qdtype, vdtype, arithmetic, cap,
                                    mask_kind, sinks,
                                );
                                let tolerance = if [qdtype, kdtype, vdtype]
                                    .iter()
                                    .all(|&d| d == WorkspaceFloatingType::Float32)
                                {
                                    2e-5
                                } else {
                                    0.012
                                };
                                for (actual, expected) in output.iter().zip(expected) {
                                    assert!(
                                        (actual - expected).abs()
                                            <= tolerance * expected.abs().max(1.0),
                                        "shape={shape:?} types={qdtype:?}/{kdtype:?}/{vdtype:?} mask={mask_kind} sinks={sinks} arithmetic={arithmetic:?} cap={cap:?}: {actual} != {expected}"
                                    );
                                }
                                scope.seal();
                                let deadline =
                                    std::time::Instant::now() + std::time::Duration::from_secs(10);
                                crate::backend::submission_recovery::wait_for_retirement(|| {
                                    let (progress, status) = observer.progress().unwrap();
                                    assert_eq!(
                                        progress,
                                        safemlx::ScopedSubmissionProgress::Observed
                                    );
                                    assert!(!status.failed() && !status.blocked());
                                    assert!(std::time::Instant::now() < deadline);
                                    status.is_settled()
                                });
                                assert_eq!(
                                    observer.retire_completed_records().unwrap(),
                                    safemlx::SubmissionRetirement::CompleteSnapshot
                                );
                                safemlx::try_with_submission_retirement(|| {
                                    drop((roots, scope, observer, failure, records, graph, budget))
                                })
                                .unwrap();
                                drop(owner);
                                safemlx::reclaim_allocation_owners();
                                assert!(!retired.load(Ordering::SeqCst));
                                drop(actual);
                                crate::backend::submission_recovery::wait_for_retirement(|| {
                                    safemlx::try_retire_completed_submissions().unwrap();
                                    MlxNeuralBackend::reclaim_retired_resources();
                                    safemlx::reclaim_allocation_owners();
                                    assert!(std::time::Instant::now() < deadline);
                                    retired.load(Ordering::SeqCst)
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}
