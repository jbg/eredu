//! Admitted native attention composition with an independent windowed reference.
use super::*;
use crate::backend::nn::workspace::{
    MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
    SpeculativeNumericalRecipe,
};
use eredu_nn::{
    workspace::{
        WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType,
        WorkspaceRepresentation, WorkspaceTensor,
    },
    AttentionArithmetic, AttentionRequest, NeuralBackend, Tensor,
};
use safemlx::OperationEvent;
#[test]
fn sliding_cpu_attention_multiple_tiles_match_independent_arithmetic_and_retire() {
    if !crate::tests::support::native_process::enter("qualified-native-source") {
        return;
    }
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams, nn::shared::MlxNeuralBackend,
            MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use safemlx::{
        Device, DeviceType, OriginalBufferBudget, OriginalScopeObserver, PrefillRoots,
        PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
        PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
        SubmissionScope,
    };
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
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
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
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

    for (queries, keys, window, offset) in [
        (1, 4, 4, 11),
        (4, 4, 2, 0),
        (257, 257, 7, 0),
        (513, 519, 7, 6),
    ] {
        for sinks in [false, true] {
            let data = |count: usize, seed: usize| {
                (0..count)
                    .map(|i| ((i * seed + 3) % 29) as f32 * 0.073 - 0.9)
                    .collect::<Vec<_>>()
            };
            let qdata = data(4 * queries as usize * 8, 7);
            let kdata = data(2 * keys as usize * 8, 11);
            let vdata = data(2 * keys as usize * 6, 13);
            let qshape = [1, 4, queries, 8];
            let kshape = [1, 2, keys, 8];
            let vshape = [1, 2, keys, 6];
            let query = MlxTensor::from_array(Array::from_slice(&qdata, &qshape));
            let key = MlxTensor::from_array(Array::from_slice(&kdata, &kshape));
            let value = MlxTensor::from_array(Array::from_slice(&vdata, &vshape));
            let sink = sinks.then(|| {
                MlxTensor::from_array(Array::from_slice(
                    &(0..4).map(|h| -0.17 + 0.11 * h as f32).collect::<Vec<_>>(),
                    &[4],
                ))
            });
            let context = WorkspaceContext::new(cpu);
            let source = |shape: &[i32]| {
                WorkspaceTensor::existing(
                    context
                        .layout(shape, WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap()
            };
            let q = source(&qshape);
            let k = source(&kshape);
            let v = source(&vshape);
            let ws = sinks.then(|| source(&[4]));
            context.begin_span();
            let output = WorkspaceBackend::sliding_window_attention_with_sinks(
                AttentionRequest {
                    queries: q,
                    keys: k,
                    values: v,
                    scale: 0.37,
                    mask: None,
                    sinks: ws.as_ref(),
                    softcap: Some(1.7),
                    arithmetic: AttentionArithmetic::Fused,
                },
                window,
                offset,
                &context,
            )
            .unwrap();
            let report = context.finish_report(&[output]).unwrap();
            let recipe =
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                    .unwrap();
            let completion = recipe.completion;
            // Component fixture: allocate the exact host table before native scope
            // entry. Facade metadata custody is exercised by model integration.
            let limits = completion.grouped_outputs;
            assert_eq!(limits.calls, 1);
            assert_eq!(
                limits.chunks,
                (queries as usize)
                    .div_ceil(crate::backend::nn::attention::SLIDING_QUERY_TILE as usize)
            );
            let mut table = PreparedGroupedOutputs::default();
            table.slots.try_reserve_exact(limits.calls).unwrap();
            for _ in 0..limits.calls {
                let mut values = Vec::new();
                values.try_reserve_exact(limits.chunks).unwrap();
                assert_eq!(values.capacity(), limits.chunks);
                table.slots.push(Some(GroupedChunkOutputs {
                    values,
                    limit: limits.chunks,
                    _custody: None,
                }));
            }
            let prepared = PreparedTokenValidations(TokenValidationBatch::default(), table);
            let physical = OriginalBufferBudget::population_layout(
                &allocator,
                usize::try_from(recipe.storage.mutable_bytes()).unwrap(),
                recipe.storage.maximum_births(),
            )
            .unwrap()
            .capacity();
            let retired = Arc::new(AtomicBool::new(false));
            let owner = Arc::new(Lifetime(retired.clone()));
            let graph = PreparedSubmissionGraphQuota::try_new(recipe.graph_capacity, owner.clone())
                .unwrap()
                .try_allocate()
                .unwrap();
            let records =
                PreparedSubmissionRecordQuota::try_new(recipe.record_capacity, owner.clone())
                    .unwrap()
                    .try_allocate()
                    .unwrap();
            let budget = PreparedOriginalBufferBudget::try_new(&allocator, physical, owner.clone())
                .unwrap()
                .try_allocate()
                .unwrap();
            let failure = PreparedPrefillFailure::try_new(owner.clone())
                .unwrap()
                .try_allocate()
                .unwrap();
            let mut roots = PrefillRoots::new_retained(&runtime, 1, &graph, &failure).unwrap();
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
            for input in [&query, &key, &value].into_iter().chain(sink.iter()) {
                OperationEvent::validate_traversal_leaf(input.as_array(), &observer).unwrap();
            }
            let collector = TokenValidationScope::begin_prepared(prepared).unwrap();
            let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
            let actual = MlxNeuralBackend::sliding_window_attention_with_sinks(
                AttentionRequest {
                    queries: query.clone(),
                    keys: key.clone(),
                    values: value.clone(),
                    scale: 0.37,
                    mask: None,
                    sinks: sink.as_ref(),
                    softcap: Some(1.7),
                    arithmetic: AttentionArithmetic::Fused,
                },
                window,
                offset,
                stream,
            )
            .unwrap();
            assert!(GroupedChunkOutputs::prepare(limits.chunks).is_err());
            let validations = collector.finish();
            assert!(validations.is_empty());
            drop(validations);
            drop(bank);
            roots.append(actual.as_array()).unwrap();
            roots
                .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
                .unwrap();
            assert!(!observer.status().failed());
            assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
            assert_eq!(actual.as_array().dtype(), Dtype::Float32);
            let output = actual
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            let origin = offset + queries - keys;
            for row in 0..queries as usize {
                for h in 0..4usize {
                    let mut scores = Vec::new();
                    for col in 0..keys as usize {
                        let dot = (0..8)
                            .map(|lane| {
                                qdata[(h * queries as usize + row) * 8 + lane]
                                    * kdata[((h / 2) * keys as usize + col) * 8 + lane]
                            })
                            .sum::<f32>();
                        let position = offset + row as i32;
                        let key_position = origin + col as i32;
                        scores.push(
                            if key_position <= position && position - key_position < window {
                                ((dot * 0.37) * (1.7f32.recip())).tanh() * 1.7
                            } else {
                                f32::NEG_INFINITY
                            },
                        );
                    }
                    if sinks {
                        scores.push(-0.17 + 0.11 * h as f32);
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denominator = scores.iter().map(|x| (x - max).exp()).sum::<f32>();
                    for lane in 0..6usize {
                        let expected = (0..keys as usize)
                            .map(|col| {
                                (scores[col] - max).exp() / denominator
                                    * vdata[((h / 2) * keys as usize + col) * 6 + lane]
                            })
                            .sum::<f32>();
                        let actual = output[(row * 4 + h) * 6 + lane];
                        assert!(
                            (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                            "queries={queries} keys={keys} window={window} offset={offset} sinks={sinks} row={row} h={h} lane={lane}: {actual} != {expected}"
                        );
                    }
                }
            }
            scope.seal();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                let (progress, status) = observer.progress().unwrap();
                assert_eq!(progress, safemlx::ScopedSubmissionProgress::Observed);
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
