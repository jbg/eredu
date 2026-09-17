use super::*;
use crate::backend::{error::Error, submission_recovery};
use eredu_core::{
    cache::LayerCachePolicy, Admission, Completion as _, EstimationCompleteness,
    ExecutionWorkspaceEstimate, InferenceGeometry, InputTokenCount, InspectableBackendSession as _,
    LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool,
};

fn zero_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless fixture without managed payload");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 0,
        available_memory_bytes: None,
    }
}

fn assert_unquoted(pool: &WorkingMemoryPool) {
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

fn assert_retired(pool: &WorkingMemoryPool) {
    submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == 0 && pool.used_bytes().unwrap() == 0
    });
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(reservation);
}

#[test]
fn quoted_zero_byte_reservation_rejects_before_communication_and_materialization() {
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend =
        MlxBackend::new(execution.stream(), execution.stream()).with_memory_pool(pool.clone());
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new()).unwrap();
    let rank = crate::backend::MlxRankContext::new(
        2,
        0,
        crate::backend::DeviceAssignment::new(DeviceType::Cpu, 0),
    )
    .unwrap();
    path_instrumentation::reset();
    let error = backend
        .materialize_after_communication(
            eredu_core::SessionCapabilities::default(),
            Some(&manifest),
            Some(rank),
            |_| {
                path_instrumentation::materialization();
                panic!("live quoted work must reject before materialization");
            },
        )
        .err()
        .unwrap();
    let Error::Other(error) = error else {
        panic!("memory admission must precede the missing-world error");
    };
    assert!(matches!(
        error.downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::ReservedWorkActive)
    ));
    assert_eq!(
        path_instrumentation::communication_realization_attempts(),
        0
    );
    assert_eq!(path_instrumentation::snapshot(), Default::default());
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(reservation);
    assert_retired(&pool);
}

#[test]
fn prepared_model_exclusion_transitions_to_idle_storage_then_operation_ownership() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let other_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let another = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend =
        MlxBackend::new(execution.stream(), execution.stream()).with_memory_pool(pool.clone());
    let peer = MlxBackend::new(another.stream(), another.stream()).with_memory_pool(pool.clone());
    let independent = MlxBackend::new(execution.stream(), execution.stream())
        .with_memory_pool(other_pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    assert!(
        pool.used_bytes().unwrap() > 0,
        "loaded storage is published"
    );
    assert_unquoted(backend.memory_pool());
    assert_unquoted(peer.memory_pool());
    let unrelated_reservation = independent
        .memory_pool()
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let mut session = peer.create_session(model).unwrap();
    drop(backend);
    submission_recovery::wait_for_retirement(|| {
        peer.memory_pool().unquoted_owner_count().unwrap() == 0
    });
    let existing_bytes = peer.memory_pool().used_bytes().unwrap();
    assert!(existing_bytes > 0, "idle storage remains registered");
    let idle_reservation = peer
        .memory_pool()
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    assert_eq!(peer.memory_pool().used_bytes().unwrap(), existing_bytes);
    drop(idle_reservation);
    session
        .submit_token_decode(&peer, 1)
        .unwrap()
        .completion
        .wait()
        .unwrap();
    assert_unquoted(&pool);
    drop(session);
    assert_retired(&pool);
    assert_eq!(other_pool.unquoted_owner_count().unwrap(), 0);
    drop(unrelated_reservation);
}

#[test]
fn extracted_executable_preserves_loaded_model_memory_ownership() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend =
        MlxBackend::new(execution.stream(), execution.stream()).with_memory_pool(pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let executable = model.into_inner().into_executable();
    drop(backend);
    assert_unquoted(&pool);
    drop(executable);
    assert_retired(&pool);
}

#[test]
fn escaped_ordinary_and_inspected_logits_retain_domain_until_the_last_native_alias_retires() {
    use safemlx::ops::indexing::TryIndexOp;

    for inspect in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend =
            MlxBackend::new(execution.stream(), execution.stream()).with_memory_pool(pool.clone());
        let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
        let model = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
            .unwrap();
        let mut session = backend.create_session(model).unwrap();
        let output = if inspect {
            let inspected = session
                .inspect_decode(
                    &backend,
                    safemlx::Array::from_slice(&[1_u32], &[1, 1]),
                    &eredu_core::ObservationRequest::selected([
                        eredu_core::ObservationSelector::Exact(
                            eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                        ),
                    ]),
                )
                .unwrap();
            inspected.output
        } else {
            let submission = session.submit_token_decode(&backend, 1).unwrap();
            submission.completion.wait().unwrap();
            let output = submission.output;
            drop(submission.completion);
            output
        };
        let allocation = output
            .logits()
            .unwrap()
            .as_array()
            .allocation_info()
            .unwrap()
            .expect("completed tiny-model logits have certified backing");
        let wrapper_alias = output.clone();
        let array = output.into_logits().unwrap().into_array();
        assert_eq!(array.allocation_info().unwrap(), Some(allocation));
        let expected = array.evaluated().unwrap().as_slice::<f32>().to_vec();
        let alias = array.clone();
        let view = array
            .try_index_device((.., 1..), execution.stream())
            .unwrap();
        view.evaluated().unwrap();
        assert_eq!(view.allocation_info().unwrap(), Some(allocation));
        assert!(view.nbytes() < allocation.bytes());
        drop((array, wrapper_alias, session, backend));
        crate::backend::ordinary_retirement::reclaim_all();
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        assert_unquoted(&pool);
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation));
        assert_eq!(alias.evaluated().unwrap().as_slice::<f32>(), expected);
        drop(alias);
        safemlx::reclaim_allocation_owners();
        assert_unquoted(&pool);
        assert_eq!(view.allocation_info().unwrap(), Some(allocation));
        assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &expected[1..]);
        drop(view);
        safemlx::reclaim_allocation_owners();
        assert_retired(&pool);
    }
}

#[test]
fn failing_materialization_and_unwind_hold_domain_ownership_inside_native_work() {
    for unwind in [false, true] {
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend =
            MlxBackend::new(execution.stream(), execution.stream()).with_memory_pool(pool.clone());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.materialize_after_communication(
                eredu_core::SessionCapabilities::default(),
                None,
                None,
                |_| {
                    assert_unquoted(&pool);
                    let values = safemlx::Array::from_slice(&[0.25_f32, -0.75], &[2]);
                    let _pending = values.square(execution.stream())?;
                    if unwind {
                        panic!("materialization unwind sentinel");
                    }
                    Err(Error::ArchitectureModel(
                        "materialization failure sentinel".into(),
                    ))
                },
            )
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            let error = result.unwrap().err().unwrap();
            assert!(error
                .to_string()
                .contains("materialization failure sentinel"));
        }
        backend.synchronize().unwrap();
        assert_retired(&pool);
    }
}
