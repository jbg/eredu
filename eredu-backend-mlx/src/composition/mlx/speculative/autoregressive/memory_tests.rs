use super::*;
use crate::backend::runtime::cache::state::MlxKeyValueState;
use crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms;
use eredu_architectures::external_assistant::{
    ExternalAssistantExecutionMechanisms, ExternalAssistantTensorPlacement,
    Gemma4AssistantArchitecture,
};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
    cache::LayerCachePolicy,
};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryError};

type External = MlxExternalAssistantMechanisms;
type Family = Gemma4AssistantArchitecture;

fn empty_state(owner: &NativeMemoryOwner) -> MlxPredictionTargetState {
    let state = MlxKeyValueState::device(
        eredu_runtime::StateLayout::new(
            LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxPredictionTargetState::new(state);
    state.retain_memory_owner(owner).unwrap();
    state
}

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
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless ownership fixture");
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

fn memory_error<'a>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    loop {
        if let Some(memory) = error.downcast_ref::<WorkingMemoryError>() {
            return Some(memory);
        }
        error = error.source()?;
    }
}

fn reclaim(pool: &WorkingMemoryPool, stream: &Stream) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while pool.unquoted_owner_count().unwrap() != 0 {
        stream.synchronize().unwrap();
        submission_recovery::reap();
        safemlx::reclaim_allocation_owners();
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn autoregressive_host_only_checkpoints_keep_the_bound_domain_until_last_restore_drops() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let state = MlxAutoregressiveState {
        native: empty_state(&owner),
        stream: StateStream::Ordinary(stream.clone()),
        pool: pool.clone(),
        source_origin: None,
        source_role: AutoregressiveSource::Target,
        original_copy: None,
    };
    drop(owner);
    let saved = MlxAutoregressiveMechanisms::checkpoint(&state).unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool);
    let restored = MlxAutoregressiveMechanisms::restore(&saved, context).unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 3);
    assert_eq!(restored.native.generation().unwrap(), 0);
    drop((state, saved));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 3);
    drop(restored);
    reclaim(&pool, &stream);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn external_snapshots_reject_reserved_destination_and_restore_preserves_installed_owners() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let source = empty_state(&owner);
    drop(owner);
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let reserved = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool);
    let error = <External as ExternalAssistantExecutionMechanisms<Family>>::control_checkpoint(
        &source, context,
    )
    .err()
    .expect("a zero-byte reservation still excludes a snapshot copy");
    assert_eq!(
        memory_error(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(source.generation().unwrap(), 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(reserved);
    let saved = <External as ExternalAssistantExecutionMechanisms<Family>>::control_checkpoint(
        &source, context,
    )
    .unwrap();
    let installed_owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let mut installed = empty_state(&installed_owner);
    drop(installed_owner);
    <External as ExternalAssistantExecutionMechanisms<Family>>::control_restore(
        &mut installed,
        &saved,
        context,
    )
    .unwrap();
    drop((source, saved));
    assert_eq!(source_pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 3);
    drop(installed);
    reclaim(&source_pool, &stream);
    reclaim(&pool, &stream);
}

#[test]
fn external_copied_tensor_keeps_its_new_owner_with_escaped_native_aliases() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let source = MlxTensor::from_array(Array::from_slice(&[2_f32, 5., 11.], &[1, 3]));
    owner.retain_array(source.as_array()).unwrap();
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool);
    let copied = <External as ExternalAssistantExecutionMechanisms<Family>>::control_copy_tensor(
        &source,
        ExternalAssistantTensorPlacement::Target,
        context,
    )
    .unwrap()
    .into_array();
    assert_ne!(
        copied.allocation_info().unwrap().unwrap().identity(),
        source
            .as_array()
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity(),
    );
    let escaped = copied.try_index_device((.., 1..), &stream).unwrap();
    drop((source, owner, copied));
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(escaped.evaluated().unwrap().as_slice::<f32>(), &[5., 11.]);
    drop(escaped);
    reclaim(&source_pool, &stream);
    reclaim(&pool, &stream);
}

#[test]
fn exact_copy_funding_refusal_crosses_the_executor_boundary_without_reboxing() {
    use eredu_nn::workspace::{WorkspaceMetadataError, WorkspaceMetadataFundingError};
    use std::error::Error as _;
    let cause = WorkspaceMetadataFundingError::Capacity {
        required: 8193,
        available: 47,
    };
    for error in [
        Error::WorkspacePlanning(cause),
        Error::Neural(WorkspaceMetadataError::Funding(cause).into()),
    ] {
        // Copy/table wrappers inspect source() before the executor consumes the
        // error. Same-display wrappers must expose the fixed leaf itself.
        let mut source: &(dyn std::error::Error + 'static) = &error;
        let retained = loop {
            if let Some(retained) = source.downcast_ref::<WorkspaceMetadataFundingError>() {
                break retained;
            }
            source = source.source().expect("exact funding source remains reachable");
        };
        assert_eq!(retained, &cause);
        let failure = MlxAutoregressiveMechanisms::take_retained_failure(error).unwrap();
        assert_eq!(
            failure.kind(),
            eredu_core::BackendFailureKind::ResourceExhausted
        );
        assert_eq!(
            failure
                .source()
                .unwrap()
                .downcast_ref::<eredu_core::HostMetadataFundingError>(),
            Some(&cause)
        );
        let failure = eredu_core::BackendFailure::from_error(failure);
        assert_eq!(
            failure
                .source()
                .unwrap()
                .downcast_ref::<eredu_core::HostMetadataFundingError>(),
            Some(&cause)
        );
    }
    let ordinary = MlxAutoregressiveMechanisms::invalid("ordinary failure");
    assert!(matches!(
        MlxAutoregressiveMechanisms::take_retained_failure(ordinary),
        Err(Error::InvalidOperation("ordinary failure"))
    ));
}
