use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, HostPreparationAuthority,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn runtime(
    stream: &Stream,
    pool: &WorkingMemoryPool,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    let backend = MlxBackend::new(stream, stream).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model = eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
        .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    settle(pool, 0, None);
    assert!(runtime.session().payload.model.has_published_idle_storage());
    assert!(pool.used_bytes().unwrap() > 0);
    (runtime, artifact)
}

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle(pool: &WorkingMemoryPool, owners: usize, bytes: Option<u64>) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == owners
            && bytes.is_none_or(|bytes| pool.used_bytes().unwrap() == bytes)
    });
}

// A real zero-byte reservation exercises the pool's exclusion protocol. This
// fixture authorizes no native inference or parser work and supplies no claim
// about their numerical workspace bounds.
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
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "host preparation exclusion fixture");
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

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(cause) = error.downcast_ref::<T>() {
            return Some(cause);
        }
        error = error.source()?;
    }
}

fn excluded(pool: &WorkingMemoryPool) {
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

#[test]
fn zero_finite_reservation_rejects_host_preparation_without_work_or_accounting_change() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, artifact) = runtime(&stream, &pool);
    let bytes = pool.used_bytes().unwrap();
    let reservation = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &zero_admission(),
            bytes,
        )
        .unwrap();
    let before = (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    );
    let native_paths = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let resets = paths::session_reset_attempts();
    let revision = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap()
        .revision()
        .clone();
    let error = MlxBackend::acquire_host_preparation(&runtime).unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(
        (
            pool.used_bytes().unwrap(),
            pool.peak_bytes().unwrap(),
            pool.effective_capacity().unwrap()
        ),
        before
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(paths::snapshot(), native_paths);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(paths::session_reset_attempts(), resets);
    assert_eq!(
        runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .revision(),
        &revision
    );
    drop(reservation);
    let authority = MlxBackend::acquire_host_preparation(&runtime).unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(authority);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((runtime, artifact));
    settle(&pool, 0, Some(0));
}

#[test]
fn independent_runtime_pool_remains_available_while_another_pool_is_reserved() {
    let stream = stream();
    let blocked = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let available = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (first, first_artifact) = runtime(&stream, &blocked);
    let (second, second_artifact) = runtime(&stream, &available);
    let reservation = blocked
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &zero_admission(),
            blocked.used_bytes().unwrap(),
        )
        .unwrap();
    let error = MlxBackend::acquire_host_preparation(&first).unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    let authority = MlxBackend::acquire_host_preparation(&second).unwrap();
    assert_eq!(blocked.unquoted_owner_count().unwrap(), 0);
    assert_eq!(available.unquoted_owner_count().unwrap(), 1);
    excluded(&available);
    drop(authority);
    assert_eq!(available.unquoted_owner_count().unwrap(), 0);
    drop((reservation, first, second, first_artifact, second_artifact));
    settle(&blocked, 0, Some(0));
    settle(&available, 0, Some(0));
}

#[test]
fn authority_aliases_outlive_runtime_and_source_without_retaining_model_payload() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, artifact) = runtime(&stream, &pool);
    let retired = runtime.session().test_payload_retirement_probe();
    let authority = MlxBackend::acquire_host_preparation(&runtime).unwrap();
    let alias = authority.clone();
    let last = alias.clone();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop((runtime, artifact, authority));
    settle(&pool, 1, Some(0));
    assert!(retired());
    excluded(&pool);
    drop(alias);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    excluded(&pool);
    drop(last);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let reservation = pool
        .reserve_with_capacity(&InferenceExecutionIdentity::default(), &zero_admission(), 0)
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(reservation);
}

struct ReentrantAuthority {
    pool: WorkingMemoryPool,
    observed: Arc<AtomicBool>,
    // Metadata-only authority drops after the reentrant destructor has run.
    _authority: HostPreparationAuthority,
}

impl Drop for ReentrantAuthority {
    fn drop(&mut self) {
        excluded(&self.pool);
        assert_eq!(self.pool.unquoted_owner_count().unwrap(), 1);
        self.observed.store(true, Ordering::SeqCst);
    }
}

#[test]
fn erased_authority_destructor_can_reenter_pool_while_exclusion_is_still_live() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, artifact) = runtime(&stream, &pool);
    let authority = MlxBackend::acquire_host_preparation(&runtime).unwrap();
    let observed = Arc::new(AtomicBool::new(false));
    let owner = HostPreparationAuthority::retain(ReentrantAuthority {
        pool: pool.clone(),
        observed: observed.clone(),
        _authority: authority,
    });
    let alias = owner.clone();
    drop((runtime, artifact, owner));
    settle(&pool, 1, Some(0));
    assert!(!observed.load(Ordering::SeqCst));
    drop(alias);
    assert!(observed.load(Ordering::SeqCst));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}
