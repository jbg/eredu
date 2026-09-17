//! Fresh copy-only prompt construction; no forward/resume authority is claimed.
use super::super::tests::{
    controls,
    funded::{finish_native, metal, publish_source, sampler, sampling_plan, settle},
    operands, state, values, FAIL_AFTER_LAYER,
};
use super::*;
use crate::backend::{
    managed_memory::NativeMemoryOwner,
    nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    runtime::residency::storage::RetainedStorage,
};
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    StateMemoryLayout, TextGenerationConfig, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceIsolatedCopyPlan};
use eredu_nn::PoolingAttentionCache as _;
use eredu_nn::Tensor;
use eredu_runtime::{
    working_memory::{
        InferenceExecutionIdentity, InferenceRequest, InferenceStateRetention,
        InferenceTextPreparation, RegisteredWorkspaceStorage, WorkingMemoryFundingRun,
        WorkingMemoryFundingScope, WorkingMemoryStorage, WorkspaceCopyLimits,
    },
    RuntimeLayerState, RuntimeState, RuntimeStateComponents,
};
use std::num::NonZeroU32;

// Actual fresh, zero-output prompt authority. Every requested byte comes from
// the concrete P+N program above, not from a synthetic forward/state estimate.
fn fresh(
    pool: &WorkingMemoryPool,
    bytes: u64,
    capacity: u64,
) -> Result<(InferenceTextPreparation, WorkingMemoryFundingRun), WorkingMemoryError> {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound =
        |n| WorkspaceBound::bounded(n, "actual dense initialization and isolated native copy");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let reservation = pool.reserve_with_capacity(
        &execution,
        &Admission {
            requested_positions: 1,
            state,
            incremental_required_bytes: bytes,
            available_memory_bytes: None,
        },
        capacity,
    )?;
    let (reservation, run) = reservation.into_funding()?;
    // The low-level preparation contract accepts zero outputs; no application
    // generation configuration or forward loop is being constructed here.
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: false,
        temperature: 0.0,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(0),
    });
    let preparation =
        InferenceRequest::from(reservation).prepare_text(&execution, geometry, config)?;
    Ok((preparation, run))
}

struct Quote {
    bytes: u64,
    complete: WorkingMemoryStorage<StorageIdentity>,
}
fn quote(plan: &PreparedResidentPoolingCopy<'_>, pool: &WorkingMemoryPool) -> Quote {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let inputs = operands(plan)
        .into_iter()
        .map(|a| projection.project(a).unwrap())
        .collect::<Vec<_>>();
    let storage = projection.into_storage();
    let source = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        storage
            .iter()
            .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
    )
    .unwrap();
    let program =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &inputs).unwrap();
    let n = program.incremental_bytes().unwrap();
    let mut complete = RetainedStorage::default();
    plan.visit_retained_arrays(&mut |a| complete.include_array(a).unwrap());
    complete
        .include_metadata(eredu_runtime::SharedHostMetadata::Layout(
            plan.shared_layout().clone(),
        ))
        .unwrap();
    let complete = complete.pin_registered(pool).unwrap();
    Quote {
        bytes: n
            .checked_add(
                plan.dense_initialization()
                    .unwrap()
                    .initialization_peak_bytes(),
            )
            .unwrap(),
        complete,
    }
}
fn publish_arrays(
    completed: &PreparedDenseResidentPoolingState<'_>,
    native: &WorkingMemoryFundingScope,
    roots: &RefCell<Vec<Array>>,
) {
    for a in roots.borrow().iter() {
        a.evaluated().unwrap();
    }
    let mut values = RetainedStorage::default();
    completed.visit_operands(&mut |a| {
        a.evaluated().unwrap();
        values.include_array(a).unwrap();
    });
    drop(values.publish_funded(native).unwrap());
}
fn positions(plan: &PreparedResidentPoolingCopy<'_>) -> Vec<i32> {
    (0..plan.len())
        .map(|i| plan.layer(i).unwrap().offset())
        .collect()
}

#[test]
fn pooling_dense_prompt_preserves_real_slots_controls_and_fresh_retention() {
    for populated in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = metal();
        let mut source = state(&stream, populated);
        publish_source(&source, &loading);
        drop(loading);
        let (old_sampler, old_preparation, old_run) = sampler(&pool);
        source
            .inference_retention_mut()
            .admit(old_preparation.request());
        let revision = source.inference_retention().revision().clone();
        let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
        let expected = values(&plan);
        let expected_controls = controls(&plan);
        let original_ids = operands(&plan)
            .into_iter()
            .map(|a| a.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
        let quoted = quote(&plan, &pool);
        let cap = before.0.checked_add(quoted.bytes).unwrap();
        assert!(matches!(
            fresh(&pool, quoted.bytes, cap - 1),
            Err(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(
            (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
            before
        );
        let (preparation, run) = fresh(&pool, quoted.bytes, cap).unwrap();
        let (slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan.dense_host_copy(&pool).unwrap(), &run, quoted.complete)
            .unwrap();
        let roots = RefCell::new(Vec::with_capacity(original_ids.len() * 2));
        let completed = plan.copy_dense_retained(slots, &stream, &roots).unwrap();
        assert_eq!(roots.borrow().len(), original_ids.len() * 2);
        let pointer = completed.layers.get(0).unwrap() as *const MlxPoolingAttentionCache;
        let token = completed.slot_metadata().clone();
        let d = completed.retained_slot_bytes();
        publish_arrays(&completed, &native, &roots);
        let before_publish = pool.used_bytes().unwrap();
        let (published, completion) = completed.publish_for_control().unwrap();
        let destination = published.into_state();
        assert_eq!(pool.used_bytes().unwrap(), before_publish);
        assert_eq!(destination.as_ref().as_ptr(), pointer);
        assert!(destination
            .layer_slot_metadata()
            .unwrap()
            .same_storage(&token));
        assert!(destination
            .shared_layout()
            .unwrap()
            .same_storage(source.shared_layout().unwrap()));
        assert!(destination.inference_retention().is_empty());
        assert_ne!(destination.inference_retention().revision(), &revision);
        assert_eq!(source.inference_retention().revision(), &revision);
        assert_eq!(source.inference_retention().requests().len(), 1);
        roots.borrow_mut().clear();
        native.certify().unwrap();
        completion.finish().unwrap();
        preparation.bind_prompt().unwrap();
        let copied = PreparedResidentPoolingCopy::prepare(&destination).unwrap();
        assert_eq!(values(&copied), expected);
        assert_eq!(controls(&copied), expected_controls);
        assert_eq!(
            values(&PreparedResidentPoolingCopy::prepare(&source).unwrap()),
            expected
        );
        let ids = operands(&copied)
            .into_iter()
            .map(|a| a.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        assert!(ids.iter().all(|id| !original_ids.contains(id)));
        assert_eq!(
            ids.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            ids.len()
        );
        let raw = operands(&copied).first().map(|a| Array::clone(a));
        let raw_bytes = raw
            .as_ref()
            .map_or(0, |a| a.allocation_info().unwrap().unwrap().bytes() as u64);
        drop(copied);
        drop((
            source,
            destination,
            old_sampler,
            old_preparation,
            old_run,
            preparation,
            run,
        ));
        settle(&pool, d + raw_bytes);
        drop(raw);
        settle(&pool, d);
        drop(token);
        settle(&pool, 0);
    }
}

thread_local! { static HOUSEKEEPING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|n| n.set(n.get() + 1));
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.with(|n| n.set(0));
        Self
    }
    fn check(&self) {
        assert_eq!(HOUSEKEEPING.with(|n| n.get()), 0);
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn copied_pooling_projection_tracks_fresh_buffers_and_preserves_unknown_sources() {
    let stream = metal();
    let source = state(&stream, true);
    let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let cold = ColdGuard::new();
    let projected = plan
        .project_dense_workspace(NonZeroU32::new(2).unwrap(), &context)
        .unwrap();
    cold.check();
    drop(cold);
    assert!(
        !projected.source_storage.is_complete(),
        "lazy source is not evaluated by the copied projection"
    );
    drop(projected);
    let expected = values(&plan);
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let cold = ColdGuard::new();
    let projected = plan
        .project_dense_workspace(NonZeroU32::new(2).unwrap(), &context)
        .unwrap();
    cold.check();
    drop(cold);
    assert!(projected.source_storage.is_complete());
    assert!(projected.copy.total_bytes.is_some());
    assert_eq!(projected.copy.operations.len(), operands(&plan).len() * 2);
    for pair in projected.copy.operations.chunks_exact(2) {
        use eredu_nn::workspace::WorkspaceOperationKind;
        assert!(matches!(pair[0].kind, WorkspaceOperationKind::Contiguous));
        assert!(matches!(pair[1].kind, WorkspaceOperationKind::DeepCopy));
    }
    let roots = projected
        .state
        .as_ref()
        .iter()
        .flat_map(RuntimeLayerState::<WorkspaceBackend>::retained_values)
        .collect::<Vec<_>>();
    assert_eq!(roots.len(), operands(&plan).len());
    let destination_roots = roots.iter().map(|root| (*root).clone()).collect::<Vec<_>>();
    context.begin_state_span(&destination_roots).unwrap();
    let union = context
        .report(&destination_roots)
        .unwrap()
        .state
        .unwrap()
        .retained_bytes
        .unwrap();
    let separate: u64 = destination_roots
        .iter()
        .map(|root| {
            context
                .report(std::slice::from_ref(root))
                .unwrap()
                .state
                .unwrap()
                .retained_bytes
                .unwrap()
        })
        .sum();
    assert_eq!(
        union, separate,
        "each copied operand owns independent future backing"
    );
    assert_eq!(
        projected
            .state
            .as_ref()
            .iter()
            .map(|s| s.position())
            .collect::<Vec<_>>(),
        positions(&plan)
    );
    assert!(projected
        .state
        .shared_layout()
        .unwrap()
        .same_storage(plan.shared_layout()));
    let original = operands(&plan);
    for (root, a) in roots.iter().zip(&original) {
        assert_eq!(root.shape(), a.shape());
    }
    assert_eq!(values(&plan), expected);
    assert!(plan
        .project_dense_workspace(NonZeroU32::new(1).unwrap(), &context)
        .is_err());
    let absent = MlxPoolingAttentionState::stateless();
    assert!(matches!(
        PreparedResidentPoolingCopy::prepare(&absent),
        Err(ResidentPoolingCopyError::AbsentTable)
    ));
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let projected = crate::backend::runtime::cache::state::MlxPoolingAttentionStateFactory::project_resident_workspace_with_storage(
        &absent, NonZeroU32::new(2).unwrap(), &context).unwrap();
    assert!(projected.state.optional_layout().is_none());
    assert!(projected
        .state
        .prepare_layer_copy_slots()
        .unwrap()
        .is_none());
}

#[test]
fn saved_pooling_source_builds_fresh_dense_state_after_original_retirement() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let source = state(&stream, true);
    publish_source(&source, &loading);
    drop(loading);
    let (sampler, old_preparation, old_run) = sampler(&pool);
    let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
    let expected = values(&plan);
    let expected_controls = controls(&plan);
    let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
    let joined = sampling
        .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
        .unwrap();
    let (saved_sampler, slots, native) = pool
        .copy_text_components(joined, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    let (custody, native) = native.into_parts();
    let roots = RefCell::new(Vec::new());
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    finish_native(&saved, native, &roots);
    drop((source, sampler, old_preparation, old_run));
    settle(
        &pool,
        custody.bytes() + saved.shared_layout().capacity_bytes().unwrap(),
    );
    let plan = saved.prepare_copy().unwrap();
    let quoted = quote(&plan, &pool);
    let (preparation, run) = fresh(
        &pool,
        quoted.bytes,
        pool.used_bytes().unwrap() + quoted.bytes,
    )
    .unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(plan.dense_host_copy(&pool).unwrap(), &run, quoted.complete)
        .unwrap();
    let completed = plan.copy_dense_retained(slots, &stream, &roots).unwrap();
    publish_arrays(&completed, &native, &roots);
    let (published, completion) = completed.publish_for_control().unwrap();
    let destination = published.into_state();
    roots.borrow_mut().clear();
    native.certify().unwrap();
    completion.finish().unwrap();
    preparation.bind_prompt().unwrap();
    drop((saved, saved_sampler, custody, preparation, run));
    let copied = PreparedResidentPoolingCopy::prepare(&destination).unwrap();
    assert_eq!(values(&copied), expected);
    assert_eq!(controls(&copied), expected_controls);
    assert!(destination.inference_retention().is_empty());
    drop(copied);
    drop(destination);
    settle(&pool, 0);
}

#[test]
fn pooling_dense_wrong_source_and_partial_failure_preserve_actual_recovery_roots() {
    for late in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = metal();
        let source = state(&stream, true);
        let other = state(&stream, true);
        publish_source(&source, &loading);
        publish_source(&other, &loading);
        drop(loading);
        let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
        let expected = values(&plan);
        let quoted = quote(&plan, &pool);
        let (preparation, run) = fresh(
            &pool,
            quoted.bytes,
            pool.used_bytes().unwrap() + quoted.bytes,
        )
        .unwrap();
        let (slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan.dense_host_copy(&pool).unwrap(), &run, quoted.complete)
            .unwrap();
        let roots = RefCell::new(Vec::new());
        if late {
            FAIL_AFTER_LAYER.with(|f| f.set(Some(0)));
            let error = plan
                .copy_dense_retained(slots, &stream, &roots)
                .unwrap_err();
            assert!(matches!(error, ResidentPoolingCopyError::Native(_)));
            assert_eq!(FAIL_AFTER_LAYER.with(|f| f.get()), None);
            assert_eq!(
                roots.borrow().len(),
                4,
                "completed local key/value copies stay recoverable"
            );
            let mut cause: &(dyn std::error::Error + 'static) = &error;
            while let Some(next) = cause.source() {
                cause = next;
            }
            assert!(cause.is::<super::super::tests::InjectedPoolingCopyFailure>());
        } else {
            let error = PreparedResidentPoolingCopy::prepare(&other)
                .unwrap()
                .copy_dense_retained(slots, &stream, &roots)
                .unwrap_err();
            assert!(matches!(
                error,
                ResidentPoolingCopyError::Memory(WorkingMemoryError::IdentityMismatch)
            ));
            assert!(roots.borrow().is_empty());
        }
        assert!(preparation.bind_prompt().is_err());
        assert_eq!(
            values(&PreparedResidentPoolingCopy::prepare(&source).unwrap()),
            expected
        );
        // The injected failure is after synchronous construction, with every
        // submitted array retained. Explicitly settle all those exact roots
        // before certifying this test's native scope; host failure alone cannot.
        for array in roots.borrow().iter() {
            array.evaluated().unwrap();
        }
        roots.borrow_mut().clear();
        native.certify().unwrap();
        drop((source, other, preparation, run));
        settle(&pool, 0);
    }
}
