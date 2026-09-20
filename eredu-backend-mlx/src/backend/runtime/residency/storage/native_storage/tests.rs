use super::*;
use eredu_core::{BackendFailureKind, SharedBackendFailure};
use eredu_runtime::working_memory::*;
use std::error::Error as _;

#[test]
fn native_issuer_cold_projection_preserves_exact_typed_source_without_retry_or_reclassification() {
    let cause = SharedBackendFailure::new(
        BackendFailureKind::Busy,
        safemlx::error::Exception::custom("retained cold preparation failure"),
    );
    let pointer = std::ptr::from_ref(
        cause
            .source_error()
            .downcast_ref::<safemlx::error::Exception>()
            .unwrap(),
    );
    let selection = NativeStorageSelection::default();
    let source = Err(cause);
    let mechanism = MlxNativeStorage::new(&source, &selection);
    let other = mechanism.clone();
    drop(source);
    drop(mechanism);
    let Err(retained) = &other.runtime else {
        panic!("failed cold runtime must stay failed")
    };
    let native = NativeStorageCause::Cold(retained.retained());
    assert_eq!(
        std::ptr::from_ref(
            native
                .source()
                .unwrap()
                .downcast_ref::<safemlx::error::Exception>()
                .unwrap()
        ),
        pointer
    );
    assert!(other.selection().same_selection(&selection));
    // No budget can be created here: no accepted custody was manufactured.
}

#[test]
fn native_issuer_requested_owner_layouts_keep_capacity_and_allocator_proof_separate() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let zero =
        PreparedOriginalBufferBudget::<OriginalNativeBudgetCustody>::layout(&runtime, 0).unwrap();
    let nonzero =
        PreparedOriginalBufferBudget::<OriginalNativeBudgetCustody>::layout(&runtime, 8192)
            .unwrap();
    assert_eq!(zero.capacity, 0);
    assert_eq!(nonzero.capacity, 8192);
    assert!(nonzero.native_owner_bytes > 0);
    assert!(nonzero.rust_node_bytes >= std::mem::size_of::<OriginalNativeBudgetCustody>());
    assert_eq!(zero.total_owner_bytes(), nonzero.total_owner_bytes());
    let sidecar = PreparedAllocationOwner::<Registration>::layout();
    assert!(sidecar.rust_node_bytes() >= std::mem::size_of::<Registration>());
    assert!(preparation_control_bytes().unwrap() > 0);
    // Layout queries neither construct custody nor issue a native allowance.
}

#[test]
fn native_issuer_positive_ordinary_proof_keeps_nonzero_alias_and_refuses_lazy_unknown() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let root = Array::from_slice(&[3.5f32, -7.0, 11.25], &[3]);
    let alias = root.clone();
    let lazy = root.add(&root, &stream).unwrap();
    let entry = safemlx::OrdinarySubmissionEntry::enter().unwrap();
    let facts = root.try_allocation_info().unwrap().unwrap();
    assert!(facts.bytes() >= 12);
    let observed = ordinary_observation(&root).unwrap();
    assert!(
        matches!(MlxNativeStorage::describe(&observed), NativeStorageObservation::Ordinary(StorageIdentity::Native(key), bytes) if key == facts.identity() && bytes == facts.bytes() as u64)
    );
    assert!(matches!(
        ordinary_observation(&lazy),
        Err(NativeStorageCause::Fixed(
            OriginalBufferCause::UncertifiedBacking
        ))
    ));
    assert!(lazy.try_allocation_info().unwrap().is_none());
    drop(observed);
    drop(entry);
    drop(root);
    assert_eq!(
        alias.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[3.5, -7.0, 11.25]
    );
    assert_eq!(
        lazy.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[7.0, -14.0, 22.5]
    );
}

#[test]
fn native_issuer_selected_plan_stays_unknown_even_for_actual_existing_source_only_report() {
    use eredu_core::{
        EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry, InputTokenCount,
        LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
    };
    use eredu_nn::workspace::*;
    let runtime = Ok(Rc::new(PreparedInputRuntime::prepare().unwrap()));
    let selection = NativeStorageSelection::default();
    let mechanism = MlxNativeStorage::new(&runtime, &selection);
    let root = Array::from_slice(&[2.5f32, -6.0], &[2]);
    let allocation = root.allocation_info().unwrap().unwrap();
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let registered = pool
        .register_storage([(
            StorageIdentity::Native(allocation.identity()),
            allocation.bytes() as u64,
        )])
        .unwrap();
    let context = WorkspaceContext::new(
        crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap(),
    );
    let existing = WorkspaceExistingStorage::new(Some(allocation.bytes() as u64), &context);
    let storage = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        [(
            StorageIdentity::Native(allocation.identity()),
            existing.clone(),
        )],
    )
    .unwrap();
    let value = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap(),
        &existing,
        &context,
    )
    .unwrap();
    let g = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let report = quote_inference_workspace(g, |_| {
        context.begin_state_span([&value])?;
        context.report(&[value.clone()])
    })
    .unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let zero = || {
        WorkspaceBound::bounded(
            0,
            "fixture visits one already registered array without new operations",
        )
    };
    let outside = ExecutionWorkspaceEstimate {
        geometry: g,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    };
    let quote = ResidualInferenceQuote::compose(&report, state, outside, &storage)
        .unwrap()
        .into_incremental();
    let before = pool.used_bytes().unwrap();
    let plan = mechanism
        .selected_plan(quote.span_workspace(), None, None, None)
        .unwrap();
    assert_eq!(plan.control_bytes(), None);
    let controls = PreparedTextControlWorkspace::prepare_controls(
        g,
        quote.span_workspace().plan(),
        TextHostControlFacts::new(Some(0), Some(0), Some(0)),
    )
    .unwrap()
    .with_native_storage(plan)
    .unwrap();
    assert!(matches!(
        quote.with_span_workspace_and_text_controls(controls),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert_eq!(
        root.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[2.5, -6.0]
    );
    drop(registered);
}

#[test]
fn native_control_owner_layout_counts_finite_custody_and_rejects_unqualified_source_keys() {
    let runtime = Ok(Rc::new(PreparedInputRuntime::prepare().unwrap()));
    let selection = NativeStorageSelection::default();
    let mechanism = MlxNativeStorage::new(&runtime, &selection);
    assert!(
        mechanism.key_clone_storage_bytes().is_some(),
        "the selected key validator has a finite supported source-key representation"
    );
    let bytes: std::sync::Arc<[u8]> = vec![3u8, 5, 7].into();
    assert!(matches!(
        mechanism.validate_source_key(&StorageIdentity::from_bytes(&bytes)),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let zero = mechanism.publication_owner_layout(0).unwrap();
    let nonzero = mechanism.publication_owner_layout(8192).unwrap();
    let Some(zero) = zero else {
        assert!(nonzero.is_none());
        return;
    };
    let nonzero = nonzero.unwrap();
    assert_eq!(
        zero.control_bytes(2, 3),
        nonzero.control_bytes(2, 3),
        "physical native capacity belongs to P and is not counted again in Q"
    );
    assert!(nonzero.control_bytes(1, 0).unwrap() > 0);
    assert!(nonzero.control_bytes(2, 3).unwrap() > nonzero.control_bytes(1, 3).unwrap());
    assert!(nonzero.control_bytes(2, 3).unwrap() > nonzero.control_bytes(2, 2).unwrap());
    assert_eq!(nonzero.control_bytes(usize::MAX, usize::MAX), None);
}

mod mutable_component;

#[test]
fn foreign_native_budget_proves_physical_ownership_without_claiming_a_payer() {
    let runtime = Rc::new(PreparedInputRuntime::prepare().unwrap());
    let plan = safemlx::OriginalMutablePairPlan::inspect(&runtime).unwrap();
    let donor = PreparedOriginalBufferBudget::try_new(&runtime, plan.facts().backing_bytes(), ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let recipient = PreparedOriginalBufferBudget::try_new(&runtime, 0, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let root = plan
        .prepare(
            [43, 47],
            donor.clone(),
            safemlx::OriginalMutablePairCustodies::new((), (), (), (), ()),
        )
        .unwrap()
        .try_construct()
        .unwrap();
    let facts = donor.inspect_array(&root).unwrap().unwrap().allocation();
    let selection = NativeStorageSelection::default();
    let mechanism = MlxNativeStorage::new(&Ok(runtime.clone()), &selection);
    let local = mechanism
        .observe(&donor, NativeStorageRoot::Array(&root))
        .unwrap();
    assert!(matches!(
        MlxNativeStorage::describe(&local),
        NativeStorageObservation::Originating(StorageIdentity::Native(key), bytes)
            if key == facts.identity() && bytes == facts.bytes() as u64
    ));
    drop(local);
    let alias = root.clone();
    drop((root, donor));
    let foreign = mechanism
        .observe(&recipient, NativeStorageRoot::Array(&alias))
        .unwrap();
    assert!(matches!(
        MlxNativeStorage::describe(&foreign),
        NativeStorageObservation::ExistingPhysical(StorageIdentity::Native(key), bytes)
            if key == facts.identity() && bytes == facts.bytes() as u64
    ));
    drop(foreign);
    assert_eq!(
        alias.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
        &[43, 47]
    );
}

#[test]
fn immutable_array_proves_physical_ownership_without_claiming_prepaid_funding() {
    use safemlx::{
        OriginalScopeObserver, OwnedHostCopyPlan, PreparedPrefillFailure,
        PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
        SubmissionScope,
    };
    let runtime = Rc::new(PreparedInputRuntime::prepare().unwrap());
    let values = vec![2.0f32, -3.0, 7.0];
    let plan = OwnedHostCopyPlan::<f32>::new(&runtime, &[3], values.capacity()).unwrap();
    let capacity = plan.facts().backing_bytes();
    let preparation =
        PreparedSubmissionGraphQuota::try_new(plan.facts().metadata_bytes(), ()).unwrap();
    let slot = plan.prepare(preparation).unwrap();
    let budget = PreparedOriginalBufferBudget::try_new(&runtime, 0, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(64 << 10, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(
        PreparedSubmissionRecordQuota::<()>::minimum_layout()
            .unwrap()
            .capacity,
        (),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let failure = PreparedPrefillFailure::try_new(())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(graph)
            .with_record_quota(records),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let root = slot.try_fill(values, &observer).unwrap();
    scope.seal();
    drop((observer, scope, failure));
    let key = root.try_allocation_info().unwrap().unwrap().identity();
    let mechanism = MlxNativeStorage::new(&Ok(runtime.clone()), &NativeStorageSelection::default());
    let observed = mechanism
        .observe(&budget, NativeStorageRoot::Array(&root))
        .unwrap();
    assert!(matches!(observed, Observation::Immutable(_)));
    assert!(matches!(
        MlxNativeStorage::describe(&observed),
        NativeStorageObservation::ExistingPhysical(StorageIdentity::Native(identity), bytes)
            if identity == key && bytes == capacity as u64
    ));
    drop(observed);
    assert_eq!(budget.occupied_bytes(), 0);
    assert_eq!(
        root.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[2.0, -3.0, 7.0]
    );
}
