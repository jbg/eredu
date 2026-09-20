//! Component-only acceptance: no TextGeneration, model fit or native issuer.
//! All four native support owners are priced from their actual final types and
//! receive a receipt before allocation. Each output's source bank now covers
//! its own immutable backing; no separate scope-wide backing grant remains.
use super::*;
use crate::backend::runtime::checkpoint::store::GgufHostCopyCause;
use eredu_core::{
    AdmissionRequest, CacheStateStrategy, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputModalities, InputTokenCount, LayerSchedule, ModelCapabilities,
    Observed, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
};
use eredu_runtime::working_memory::{
    plan_prefill_incremental_with_capacity, quote_inference_workspace, HostDestinationCause,
    HostDestinationFacts, HostSourceConstructionFacts, InferenceExecutionIdentity,
    OriginalHostDestinationBank, OriginalHostSourceBank, OriginalHostSourceReceipt,
    PrefillPlanningError, PreparedTextControlWorkspace, RegisteredWorkspaceStorage,
    ResidualInferenceQuote, TextHostControlFacts, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{
    OwnedHostCopyPlan, PreparedPrefillFailure, PreparedSubmissionGraphQuota,
    PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner, SubmissionRetirement,
    SubmissionScope,
};
use std::{
    mem::size_of,
    num::NonZeroU8,
    time::{Duration, Instant},
};

#[derive(Debug)]
struct EmptyWorkspace;
impl WorkspaceMechanisms for EmptyWorkspace {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        // No tensor operation is emitted by this component's empty report. An
        // accidental new tensor operation must remain unknown, never cost zero.
        Ok(None)
    }
}
struct ScopeCustody {
    _scope: OriginalHostSourceReceipt,
}
struct SupportFacts {
    graph: u64,
    record: u64,
    failure: u64,
    scope: u64,
}
impl SupportFacts {
    // These are actual allocated hard ceilings for the enclosing test scope,
    // not estimates or reserves substituted for a selected model's graph fit.
    const GRAPH: usize = 4 << 20;
    const RECORD: usize = 1 << 20;
    fn new() -> Self {
        let scope = PreparedSubmissionScopeOwner::<ScopeCustody>::layout().unwrap();
        let scope = [
            scope.rust_node_bytes,
            scope.native_scope_bytes,
            scope.native_retirement_control_bytes,
            scope.prepared_bytes,
            scope.preparation_control_bytes,
            scope.begin_control_bytes,
            scope.retirement_control_bytes,
            scope.preparation_failure_bytes,
            scope.begin_failure_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .unwrap();
        Self {
            graph: PreparedSubmissionGraphQuota::<OriginalHostSourceReceipt>::layout(Self::GRAPH)
                .unwrap()
                .total_bytes()
                .unwrap() as u64,
            record: PreparedSubmissionRecordQuota::<OriginalHostSourceReceipt>::layout(Self::RECORD)
                .unwrap()
                .total_bytes()
                .unwrap() as u64,
            failure: PreparedPrefillFailure::<OriginalHostSourceReceipt>::layout()
                .unwrap()
                .total_bytes()
                .unwrap() as u64,
            scope: scope as u64,
        }
    }
    fn bytes(&self) -> u64 {
        self.graph + self.record + self.failure + self.scope
    }
}

fn immutable_backing(fixture: &OriginalGgufMissFixture) -> u64 {
    use crate::backend::runtime::checkpoint::gguf;
    use eredu_gguf::LogicalDtype;
    let plan = fixture
        .source
        .conversion_plan(&OriginalGgufMissFixture::request())
        .unwrap();
    fn one<T: safemlx::ArrayElement + Send + 'static>(
        fixture: &OriginalGgufMissFixture,
        shape: &[i32],
        elements: usize,
    ) -> u64 {
        OwnedHostCopyPlan::<T>::new(&fixture.host_runtime, shape, elements)
            .unwrap()
            .facts()
            .backing_bytes() as u64
    }
    plan.conversion().outputs().iter().enumerate().map(|(ordinal, output)| {
        let shape = gguf::mlx_shape_i32(plan.output_name(), output.shape()).unwrap();
        let request = crate::backend::runtime::execution::generic::gguf_host_typed::TypedOutputRequest::for_output(plan.conversion(), ordinal).unwrap();
        match request.dtype {
            LogicalDtype::F32 => one::<f32>(fixture, &shape, request.output_elements),
            LogicalDtype::F16 => one::<half::f16>(fixture, &shape, request.output_elements),
            LogicalDtype::U8 => one::<u8>(fixture, &shape, request.output_elements),
            LogicalDtype::U32 => one::<u32>(fixture, &shape, request.output_elements),
            other => panic!("unexpected component scalar {other:?}"),
        }
    }).sum()
}

fn component_destinations<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    destinations: Option<(u64, usize, usize)>,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostDestinationBank,
    ) -> T,
) -> T {
    component_destinations_with_ceiling(
        source_bytes,
        attempts,
        partitions,
        destinations,
        |_| 0,
        operation,
    )
}

fn component_destinations_with_ceiling<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    destinations: Option<(u64, usize, usize)>,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostDestinationBank,
    ) -> T,
) -> T {
    component_destinations_with_baseline(
        source_bytes,
        attempts,
        partitions,
        destinations,
        0,
        additional_ceiling,
        operation,
    )
}

fn component_destinations_with_baseline<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    destinations: Option<(u64, usize, usize)>,
    existing: u64,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostDestinationBank,
    ) -> T,
) -> T {
    component_destinations_with_retained_account(
        source_bytes,
        attempts,
        partitions,
        destinations,
        existing,
        additional_ceiling,
        0,
        operation,
    )
}

// Independently initialized shared owners may outlive this component alongside
// its host hold. Their exact original debit stays separate from the component
// quote and must still be present at its exit.
fn component_destinations_with_retained_account<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    destinations: Option<(u64, usize, usize)>,
    existing: u64,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
    retained_account: u64,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostDestinationBank,
    ) -> T,
) -> T {
    component_destinations_with_retained_account_and_controls(
        source_bytes,
        attempts,
        partitions,
        destinations,
        existing,
        additional_ceiling,
        retained_account,
        0,
        operation,
    )
}

fn component_destinations_with_retained_account_and_controls<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    destinations: Option<(u64, usize, usize)>,
    existing: u64,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
    retained_account: u64,
    additional_controls: u64,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostDestinationBank,
    ) -> T,
) -> T {
    component_destinations_with_native_budget(
        source_bytes,
        attempts,
        partitions,
        destinations,
        existing,
        additional_ceiling,
        retained_account,
        additional_controls,
        None,
        operation,
    )
}

fn component_destinations_with_native_budget<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    destinations: Option<(u64, usize, usize)>,
    existing: u64,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
    retained_account: u64,
    additional_controls: u64,
    native_budget: Option<&safemlx::OriginalBufferBudget>,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostDestinationBank,
    ) -> T,
) -> T {
    let support = SupportFacts::new();
    let source = HostSourceConstructionFacts::new(
        source_bytes + support.bytes(),
        attempts + 4,
        partitions + 1,
    )
    .unwrap();
    let (bytes, calls, parts) = destinations.unwrap_or((0, 0, 0));
    let host = HostDestinationFacts::new(bytes, calls)
        .expect("component requires actual qualified host storage")
        .with_partitions(parts + 1)
        .unwrap()
        .with_source_constructions(source)
        .unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        input_positions: 1,
        cached_positions: 0,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let pool = WorkingMemoryPool::new(u64::MAX, existing).unwrap();
    let context = WorkspaceContext::new(EmptyWorkspace);
    let storage = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        std::iter::empty::<(
            crate::backend::runtime::residency::storage::StorageIdentity,
            eredu_nn::workspace::WorkspaceExistingStorage,
        )>(),
    )
    .unwrap();
    let report = quote_inference_workspace(geometry, |_| {
        context.begin_state_span([])?;
        context.report(&[])
    })
    .unwrap();
    let request = AdmissionRequest {
        input: InputTokenCount::text(1),
        max_output_tokens: 1,
        batch_size: 1,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
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
        request.input,
        1,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let empty =
        || WorkspaceBound::bounded(0, "component emits no inference tensor/state operation");
    let outside = ExecutionWorkspaceEstimate {
        geometry,
        activations: empty(),
        attention: empty(),
        vocabulary: empty(),
        state_update: empty(),
        materialization: empty(),
        retained: empty(),
    };
    let quote = ResidualInferenceQuote::compose(&report, state, outside, &storage)
        .unwrap()
        .into_incremental();
    // No facade/text machinery is constructed by this fixture. The native
    // support and source construction representations are included above.
    let named = (size_of::<SupportFacts>()
        + size_of::<ScopeCustody>()
        + OriginalScopeObserver::control_bytes().unwrap()) as u64;
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry,
        quote.span_workspace().plan(),
        TextHostControlFacts::new(Some(0), Some(named + additional_controls), Some(0)),
    )
    .unwrap()
    .with_host_destinations(host)
    .unwrap();
    let quote = quote
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let capacity = existing.checked_add(quote.incremental_bytes()).unwrap();
    // A later accepted alias publication is quoted against this same pool
    // before A's retained raw host custody installs its capacity constraint.
    let common_ceiling = capacity.checked_add(additional_ceiling(&pool)).unwrap();
    let capabilities = ModelCapabilities {
        effective_model_type: "bounded source-arena component only".into(),
        native_max_context: Observed::exact(2, "empty component"),
        effective_max_context: Observed::exact(2, "empty component"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(plan_prefill_incremental_with_capacity(
        &execution,
        &pool,
        &capabilities,
        request,
        geometry,
        capacity - 1,
        |_| Ok(quote.clone())
    ), Err(PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })) if required_bytes == available_bytes + 1));
    if common_ceiling != capacity {
        // Probe A's own exact bound without retaining its smaller constraint
        // across the later B request. No source/native work starts here.
        let before = pool.used_bytes().unwrap();
        let exact = plan_prefill_incremental_with_capacity(
            &execution,
            &pool,
            &capabilities,
            request,
            geometry,
            capacity,
            |_| Ok(quote.clone()),
        )
        .unwrap();
        drop(exact);
        assert_eq!(pool.used_bytes().unwrap(), before);
    }
    let (reservation, accepted) = plan_prefill_incremental_with_capacity(
        &execution,
        &pool,
        &capabilities,
        request,
        geometry,
        common_ceiling,
        |_| Ok(quote.clone()),
    )
    .unwrap();
    let (reservation, funding) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted
        .into_funded_text_span_workspace(&funding, &reservation)
        .unwrap();
    let protected = span.protected_host_bytes();
    let controls = span.control_guard();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut support_bank = host.split(0, 0, Some((support.bytes(), 4))).unwrap();
    let mut source = support_bank.take_source_constructions().unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(
        SupportFacts::GRAPH,
        source.try_debit(support.graph).unwrap(),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(
        SupportFacts::RECORD,
        source.try_debit(support.record).unwrap(),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let failure = PreparedPrefillFailure::try_new(source.try_debit(support.failure).unwrap())
        .unwrap()
        .try_allocate()
        .unwrap();
    let scope = source.try_debit(support.scope).unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(ScopeCustody { _scope: scope })
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone()),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    if let Some(budget) = native_budget {
        scope.bind_original_buffer_budget(budget).unwrap();
    }
    let observer = OriginalScopeObserver::require_current().unwrap();
    assert_eq!(
        (
            source.remaining_bytes(),
            source.remaining_attempts(),
            source.remaining_partitions()
        ),
        (0, 0, 0)
    );
    drop((source, support_bank));
    let result = operation(&controls, &observer, &pool, host);
    scope.seal();
    assert!(observer.status().is_settled() && !observer.status().failed());
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let retired = observer.retire_completed_records().unwrap();
        if retired == SubmissionRetirement::CompleteSnapshot {
            break;
        }
        assert_eq!(retired, SubmissionRetirement::Busy);
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop((observer, scope, failure, records, graph));
    safemlx::reclaim_allocation_owners();
    drop((
        controls,
        span,
        reservation,
        funding,
        quote,
        storage,
        context,
    ));
    assert_eq!(
        pool.used_bytes().unwrap(),
        existing + protected + retained_account,
        "actual result retains exactly the host hold, baseline and separate shared account"
    );
    result
}

fn settle_pool(pool: &WorkingMemoryPool) {
    settle_pool_at(pool, 0);
}
fn settle_pool_at(pool: &WorkingMemoryPool, existing: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        safemlx::reclaim_allocation_owners();
        if pool.used_bytes().unwrap() == existing {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "source ownership has not retired"
        );
        std::thread::yield_now();
    }
}

#[test]
fn source_arena_component_executes_funded_nonzero_all_format_cache_lifetimes() {
    for (constructor, outputs) in [
        (
            OriginalGgufMissFixture::new as fn() -> OriginalGgufMissFixture,
            1,
        ),
        (OriginalGgufMissFixture::new_affine, 3),
        (OriginalGgufMissFixture::new_mxfp4, 2),
        (OriginalGgufMissFixture::new_packed, 1),
    ] {
        let fixture = constructor();
        let layouts = fixture.source_storage_layouts();
        assert_eq!(layouts.len(), outputs);
        let (array, pool) = component(
            layouts.iter().sum::<u64>() * 3,
            outputs * 3,
            3,
            |controls, observer, pool, bank| {
                (
                    fixture.funded_source_miss_hit_miss(controls, observer, bank),
                    pool.clone(),
                )
            },
        );
        drop(fixture);
        assert!(pool.used_bytes().unwrap() > 0);
        std::thread::spawn(move || drop(array)).join().unwrap();
        settle_pool(&pool);
    }
}

#[test]
fn source_arena_component_refuses_short_dense_and_second_affine_before_allocation() {
    for affine in [false, true] {
        let fixture = if affine {
            OriginalGgufMissFixture::new_affine()
        } else {
            OriginalGgufMissFixture::new()
        };
        let layouts = fixture.source_storage_layouts();
        let ordinal = usize::from(affine);
        let bytes = if affine { layouts[0] } else { layouts[0] - 1 };
        let (error, pool) = component(bytes, ordinal + 1, 0, |controls, observer, pool, bank| {
            (
                fixture.funded_source_refusal(controls, observer, bank, ordinal),
                pool.clone(),
            )
        });
        drop(fixture);
        assert!(pool.used_bytes().unwrap() > 0);
        drop(error);
        settle_pool(&pool);
    }
}

fn component<T>(
    source_bytes: u64,
    attempts: usize,
    partitions: usize,
    operation: impl FnOnce(
        &OriginalTextControlGuard,
        &OriginalScopeObserver,
        &WorkingMemoryPool,
        OriginalHostSourceBank,
    ) -> T,
) -> T {
    component_destinations(
        source_bytes,
        attempts,
        partitions,
        None,
        |controls, observer, pool, mut host| {
            let source = host.take_source_constructions().unwrap();
            assert_eq!(
                (
                    source.remaining_bytes(),
                    source.remaining_attempts(),
                    source.remaining_partitions()
                ),
                (source_bytes, attempts, partitions)
            );
            operation(controls, observer, pool, source)
        },
    )
}

#[test]
fn paired_host_source_component_preserves_all_format_miss_hit_retry_storage() {
    for constructor in [
        OriginalGgufMissFixture::new as fn() -> OriginalGgufMissFixture,
        OriginalGgufMissFixture::new_affine,
        OriginalGgufMissFixture::new_mxfp4,
        OriginalGgufMissFixture::new_packed,
    ] {
        let fixture = constructor();
        let layouts = fixture.source_storage_layouts();
        let plan = fixture
            .source
            .conversion_plan(&OriginalGgufMissFixture::request())
            .unwrap();
        let bound = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
        let (array, pool) = component_destinations(
            (layouts.iter().sum::<u64>()
                + super::super::super::super::cache::control_bytes().unwrap())
                * 3,
            (layouts.len() + 1) * 3,
            3,
            Some((bound.bytes() as u64 * 3, bound.calls() * 3, 3)),
            |controls, observer, pool, bank| {
                (
                    fixture.funded_destinations_miss_hit_miss(controls, observer, bank),
                    pool.clone(),
                )
            },
        );
        drop(fixture);
        std::thread::spawn(move || drop(array)).join().unwrap();
        settle_pool(&pool);
    }
}

#[test]
fn paired_host_source_component_retains_affine_prefix_on_later_arena_refusal() {
    for affine in [false, true] {
        let fixture = if affine {
            OriginalGgufMissFixture::new_affine()
        } else {
            OriginalGgufMissFixture::new()
        };
        let layouts = fixture.source_storage_layouts();
        let ordinal = usize::from(affine);
        let source_bytes = if affine { layouts[0] } else { layouts[0] - 1 };
        let plan = fixture
            .source
            .conversion_plan(&OriginalGgufMissFixture::request())
            .unwrap();
        let bound = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
        let (error, pool) = component_destinations(
            source_bytes,
            ordinal + 1,
            0,
            Some((bound.bytes() as u64, bound.calls(), 0)),
            |controls, observer, pool, bank| {
                (
                    fixture.funded_destinations_refusal(controls, observer, bank, ordinal),
                    pool.clone(),
                )
            },
        );
        drop(fixture);
        assert!(pool.used_bytes().unwrap() > 0);
        drop(error);
        settle_pool(&pool);
    }
}

#[test]
fn immutable_source_layouts_price_each_backing_once_without_changing_control_diagnostics() {
    for constructor in [
        OriginalGgufMissFixture::new as fn() -> OriginalGgufMissFixture,
        OriginalGgufMissFixture::new_affine,
        OriginalGgufMissFixture::new_mxfp4,
        OriginalGgufMissFixture::new_packed,
    ] {
        let fixture = constructor();
        let reads = fixture.source_physical_reads();
        let controls = fixture.source_arena_layouts();
        let combined = fixture.source_storage_layouts();
        assert_eq!(controls.len(), combined.len());
        assert!(combined.iter().zip(&controls).all(|(total, c)| total > c));
        // Independent caller of the actual safe/native producer: B is not
        // inferred from logical bytes or observed capacity after allocation.
        assert_eq!(
            combined.iter().sum::<u64>() - controls.iter().sum::<u64>(),
            immutable_backing(&fixture)
        );
        assert_eq!(fixture.source_physical_reads(), reads);
    }
}

#[test]
fn immutable_source_component_rejects_control_only_credit_before_backing_and_arena() {
    let fixture = OriginalGgufMissFixture::new();
    let controls_only = fixture.source_arena_layouts()[0];
    let combined = fixture.source_storage_layouts()[0];
    assert!(combined > controls_only);
    let (error, pool) = component(controls_only, 1, 0, |controls, observer, pool, bank| {
        (
            fixture.funded_source_refusal(controls, observer, bank, 0),
            pool.clone(),
        )
    });
    let GgufHostCopyCause::SourceFunding(cause) = error.cause() else {
        panic!("expected reached combined debit refusal")
    };
    assert!(
        matches!(cause.cause(), HostDestinationCause::Capacity { required, remaining }
        if *required == combined && *remaining == controls_only)
    );
    drop(fixture);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(error);
    settle_pool(&pool);
}

#[test]
fn immutable_source_component_keeps_same_birth_and_hold_through_final_alias() {
    let fixture = OriginalGgufMissFixture::new();
    let layouts = fixture.source_storage_layouts();
    let backing = immutable_backing(&fixture);
    let plan = fixture
        .source
        .conversion_plan(&OriginalGgufMissFixture::request())
        .unwrap();
    let bound = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
    let (array, alias, pool, allocation) = component_destinations(
        (layouts.iter().sum::<u64>() + super::super::super::super::cache::control_bytes().unwrap())
            * 3,
        (layouts.len() + 1) * 3,
        3,
        Some((bound.bytes() as u64 * 3, bound.calls() * 3, 3)),
        |controls, observer, pool, bank| {
            let array = fixture.funded_destinations_miss_hit_miss(controls, observer, bank);
            assert_eq!(
                super::values(&array, Some(observer)),
                fixture.expected["bank.weight"]
            );
            let allocation = array.try_allocation_info().unwrap().unwrap();
            assert_eq!(allocation.bytes() as u64, backing);
            // Clone while the genuinely funded Graph/Record support is active.
            let alias = array.clone();
            let alias_info = alias.try_allocation_info().unwrap().unwrap();
            assert_eq!(alias_info.identity(), allocation.identity());
            assert_eq!(alias_info.bytes(), allocation.bytes());
            (array, alias, pool.clone(), allocation)
        },
    );
    drop((plan, fixture));
    let held = pool.used_bytes().unwrap();
    assert!(held > 0);
    drop(array);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    let alias_info = alias.try_allocation_info().unwrap().unwrap();
    assert_eq!(alias_info.identity(), allocation.identity());
    assert_eq!(alias_info.bytes(), allocation.bytes());
    // The source debit supplies no mutable/ordinary publication certificate.
    assert!(alias.inspect_original_buffer_alias().unwrap().is_none());
    assert!(matches!(
        alias.inspect_ordinary_buffer().unwrap(),
        safemlx::OrdinaryBufferInspection::Unknown
    ));
    std::thread::spawn(move || drop(alias)).join().unwrap();
    settle_pool(&pool);
}

#[path = "funded_component/publication.rs"]
mod publication;

#[path = "funded_component/cache_metadata.rs"]
mod cache_metadata;

#[path = "funded_component/acquisition.rs"]
mod acquisition;

#[path = "funded_component/materialization_submission.rs"]
mod materialization_submission;

#[test]
fn original_fixed_collector_deduplicates_refuses_before_clone_and_retires_final_alias() {
    use crate::backend::runtime::residency::{manager::ResidencyError, storage::RetainedStorage};
    let Some(one) = RetainedStorage::original_collector_control_bytes(1) else {
        assert_ne!(
            std::env::var("EREDU_REQUIRE_QUALIFIED_NATIVE_COLLECTOR").as_deref(),
            Ok("1")
        );
        return;
    };
    // These ordinary numerical sources are fixture inputs. The component below
    // funds only its two real inventory/clone destinations, not source creation.
    let source = Array::from_slice(&[7u32, 13, 29], &[1, 3]);
    let alias = source.clone();
    let distinct = Array::from_slice(&[31u32, 37, 41], &[1, 3]);
    let facts = source.try_allocation_info().unwrap().unwrap();
    let mut ordinary = RetainedStorage::default();
    ordinary.include_array(&source).unwrap();
    let mut ordinary = ordinary.into_retained_arrays().unwrap();
    let ordinary_root = ordinary.next().unwrap();
    assert!(ordinary.next().is_none());
    assert_eq!(
        ordinary_root
            .evaluated()
            .unwrap()
            .try_as_slice::<u32>()
            .unwrap(),
        &[7, 13, 29]
    );
    drop((ordinary_root, ordinary));
    assert_eq!(alias.try_allocation_info().unwrap(), Some(facts));
    assert_ne!(
        distinct.try_allocation_info().unwrap().unwrap().identity(),
        facts.identity()
    );
    let bytes = u64::try_from(one + size_of::<RetainedStorage>()).unwrap();
    let (first, last, pool) = component(
        bytes.checked_mul(2).unwrap(),
        2,
        0,
        |controls, _observer, pool, mut bank| {
            let receipt = bank.try_debit(bytes).unwrap();
            let mut first =
                RetainedStorage::prepare_original(1, controls.metadata_custody()).unwrap();
            drop(receipt);
            assert_eq!(first.original_clone_progress(), Some((0, 1, 0)));
            first.include_array(&source).unwrap();
            assert_eq!(first.original_clone_progress(), Some((1, 0, 1)));
            first.include_array(&alias).unwrap();
            assert_eq!(first.original_clone_progress(), Some((1, 0, 1)));
            for _ in 0..3 {
                assert!(matches!(
                    first.include_array(&distinct),
                    Err(ResidencyError::OriginalInventory(
                        WorkingMemoryError::CollectorCapacity {
                            kind: eredu_runtime::working_memory::CollectorCapacityKind::InventoryRows,
                            used: 1,
                            capacity: 1,
                        }
                    ))
                ));
                // The actual fixed clone worker was never entered by either
                // deduplication or refusal; no extra C handle fill is attempted.
                assert_eq!(first.original_clone_progress(), Some((1, 0, 1)));
                assert_eq!(first.byte_bound().unwrap(), Some(facts.bytes() as u64));
            }
            let receipt = bank.try_debit(bytes).unwrap();
            let mut last =
                RetainedStorage::prepare_original(1, controls.metadata_custody()).unwrap();
            drop(receipt);
            last.include_array(&alias).unwrap();
            assert_eq!(last.byte_bound().unwrap(), first.byte_bound().unwrap());
            assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
            drop((bank, source, alias, distinct));
            (first, last, pool.clone())
        },
    );
    let held = pool.used_bytes().unwrap();
    assert!(held > 0);
    // Original bare extraction returns the actual inventory intact. It cannot
    // yield a prepared Array shell after discarding the shell's raw custody.
    let first = match first.into_retained_arrays() {
        Err((ResidencyError::OriginalInventory(WorkingMemoryError::IdentityMismatch), storage)) => {
            storage
        }
        Err((cause, _)) => panic!("unexpected extraction refusal: {cause}"),
        Ok(_) => panic!("original clone shell escaped its inventory"),
    };
    assert_eq!(first.original_clone_progress(), Some((1, 0, 1)));
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(first);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert_eq!(last.byte_bound().unwrap(), Some(facts.bytes() as u64));
    drop(last);
    settle_pool(&pool);
}

#[path = "funded_component/affine_tile.rs"]
mod affine_tile;

#[path = "funded_component/encoded_input_affine.rs"]
mod encoded_input_affine;
