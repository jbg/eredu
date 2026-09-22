mod quarantine_pins;
mod registered_sources;

use super::*;
use crate::working_memory::{
    InferenceWorkspaceSpan, plan_prefill_with_capacity, quote_inference_workspace,
};
use eredu_core::{
    AdmissionRejection, CacheStateStrategy, EstimationCompleteness, InputModalities,
    InputTokenCount, LayerSchedule, Observed, OutputDemand, StateMemoryLayout,
};
use eredu_nn::{Error, Tensor, workspace::*};
use std::{cell::Cell, num::NonZeroU8};

mod capacity_handoff;
mod controller;
mod incomplete;
mod terminal;

#[derive(Debug, Default)]
struct Facts {
    missing_tensor: bool,
    missing_host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if self.missing_tensor {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| {
                    if matches!(
                        operation.kind,
                        WorkspaceOperationKind::View(_)
                            | WorkspaceOperationKind::Transpose(_)
                            | WorkspaceOperationKind::Index { .. }
                            | WorkspaceOperationKind::StaticSlice { .. }
                    ) {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        layout.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "portable fixture: exact output allocations and storage-sharing views"
                .into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no disjoint host allocation".into(),
        }))
    }
}

fn capacity_numbers(error: &WorkingMemoryError) -> Option<(u64, u64)> {
    match error {
        WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded {
            domain,
            limit_bytes,
            existing_bytes,
            requested_bytes,
        }) => {
            assert_eq!(
                *domain,
                crate::working_memory::memory_fixture::host_topology().host_domain()
            );
            Some((
                *requested_bytes,
                limit_bytes.checked_sub(*existing_bytes).unwrap_or(0),
            ))
        }
        WorkingMemoryError::DomainAllowanceExceeded {
            domain,
            required_bytes,
            available_bytes,
        } => {
            assert_eq!(
                *domain,
                crate::working_memory::memory_fixture::host_topology().host_domain()
            );
            Some((*required_bytes, *available_bytes))
        }
        _ => None,
    }
}
fn publication_controls() -> u64 {
    crate::working_memory::StoragePublicationLayout::<u32>::new(1)
        .unwrap()
        .requested_bytes()
        .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())
        .unwrap()
}
fn reservation_controls_used(pool: &MemoryLedger) -> u64 {
    pool.snapshot()
        .unwrap()
        .domains
        .iter()
        .map(|d| d.reservation_control_bytes)
        .sum()
}
fn physical_used(pool: &MemoryLedger) -> u64 {
    let snapshot = pool.snapshot().unwrap();
    let host = snapshot
        .domains
        .iter()
        .find(|domain| domain.domain == pool.topology().host_domain())
        .unwrap();
    pool.payload_used_bytes()
        .unwrap()
        .checked_add(host.registry_metadata_bytes)
        .unwrap()
        .checked_add(host.reservation_control_bytes)
        .unwrap()
}
fn exact_capacity<Q: FixtureQuote + ?Sized>(pool: &MemoryLedger, quote: &Q) -> u64 {
    physical_used(pool)
        .checked_add(quote_reservation_bytes(quote))
        .unwrap()
}
trait FixtureQuote {
    fn fixture_proof(&self) -> &IncrementalInferenceQuote;
}
impl FixtureQuote for IncrementalInferenceQuote {
    fn fixture_proof(&self) -> &IncrementalInferenceQuote {
        self
    }
}
impl<K: Ord + Send + 'static> FixtureQuote for ResidualInferenceQuote<K> {
    fn fixture_proof(&self) -> &IncrementalInferenceQuote {
        &self.proof
    }
}
impl<Q: FixtureQuote + ?Sized> FixtureQuote for &Q {
    fn fixture_proof(&self) -> &IncrementalInferenceQuote {
        (*self).fixture_proof()
    }
}
fn fixture_admission(q: &IncrementalInferenceQuote) -> Admission {
    let g = q.geometry();
    Admission {
        requested_positions: g.cached_positions + g.input_positions + g.max_output_tokens,
        state: q.state().clone(),
        incremental_required_bytes: q.incremental_bytes(),
        memory_limits: Default::default(),
        additional_headroom: Default::default(),
    }
}
fn quote_metadata_bytes<Q: FixtureQuote + ?Sized>(quote: &Q) -> u64 {
    let q = quote.fixture_proof();
    if q.metadata_funding().is_some() {
        return 0;
    }
    let admission = fixture_admission(q);
    let requirements = q.reservation_requirements(&admission).unwrap();
    requirements
        .get(q.pool().topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
        .checked_sub(
            q.incremental_requirements()
                .unwrap()
                .get(q.pool().topology().host_domain())
                .unwrap()
                .total()
                .unwrap(),
        )
        .unwrap()
}
fn payload_capacity_with_quote<Q: FixtureQuote + ?Sized>(
    pool: &MemoryLedger,
    quote: &Q,
    request: &AdmissionRequest,
    payload_capacity: u64,
) -> u64 {
    if payload_capacity == u64::MAX {
        return payload_capacity;
    }
    let q = quote.fixture_proof();
    let mut admission = fixture_admission(q);
    admission.memory_limits = request.memory_limits.clone();
    admission.additional_headroom = request.additional_headroom.clone();
    let controls = if q.metadata_funding().is_some() {
        0
    } else {
        let Ok(requirements) = q.reservation_requirements(&admission) else {
            return u64::MAX;
        };
        let payload = q
            .incremental_requirements()
            .unwrap()
            .checked_add(
                &admission
                    .additional_headroom
                    .resolve(pool.topology())
                    .unwrap(),
            )
            .unwrap();
        requirements
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
            .checked_sub(
                payload
                    .get(pool.topology().host_domain())
                    .unwrap()
                    .total()
                    .unwrap(),
            )
            .unwrap()
    };
    payload_capacity
        .checked_add(physical_used(pool) - pool.payload_used_bytes().unwrap())
        .unwrap()
        .checked_add(controls)
        .unwrap()
}
fn quote_reservation_bytes<Q: FixtureQuote + ?Sized>(quote: &Q) -> u64 {
    quote
        .fixture_proof()
        .incremental_bytes()
        .unwrap()
        .checked_add(quote_metadata_bytes(quote))
        .unwrap()
}
fn reservation_control_bytes(r: &WorkingMemoryReservation) -> u64 {
    r.0.pool
        .0
        .usage
        .lock()
        .unwrap()
        .funding
        .constructor_control_bytes(r.0.account_id)
        .unwrap()
}
fn reservation_payload_bytes(r: &WorkingMemoryReservation) -> u64 {
    r.requirements()
        .get(r.0.pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
        .checked_sub(reservation_control_bytes(r))
        .unwrap()
}

fn placed_root(bytes: Option<u64>, context: &WorkspaceContext) -> WorkspaceExistingStorage {
    WorkspaceExistingStorage::try_new_placed(
        bytes,
        crate::working_memory::memory_fixture::host_placement(),
        context,
    )
    .unwrap()
}
fn fixture_requirements(bytes: u64) -> eredu_core::DomainMemoryRequirements {
    let topology = crate::working_memory::memory_fixture::host_topology();
    let mut result = eredu_core::DomainMemoryRequirements::zero(&topology);
    result
        .add_allocation(
            bytes,
            crate::working_memory::memory_fixture::host_placement(),
        )
        .unwrap();
    result
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 1,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn request(g: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
    }
}
fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "portable residual fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    }
}
fn state(g: InferenceGeometry) -> RuntimeStateEstimate {
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let mut value = eredu_core::estimate_runtime_state(
        &layout,
        request(g).input,
        g.max_output_tokens,
        g.batch_size,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    value.physical_domains = Some(eredu_core::DomainRuntimeStateEstimate {
        geometry: g,
        decoder_state: fixture_requirements(0),
        media_embeddings: fixture_requirements(0),
        media_workspace: fixture_requirements(0),
    });
    value
}
fn outside(g: InferenceGeometry, bytes: u64) -> ExecutionWorkspaceEstimate {
    let bound = |n| WorkspaceBound::bounded(n, "complete enclosing fixture allocation bound");
    ExecutionWorkspaceEstimate {
        geometry: g,
        activations: bound(bytes),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(0),
        physical_domains: Some(eredu_core::DomainExecutionWorkspaceEstimate {
            geometry: g,
            activations: fixture_requirements(bytes),
            attention: fixture_requirements(0),
            vocabulary: fixture_requirements(0),
            state_update: fixture_requirements(0),
            materialization: fixture_requirements(0),
            retained: fixture_requirements(0),
        }),
    }
}
fn view(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    elements: i32,
) -> WorkspaceTensor {
    WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[elements], WorkspaceDtype::Float32).unwrap(),
        root,
        context,
    )
    .unwrap()
}
fn used(pool: &MemoryLedger) -> (u64, u64) {
    (
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
    )
}
fn replacement_report(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    g: InferenceGeometry,
) -> InferenceWorkspaceReport {
    let mut current = view(context, root, 2);
    quote_inference_workspace(g, |span| {
        context.begin_state_span([&current])?;
        if let InferenceWorkspaceSpan::Decode { index, .. } = span {
            // A has 64 bytes of capacity despite this initial eight-byte view.
            // B=16 replaces it; the later C=80 overlaps B, not A.
            current =
                WorkspaceTensor::full_f32(0.75, &[if *index == 0 { 4 } else { 20 }], context)?;
        }
        context.report(&[current.clone()])
    })
    .unwrap()
}
fn replacement_quote(
    pool: &MemoryLedger,
    g: InferenceGeometry,
    extra: u64,
) -> ResidualInferenceQuote<u32> {
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let storage = RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, g);
    ResidualInferenceQuote::compose(&report, state(g), outside(g, extra), &storage).unwrap()
}
fn plan(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    quote: &ResidualInferenceQuote<u32>,
    request: AdmissionRequest,
    capacity: u64,
) -> Result<(WorkingMemoryReservation, ResidualInferenceQuote<u32>), PrefillPlanningError> {
    let capacity = payload_capacity_with_quote(pool, quote, &request, capacity);
    plan_prefill_residual_with_capacity(
        execution,
        pool,
        &capabilities(),
        request,
        quote.geometry(),
        crate::working_memory::memory_fixture::resolved_host_limits(pool, capacity),
        |_| Ok(quote.clone()),
    )
}

#[test]
fn replacement_residual_uses_each_span_identity_union_instead_of_scalar_subtraction() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let escaped = original.clone();
    let quote = replacement_quote(&pool, geometry(), 0);
    assert_eq!(quote.state().requested_state_bytes, 80);
    assert_eq!(
        quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap(),
        Some(64)
    );
    assert_eq!(quote.incremental_bytes().unwrap(), 96);
    let full = quote.state().requested_state_bytes
        + quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap();
    assert_eq!(full, 144);
    assert!(
        quote.incremental_bytes().unwrap() > full - 64,
        "the later B+C peak contains no A to subtract"
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        plan(&pool, &execution, &quote, request(geometry()), 159),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((required, available)) if required == available + 1)));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let (reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    assert_eq!(reservation.admission().state, *quote.state());
    assert_eq!(reservation_payload_bytes(&reservation), 96);
    assert!(reservation.requires_funding_scope());
    assert_eq!(pool.payload_used_bytes().unwrap(), 160);
    // Residual diagnostics are not an ordinary complete admission or source proof.
    assert!(matches!(
        pool.reserve(&execution, reservation.admission()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut complete = reservation.admission().clone();
    complete.incremental_required_bytes = Some(full);
    assert!(matches!(
        pool.reserve(&execution, &complete),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    drop((reservation, accepted, quote, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    drop(escaped);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn binding_requires_exact_registered_identity_bijection_and_known_capacities() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64), (2, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let other = WorkspaceContext::new(Facts::default());
    let a = placed_root(Some(64), &context);
    let b = placed_root(Some(64), &context);
    let foreign = placed_root(Some(64), &other);
    let mismatch = placed_root(Some(63), &context);
    let unknown = placed_root(None, &context);
    for inventory in [
        vec![(1u32, a.clone()), (1, b.clone())],
        vec![(1, a.clone()), (2, a.clone())],
        vec![(1, a.clone()), (3, b.clone())],
        vec![(1, a.clone()), (2, mismatch)],
        vec![(1, foreign)],
    ] {
        assert!(matches!(
            RegisteredWorkspaceStorage::bind(&pool, &context, inventory),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), 128);
        assert!(context.report(&[]).unwrap().residual.is_none());
    }
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, unknown)]),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let foreign_pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&foreign_pool, &context, [(1u32, a.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u64, a.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let storage = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        [(1u32, a.clone()), (1, a.clone()), (2, b.clone())],
    )
    .unwrap();
    assert_eq!(storage.borrowed_storage().total_bytes(), Some(128));
    assert_eq!(storage.borrowed_storage().roots().len(), 2);
    assert!(
        storage
            .borrowed_storage()
            .same_identity(storage.clone().borrowed_storage())
    );
    // A second selection is rejected and its temporary registry pin rolls back.
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, a)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop((storage, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn borrowed_aliases_are_free_but_equal_sized_independent_roots_remain_incremental() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let alias = view(&context, &root, 2);
    let independent = placed_root(Some(64), &context);
    let independent = view(&context, &independent, 2);
    let report = quote_inference_workspace(geometry(), |_| {
        context.begin_state_span([&alias, &alias, &independent])?;
        let reshaped = alias.reshape(&[1, 2], &context)?;
        context.report(&[alias.clone(), reshaped, independent.clone()])
    })
    .unwrap();
    let quote = ResidualInferenceQuote::compose(
        &report,
        state(geometry()),
        outside(geometry(), 0),
        &storage,
    )
    .unwrap();
    assert_eq!(quote.state().requested_state_bytes, 128);
    assert_eq!(quote.incremental_bytes().unwrap(), 64);
    drop((quote, storage, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn residual_composition_and_planning_bind_geometry_selection_and_domain() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, geometry());
    let other_context = WorkspaceContext::new(Facts::default());
    let other_root = placed_root(Some(64), &other_context);
    let other_storage =
        RegisteredWorkspaceStorage::bind(&pool, &other_context, [(1u32, other_root)]).unwrap();
    assert!(matches!(
        ResidualInferenceQuote::compose(
            &report,
            state(geometry()),
            outside(geometry(), 0),
            &other_storage
        ),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let changed = InferenceGeometry {
        max_output_tokens: 3,
        ..geometry()
    };
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(geometry()), outside(changed, 0), &storage),
        Err(ResidualQuoteError::Estimate(_))
    ));
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(changed), outside(geometry(), 0), &storage),
        Err(ResidualQuoteError::Estimate(_))
    ));
    let quote = ResidualInferenceQuote::compose(
        &report,
        state(geometry()),
        outside(geometry(), 0),
        &storage,
    )
    .unwrap();
    let foreign_pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        plan(&foreign_pool, &execution, &quote, request(geometry()), 1000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let invoked = Cell::new(0);
    let mut bad_request = request(geometry());
    bad_request.max_output_tokens += 1;
    assert!(matches!(
        plan_prefill_residual_with_capacity(
            &execution,
            &pool,
            &capabilities(),
            bad_request,
            geometry(),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1000),
            |_| {
                invoked.set(invoked.get() + 1);
                Ok(quote.clone())
            }
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(invoked.get(), 0);
    assert!(matches!(
        plan_prefill_residual_with_capacity(
            &execution,
            &pool,
            &capabilities(),
            request(changed),
            changed,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1000),
            |_| Ok(quote.clone())
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    assert_eq!(foreign_pool.payload_used_bytes().unwrap(), 0);
    drop((quote, storage, other_storage, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn missing_state_spans_tensor_host_or_full_coverage_never_become_residual_permission() {
    for gap in 0..3 {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let original = pool.register_host_storage([(1u32, 64)]).unwrap();
        let context = WorkspaceContext::new(Facts {
            missing_tensor: gap == 1,
            missing_host: gap == 2,
        });
        let root = placed_root(Some(64), &context);
        let storage =
            RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
        let old = view(&context, &root, 2);
        let report = quote_inference_workspace(geometry(), |span| {
            if gap == 0 && matches!(span, InferenceWorkspaceSpan::Decode { index: 0, .. }) {
                context.begin_span();
            } else {
                context.begin_state_span([&old])?;
            }
            let _temporary = old.square(&context)?;
            context.report(&[old.clone()])
        })
        .unwrap();
        assert_eq!(report.completed_spans(), 3);
        assert!(report.first_gap().is_some());
        let result = ResidualInferenceQuote::compose(
            &report,
            state(geometry()),
            outside(geometry(), 0),
            &storage,
        );
        if gap == 0 {
            assert!(matches!(
                result,
                Err(ResidualQuoteError::Storage(
                    WorkingMemoryError::UnknownBound
                ))
            ));
        } else {
            assert!(matches!(
                result,
                Err(ResidualQuoteError::IncompleteWorkspace(_))
            ));
        }
        drop((storage, original));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, geometry());
    let mut unpriced = outside(geometry(), 0);
    unpriced.retained = WorkspaceBound::Unknown {
        reason: "controller payload unpriced".into(),
    };
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(geometry()), unpriced, &storage),
        Err(ResidualQuoteError::IncompleteWorkspace(_))
    ));
    let error = ResidualInferenceQuote::compose(
        &report,
        state(geometry()),
        outside(geometry(), u64::MAX),
        &storage,
    )
    .unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = match &error {
        ResidualQuoteError::Estimate(cause) => cause,
        _ => &error,
    };
    loop {
        if matches!(
            cause.downcast_ref::<eredu_core::MemoryDomainError>(),
            Some(eredu_core::MemoryDomainError::Overflow)
        ) || matches!(
            cause.downcast_ref::<WorkingMemoryError>(),
            Some(WorkingMemoryError::Overflow)
        ) || matches!(
            cause.downcast_ref::<eredu_core::CapabilityError>(),
            Some(
                eredu_core::CapabilityError::ArithmeticOverflow { .. }
                    | eredu_core::CapabilityError::MemoryDomain(
                        eredu_core::MemoryDomainError::Overflow
                    )
            )
        ) || matches!(
            cause.downcast_ref::<eredu_core::AdmissionPolicyError>(),
            Some(eredu_core::AdmissionPolicyError::ArithmeticOverflow { .. })
        ) {
            break;
        }
        cause = cause.source().unwrap_or_else(|| {
            panic!("checked overflow is preserved in the public error chain: {error:?}")
        });
    }
    let mut partial_state = state(geometry());
    partial_state.persistent_state_completeness = EstimationCompleteness::PersistentStateOnly;
    let partial =
        ResidualInferenceQuote::compose(&report, partial_state, outside(geometry(), 0), &storage)
            .unwrap();
    let mut permissive = request(geometry());
    assert!(matches!(
        plan(
            &pool,
            &InferenceExecutionIdentity::default(),
            &partial,
            permissive,
            1000
        ),
        Err(PrefillPlanningError::Admission(
            AdmissionRejection::EstimationUnsupported { .. }
        ))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    drop((partial, storage, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn different_borrowed_tokens_between_completed_spans_reject_the_whole_trace() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let a = WorkspaceContext::new(Facts::default());
    let b = WorkspaceContext::new(Facts::default());
    let a_root = placed_root(Some(64), &a);
    let b_root = placed_root(Some(64), &b);
    let a_storage = RegisteredWorkspaceStorage::bind(&pool, &a, [(1u32, a_root.clone())]).unwrap();
    let b_storage = RegisteredWorkspaceStorage::bind(&pool, &b, [(1u32, b_root.clone())]).unwrap();
    let a_view = view(&a, &a_root, 2);
    let b_view = view(&b, &b_root, 2);
    let result = quote_inference_workspace(geometry(), |span| {
        let (context, root) = if matches!(span, InferenceWorkspaceSpan::Prefill(_)) {
            (&a, &a_view)
        } else {
            (&b, &b_view)
        };
        context.begin_state_span([root])?;
        context.report(&[root.clone()])
    });
    assert!(matches!(
        result,
        Err(crate::working_memory::InferenceWorkspaceError::Geometry(_))
    ));
    drop((a_storage, b_storage, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn application_budget_safety_and_domain_capacity_use_complete_incremental_cost() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 8);
    assert_eq!(quote.incremental_bytes().unwrap(), 104);
    let execution = InferenceExecutionIdentity::default();
    let mut request = request(geometry());
    request.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 7)]);
    request.memory_limits = crate::working_memory::memory_fixture::host_limits(0);
    let exact = payload_capacity_with_quote(&pool, &quote, &request, 175);
    request.memory_limits = crate::working_memory::memory_fixture::host_limits(exact - 1);
    assert!(matches!(
        plan(&pool, &execution, &quote, request.clone(), 1000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    request.memory_limits = crate::working_memory::memory_fixture::host_limits(exact);
    assert!(matches!(
        plan(&pool, &execution, &quote, request.clone(), 174),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((required, available)) if required == available + 1)));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let (reservation, accepted) = plan(&pool, &execution, &quote, request, 175).unwrap();
    assert_eq!(
        reservation.admission().incremental_required_bytes,
        Some(104)
    );
    assert_eq!(reservation.admission().state, *quote.state());
    assert_eq!(pool.payload_used_bytes().unwrap(), 175);
    drop((reservation, accepted, quote, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

fn chunk_quote(pool: &MemoryLedger, g: InferenceGeometry) -> ResidualInferenceQuote<u32> {
    let context = WorkspaceContext::new(Facts::default());
    let root = placed_root(Some(64), &context);
    let storage = RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, root.clone())]).unwrap();
    let old = view(&context, &root, 2);
    let report = quote_inference_workspace(g, |span| {
        context.begin_state_span([&old])?;
        let count = match span {
            InferenceWorkspaceSpan::Sampling(_) => unreachable!("model-only traversal fixture"),
            InferenceWorkspaceSpan::Prefill(chunk) => chunk.input.end - chunk.input.start,
            InferenceWorkspaceSpan::Decode { .. } => 1,
        };
        let _temporary = WorkspaceTensor::full_f32(0.5, &[count as i32 * 4], &context)?;
        context.report(&[old.clone()])
    })
    .unwrap();
    ResidualInferenceQuote::compose(&report, state(g), outside(g, 0), &storage).unwrap()
}

#[test]
fn residual_chunk_retries_share_planner_policy_without_weakening_full_reservations() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let g = InferenceGeometry {
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 3,
        ..geometry()
    };
    let execution = InferenceExecutionIdentity::default();
    let probe = chunk_quote(
        &pool,
        InferenceGeometry {
            prefill_chunk_positions: 2,
            ..g
        },
    );
    let capacity = exact_capacity(&pool, &probe);
    drop(probe);
    let mut full_candidates = Vec::new();
    assert!(matches!(
        plan_prefill_with_capacity(
            &execution,
            &pool,
            &capabilities(),
            request(g),
            g,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
            |candidate| {
                full_candidates.push(candidate.prefill_chunk_positions);
                Ok(chunk_quote(&pool, candidate).state().clone())
            }
        ),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((required, available)) if required > available)));
    assert_eq!(full_candidates, [3, 2, 1]);
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let mut residual_candidates = Vec::new();
    let (reservation, accepted) = plan_prefill_residual_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(g),
        g,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        |candidate| {
            residual_candidates.push(candidate.prefill_chunk_positions);
            Ok(chunk_quote(&pool, candidate))
        },
    )
    .unwrap();
    assert_eq!(residual_candidates, [3, 2]);
    assert_eq!(reservation.geometry().prefill_chunk_positions, 2);
    assert_eq!(reservation.admission().incremental_required_bytes, Some(32));
    assert_eq!(accepted.state().requested_state_bytes, 64);
    assert_eq!(pool.payload_used_bytes().unwrap(), 96);
    drop((reservation, accepted, original));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn residual_reservation_retains_borrowed_charge_after_quote_and_original_owner_drop() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0);
    let execution = InferenceExecutionIdentity::default();
    let (reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    let alias = reservation.clone();
    drop((accepted, quote, original, reservation));
    assert_eq!(pool.payload_used_bytes().unwrap(), 160);
    assert!(alias.requires_funding_scope());
    alias.validate(&execution, geometry()).unwrap();
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 1_000_000);
}

#[test]
fn conversion_moves_pins_to_live_run_and_scopes_while_metadata_remains_cold_evidence() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), publication_controls());
    let execution = InferenceExecutionIdentity::default();
    let (reservation, accepted) = plan(
        &pool,
        &execution,
        &quote,
        request(geometry()),
        160 + publication_controls(),
    )
    .unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    let output = scope.adopt_host_storage_individually([(2u32, 80)]).unwrap();
    drop((accepted, quote, original));
    run.close().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 160);
    scope.certify().unwrap();
    // The old borrowed A and unused workspace expired. Only C remains charged.
    assert_eq!(pool.payload_used_bytes().unwrap(), 80);
    assert!(metadata.requires_funding_scope());
    metadata.validate(&execution, geometry()).unwrap();
    assert_eq!(
        reservation_payload_bytes(&metadata),
        96 + publication_controls()
    );
    drop(output);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(metadata);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn uncertified_residual_scope_keeps_borrowed_charge_and_remaining_envelope() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let original = pool.register_host_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0);
    let execution = InferenceExecutionIdentity::default();
    let (reservation, accepted) =
        plan(&pool, &execution, &quote, request(geometry()), 160).unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    drop((accepted, quote, original, metadata, run));
    drop(scope);
    assert_eq!(pool.payload_used_bytes().unwrap(), 160);
    assert_eq!(
        pool.payload_effective_capacity().unwrap(),
        physical_used(&pool)
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

mod span_workspace;

#[test]
fn finite_binding_consumes_counted_rows_and_keeps_exact_existing_only_pins() {
    if !crate::working_memory::qualified_storage::qualified() {
        return;
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = pool.register_host_storage([(1u32, 64), (2, 32)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let a = placed_root(Some(64), &context);
    let b = placed_root(Some(32), &context);
    let initial = used(&pool);
    assert!(matches!(
        RegisteredWorkspaceStorageLayout::<u32>::new(1)
            .unwrap()
            .construct(&pool, &context, [(1, a.clone()), (2, b.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(used(&pool), initial);
    assert!(context.report(&[]).unwrap().residual.is_none());
    let layout = RegisteredWorkspaceStorageLayout::<u32>::new(3).unwrap();
    assert_eq!(layout.source_slots(), 3);
    assert!(layout.requested_bytes() > 0);
    let registered = layout
        .construct(&pool, &context, [(2, b), (1, a.clone()), (1, a.clone())])
        .unwrap();
    assert_eq!(registered.borrowed_storage().roots().len(), 2);
    assert!(registered.borrowed_storage().roots()[0].same_storage(&a));
    assert_eq!(registered.borrowed_storage().total_bytes(), Some(96));
    assert_eq!(used(&pool), initial);
    let shared = registered.clone();
    drop((registered, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 96);
    drop(shared);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(matches!(
        RegisteredWorkspaceStorageLayout::<u32>::new(usize::MAX),
        Err(WorkingMemoryError::Overflow)
    ));
}

mod reservation_metadata;

#[test]
fn rejected_residual_keeps_all_physical_categories_peak_and_identifiers_unchanged() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let _source = pool.register_host_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0);
    let before = pool.snapshot().unwrap();
    let next_funding = pool.0.usage.lock().unwrap().next_funding;
    let error = plan(
        &pool,
        &InferenceExecutionIdentity::default(),
        &quote,
        request(geometry()),
        159,
    )
    .unwrap_err();
    let PrefillPlanningError::Reservation(cause) = error else {
        panic!("physical capacity rejection")
    };
    let (required, available) = capacity_numbers(&cause).expect("typed host-domain capacity");
    assert_eq!(required, available + 1);
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, next_funding);
}

#[test]
fn registered_backing_controls_require_their_actual_canonical_source_pin() {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let storage = pool.register_host_storage([(1u32, 64), (2, 7)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::try_new_placed_with_host_controls(
        Some(64),
        pool.host_placement(),
        Some(7),
        &context,
    )
    .unwrap();
    let before = pool.payload_used_bytes().unwrap();
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(
            &pool,
            &context,
            [RegisteredWorkspaceStorageRow::new(1u32, root.clone()).with_host_controls(9)]
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), before);
    let source = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        [RegisteredWorkspaceStorageRow::new(1u32, root.clone()).with_host_controls(2)],
    )
    .unwrap();
    assert_eq!(
        source
            .borrowed_storage()
            .requirements(&context)
            .unwrap()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        71
    );
    drop(storage);
    assert_eq!(pool.payload_used_bytes().unwrap(), 71);
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn conflicting_backing_controls_reject_before_installing_borrowed_selection() {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let storage = pool.register_host_storage([(1u32, 64), (2, 7)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::try_new_placed_with_host_controls(
        Some(64),
        pool.host_placement(),
        Some(8),
        &context,
    )
    .unwrap();
    assert!(matches!(
        RegisteredWorkspaceStorage::bind(
            &pool,
            &context,
            [RegisteredWorkspaceStorageRow::new(1u32, root).with_host_controls(2)]
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let exact = WorkspaceExistingStorage::try_new_placed_with_host_controls(
        Some(64),
        pool.host_placement(),
        Some(7),
        &context,
    )
    .unwrap();
    let bound = RegisteredWorkspaceStorage::bind(
        &pool,
        &context,
        [RegisteredWorkspaceStorageRow::new(1u32, exact).with_host_controls(2)],
    )
    .unwrap();
    drop((bound, storage));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
