//! Private, fixed-producer certificate. No TextGeneration/model fit is enabled.
use super::*;
use eredu_core::{
    AdmissionRequest, CacheStateStrategy, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputModalities, InputTokenCount, LayerSchedule, ModelCapabilities,
    Observed, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_nn::workspace::*;
use safemlx::{
    AllocationIdentity, OriginalMutablePairCause, OriginalMutablePairCustodies,
    OriginalMutablePairPlan,
};
use std::{alloc::Layout, mem::size_of};

fn require_qualified_or_report_unknown(kind: &str) {
    assert_ne!(
        std::env::var_os("EREDU_REQUIRE_MUTABLE_PAIR_QUALIFICATION"),
        Some("1".into()),
        "pinned mutable-pair validation requires positive {kind} qualification"
    );
    eprintln!("mutable-pair qualification={kind}-unknown; typed refusal before bank installation");
}
fn component_runtime() -> Option<Rc<PreparedInputRuntime>> {
    let runtime = Rc::new(PreparedInputRuntime::prepare().unwrap());
    let qualification = OriginalMutablePairPlan::inspect(&runtime).map(|_| ());
    match qualification {
        Ok(()) => Some(runtime),
        Err(cause) => {
            assert_eq!(cause, OriginalMutablePairCause::Layout);
            require_qualified_or_report_unknown("native-layout");
            None
        }
    }
}

type PairRegistration = NativeStorageRegistration<AllocationIdentity>;
type PairAttachment = PreparedAllocationOwner<PairRegistration>;
type PairBank = OriginalNativeStorageBank<PairMechanism>;

#[derive(Clone)]
struct PairMechanism {
    runtime: Rc<PreparedInputRuntime>,
    selection: NativeStorageSelection,
    initial_publication:
        Option<crate::backend::runtime::residency::storage::RetainedStoragePublication>,
}
enum PairObservation<'a> {
    Origin(OriginalBufferWitness<'a>),
    Existing(OriginalBufferAliasWitness<'a>),
    Ordinary(OrdinaryBufferWitness<'a>),
}
enum PairCause {
    Fixed(OriginalBufferCause),
    Budget(OriginalBufferCause, OriginalNativeBudgetCustody),
    Attachment(PreparedAllocationOwnerCause, PairRegistration),
}
impl fmt::Debug for PairCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(v) | Self::Budget(v, _) => v.fmt(f),
            Self::Attachment(v, _) => v.fmt(f),
        }
    }
}
impl fmt::Display for PairCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
impl std::error::Error for PairCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Fixed(v) | Self::Budget(v, _) => v,
            Self::Attachment(v, _) => v,
        })
    }
}
impl OriginalNativeStorageMechanism for PairMechanism {
    type Key = AllocationIdentity;
    type Budget = OriginalBufferBudget;
    type Root<'a> = &'a Array;
    type Attachment = PairAttachment;
    type Error = PairCause;
    type Observation<'a> = PairObservation<'a>;
    fn selection(&self) -> &NativeStorageSelection {
        &self.selection
    }
    fn uniform_budget_placement(&self) -> Option<std::sync::Arc<eredu_core::MemoryPlacement>> {
        Some(crate::backend::managed_memory::ledger().host_placement_handle())
    }
    fn key_clone_storage_bytes(&self) -> Option<u64> {
        // This concrete key is scalar namespace + nonrepeating generation. The
        // broad StorageIdentity/source/Weak routes remain unknown in production.
        Some(0)
    }
    fn create_budget(
        &self,
        custody: OriginalNativeBudgetCustody,
    ) -> Result<Self::Budget, PairCause> {
        let capacity = usize::try_from(custody.capacity_bytes()).unwrap();
        let prepared = PreparedOriginalBufferBudget::try_new(
            &self.runtime,
            capacity,
            NativeBudgetOwner(custody),
        )
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            PairCause::Budget(cause, owner.0)
        })?;
        prepared.try_allocate_observed().map_err(|error| {
            let (cause, owner) = error.into_parts();
            PairCause::Budget(cause, owner.into_owner().0)
        })
    }
    fn observe<'a, 'root: 'a>(
        &'a self,
        budget: &'a OriginalBufferBudget,
        root: &'root Array,
    ) -> Result<Self::Observation<'a>, PairCause> {
        match budget.inspect_array(root) {
            Ok(Some(value)) => Ok(PairObservation::Origin(value)),
            Err(OriginalBufferCause::ForeignDomain) => root
                .inspect_original_buffer_alias()
                .map_err(PairCause::Fixed)?
                .map(PairObservation::Existing)
                .ok_or(PairCause::Fixed(OriginalBufferCause::UncertifiedBacking)),
            Err(cause) => Err(PairCause::Fixed(cause)),
            Ok(None) => match root.inspect_ordinary_buffer().map_err(PairCause::Fixed)? {
                OrdinaryBufferInspection::Allocation(witness) => {
                    Ok(PairObservation::Ordinary(witness))
                }
                _ => Err(PairCause::Fixed(OriginalBufferCause::UncertifiedBacking)),
            },
        }
    }
    fn placement(
        _: &Self::Observation<'_>,
        topology: &eredu_core::MemoryTopology,
    ) -> Result<
        std::sync::Arc<eredu_core::MemoryPlacement>,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        Ok(std::sync::Arc::new(eredu_core::MemoryPlacement::fixed(
            topology,
            topology.host_domain(),
        )?))
    }
    fn describe(
        observation: &Self::Observation<'_>,
    ) -> NativeStorageObservation<AllocationIdentity> {
        match observation {
            PairObservation::Origin(witness) => {
                let facts = witness.allocation();
                NativeStorageObservation::Originating(facts.identity(), facts.bytes() as u64)
            }
            PairObservation::Existing(witness) => {
                let facts = witness.allocation();
                NativeStorageObservation::Existing(facts.identity(), facts.bytes() as u64)
            }
            PairObservation::Ordinary(witness) => {
                let facts = witness.allocation();
                NativeStorageObservation::Ordinary(facts.identity(), facts.bytes() as u64)
            }
        }
    }
    fn has_retained_attachment(
        &self,
        previous: &Self::Observation<'_>,
        observation: &Self::Observation<'_>,
        pool: &MemoryLedger,
    ) -> bool {
        let facts = |observation: &Self::Observation<'_>| match observation {
            PairObservation::Origin(witness) => witness.allocation(),
            PairObservation::Existing(witness) => witness.allocation(),
            PairObservation::Ordinary(witness) => witness.allocation(),
        };
        let current = facts(observation);
        if std::mem::discriminant(previous) != std::mem::discriminant(observation)
            || facts(previous) != current
        {
            return false;
        }
        self.initial_publication
            .as_ref()
            .is_some_and(|publication| {
                publication.has_native_attachment(pool.shared_storage_accounting_id(), current)
            })
    }
    fn prepare_attachment(&self, owner: PairRegistration) -> Result<PairAttachment, PairCause> {
        PreparedAllocationOwner::try_new(owner).map_err(|error| {
            let (cause, owner) = error.into_parts();
            PairCause::Attachment(cause, owner)
        })
    }
    fn registration(owner: &PairAttachment) -> &PairRegistration {
        owner.owner()
    }
    fn attach(
        observation: Self::Observation<'_>,
        owner: PairAttachment,
    ) -> Result<(), (PairCause, PairAttachment)> {
        let result = match observation {
            PairObservation::Origin(witness) => witness.try_attach(owner),
            PairObservation::Existing(witness) => witness.try_attach(owner),
            PairObservation::Ordinary(witness) => witness.try_attach(owner),
        };
        result.map_err(|error| {
            let (cause, owner) = error.into_parts();
            (PairCause::Fixed(cause), owner)
        })
    }
}

impl PairMechanism {
    fn new(runtime: &Rc<PreparedInputRuntime>) -> Self {
        Self {
            runtime: runtime.clone(),
            selection: NativeStorageSelection::default(),
            initial_publication: None,
        }
    }
    fn provider_controls(&self, capacity: usize, attempts: usize, rows: usize) -> Option<u64> {
        let budget =
            PreparedOriginalBufferBudget::<NativeBudgetOwner>::layout(&self.runtime, capacity)
                .ok()?;
        let sidecar = PreparedAllocationOwner::<PairRegistration>::layout();
        let one = [
            sidecar.allocation_bytes()?,
            sidecar.preparation_control_bytes(),
            sidecar.prepared_bytes(),
            sidecar.preparation_failure_bytes(),
            sidecar.attachment_failure_bytes(),
            sidecar.original_attachment_control_bytes(),
            OriginalBufferAliasWitness::inspection_control_bytes()?,
            OrdinaryBufferWitness::inspection_control_bytes()?,
            crate::backend::runtime::residency::storage::RetainedStoragePublication::attachment_lookup_control_bytes()?,
            // Borrowed native observations own no key or backing. The neutral
            // qualified registry separately prices all scalar key copies.
            size_of::<PairObservation<'static>>(),
            size_of::<(&PairObservation<'static>, &PairObservation<'static>)>(),
            size_of::<[safemlx::AllocationInfo; 2]>(),
            size_of::<NativeStorageObservation<AllocationIdentity>>(),
            size_of::<Result<PairObservation<'static>, PairCause>>(),
            size_of::<Result<(), (PairCause, PairAttachment)>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        let overlap = [
            size_of::<PairCause>(),
            size_of::<NativeStorageError<PairCause>>(),
            size_of::<PairMechanism>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<Result<OriginalBufferBudget, PairCause>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        let bytes = one
            .checked_mul(attempts)?
            .checked_mul(rows)?
            .checked_add(overlap.checked_mul(attempts.checked_add(2)?)?)?
            .checked_add(budget.total_owner_bytes()?)?;
        u64::try_from(bytes).ok()
    }
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn request() -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(1),
        max_output_tokens: 1,
        batch_size: 1,
        additional_headroom: crate::memory_fixture::headroom(0),
        memory_limits: Default::default(),
    }
}
fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "private fixed U32 pair component".into(),
        native_max_context: Observed::exact(2, "one fixed component"),
        effective_max_context: Observed::exact(2, "one fixed component"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    }
}
fn component_quote(
    pool: &MemoryLedger,
    mechanism: &PairMechanism,
    attempts: usize,
) -> Result<IncrementalInferenceQuote, ResidualQuoteError> {
    let native = OriginalMutablePairPlan::inspect(&mechanism.runtime).unwrap();
    let facts = native.facts();
    let context = WorkspaceContext::new(
        crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap(),
    );
    let storage = RegisteredWorkspaceStorage::<AllocationIdentity>::bind(
        pool,
        &context,
        std::iter::empty::<(AllocationIdentity, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let report = quote_inference_workspace(geometry(), |_| {
        context.begin_state_span(std::iter::empty::<&WorkspaceTensor>())?;
        context.report(&[])
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
        request().input,
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let zero = || {
        WorkspaceBound::bounded(
            0,
            "empty residual component; fixed producer has separate P and Q",
        )
    };
    let outside = crate::memory_fixture::workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: geometry(),
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        // This real producer's independently retained physical backing is P.
        // NativeStorageLayout.capacity describes P but does not reserve it;
        // Q below covers controls only. The empty equation spans stay zero.
        retained: WorkspaceBound::bounded(
            facts.backing_bytes() as u64,
            "one fixed mutable U32 pair physical backing, once outside Q",
        ),
    });
    let quote = ResidualInferenceQuote::compose(&report, state, outside, &storage)
        .unwrap()
        .into_incremental();
    assert_eq!(
        quote.incremental_bytes(),
        Some(facts.backing_bytes() as u64),
        "the original producer reservation contains P exactly once before Q"
    );
    assert!((0..quote.span_workspace().plan().records().len())
        .all(|i| quote.span_workspace().span_bytes(i) == Some(0)));
    let provider_controls = mechanism.provider_controls(facts.backing_bytes(), attempts, 1);
    assert!(
        provider_controls.is_some(),
        "fixed provider layouts must be known"
    );
    let plan = PreparedNativeStoragePlan::<PairMechanism>::prepare_qualified(
        quote.span_workspace(),
        mechanism,
        Some(facts.backing_bytes() as u64),
        Some((attempts, 1)),
        (0..quote.span_workspace().plan().records().len()).map(|_| Some(0)),
        provider_controls,
    )
    .unwrap();
    // All concrete facts above are known. The existing compiler/liballoc
    // qualifier may remain unknown on another toolchain; preserve its actual
    // UnknownBound through sealing instead of replacing any term with zero.
    let qualified = plan.control_bytes().is_some();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        quote.span_workspace().plan(),
        TextHostControlFacts::new(
            Some(facts.control_bytes::<OriginalTextControlGuard>().unwrap() as u64),
            Some(0),
            Some(0),
        ),
    )
    .unwrap()
    .with_native_storage(plan)
    .unwrap();
    let sealed = quote.with_span_workspace_and_text_controls(controls);
    if !qualified {
        assert!(matches!(
            &sealed,
            Err(ResidualQuoteError::Storage(
                WorkingMemoryError::UnknownBound
            ))
        ));
    }
    sealed
}

struct Accepted {
    bank: PairBank,
    span: OwnedTextSpanWorkspace,
    reservation: WorkingMemoryReservation,
    run: WorkingMemoryFundingRun,
}
fn complete_requirements(
    quote: &IncrementalInferenceQuote,
) -> eredu_core::DomainMemoryRequirements {
    quote
        .reservation_requirements(&eredu_core::Admission {
            requested_positions: 2,
            state: quote.state().clone(),
            incremental_required_bytes: quote.incremental_bytes(),
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
        })
        .unwrap()
}
fn shared_capacity(
    pool: &MemoryLedger,
    mechanism: &PairMechanism,
    attempts: [usize; 2],
) -> Option<u64> {
    let mut capacity = pool.fixture_host_current().unwrap();
    // Both live requests must accept the same aggregate ceiling. Compute their
    // actual control requirements before admitting A; its retained origin must
    // not impose a smaller ceiling that excludes all of B's future controls.
    for count in attempts {
        match component_quote(pool, mechanism, count) {
            Ok(quote) => {
                capacity = capacity
                    .checked_add(crate::memory_fixture::host_total(&complete_requirements(
                        &quote,
                    )))
                    .unwrap()
            }
            Err(ResidualQuoteError::Storage(WorkingMemoryError::UnknownBound)) => {
                require_qualified_or_report_unknown("rust-managed");
                return None;
            }
            Err(cause) => panic!("unexpected shared component quote failure: {cause}"),
        }
    }
    Some(capacity)
}
fn accept(
    pool: &MemoryLedger,
    mechanism: &PairMechanism,
    attempts: usize,
    capacity: u64,
) -> Option<Accepted> {
    let before = pool.fixture_host_current().unwrap();
    let quote = match component_quote(pool, mechanism, attempts) {
        Ok(quote) => quote,
        Err(ResidualQuoteError::Storage(WorkingMemoryError::UnknownBound)) => {
            assert_eq!(pool.fixture_host_current().unwrap(), before);
            require_qualified_or_report_unknown("rust-managed");
            return None;
        }
        Err(cause) => panic!("unexpected fixed component quote failure: {cause}"),
    };
    let exact = before
        .checked_add(crate::memory_fixture::host_total(&complete_requirements(
            &quote,
        )))
        .unwrap();
    assert!(matches!(
        plan_prefill_incremental_with_capacity(
            &InferenceExecutionIdentity::default(),
            pool,
            &capabilities(),
            request(),
            geometry(),
            crate::memory_fixture::physical_host_limits(pool, exact - 1),
            |_| Ok(quote.clone())
        ),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    ));
    assert_eq!(pool.fixture_host_current().unwrap(), before);
    let (exact_reservation, _) = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &capabilities(),
        request(),
        geometry(),
        crate::memory_fixture::physical_host_limits(pool, exact),
        |_| Ok(quote.clone()),
    )
    .unwrap();
    assert_eq!(pool.fixture_host_current().unwrap(), exact);
    // No construction or submission has started. Release this exact-capacity
    // admission probe before accepting the shared ceiling used by A and B.
    drop(exact_reservation);
    assert_eq!(pool.fixture_host_current().unwrap(), before);
    assert!(
        capacity >= exact,
        "both quoted components fit the shared ceiling"
    );
    let (reservation, accepted) = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &capabilities(),
        request(),
        geometry(),
        crate::memory_fixture::physical_host_limits(pool, capacity),
        |_| Ok(quote.clone()),
    )
    .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted
        .into_funded_text_span_workspace(&run, &reservation)
        .unwrap();
    let mut bank = span
        .take_native_storage_bank::<PairMechanism>(&run, mechanism.selection())
        .unwrap()
        .unwrap();
    bank.install(mechanism.clone()).unwrap();
    Some(Accepted {
        bank,
        span,
        reservation,
        run,
    })
}
fn construct(component: &Accepted, mechanism: &PairMechanism, values: [u32; 2]) -> Array {
    let plan = OriginalMutablePairPlan::inspect(&mechanism.runtime).unwrap();
    let controls = component.span.control_guard();
    let budget = component
        .bank
        .budget_for_controls(&controls)
        .unwrap()
        .clone();
    plan.prepare(
        values,
        budget,
        OriginalMutablePairCustodies::new(
            controls.clone(),
            controls.clone(),
            controls.clone(),
            controls.clone(),
            controls,
        ),
    )
    .unwrap()
    .try_construct()
    .unwrap()
}
fn baseline(runtime: &PreparedInputRuntime) -> u64 {
    let facts = OriginalMutablePairPlan::inspect(runtime).unwrap().facts();
    // Same exact pinned repr(C, align(2)) RcInner recipe as the qualified neutral
    // owner helper. prepare_qualified is mandatory above; no standalone new
    // compiler qualification or unqualified accepting branch is introduced.
    let runtime_rc = Layout::new::<[usize; 2]>()
        .align_to(2)
        .unwrap()
        .pad_to_align()
        .extend(Layout::new::<PreparedInputRuntime>())
        .unwrap()
        .0
        .pad_to_align()
        .size();
    u64::try_from(
        facts
            .module_bytes()
            .checked_add(facts.thread_bytes())
            .unwrap()
            .checked_add(runtime_rc)
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn mutable_component_accepts_exact_controls_and_preserves_a_origin_under_b_without_double_charge() {
    // Cold allocator/driver initialization is a selected-context prerequisite,
    // outside this explicit component certificate. Fixed submission module/TLS
    // and the actual retained runtime Rc are the component's existing baseline.
    let Some(runtime) = component_runtime() else {
        return;
    };
    let initial = baseline(&runtime);
    let pool = crate::memory_fixture::ledger(u64::MAX, initial).unwrap();
    let a_mechanism = PairMechanism::new(&runtime);
    let Some(capacity) = shared_capacity(&pool, &a_mechanism, [1, 1]) else {
        return;
    };
    let Some(mut a) = accept(&pool, &a_mechanism, 1, capacity) else {
        return;
    };
    let root = construct(&a, &a_mechanism, [101, 0x8765_4321]);
    assert_eq!(
        root.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
        &[101, 0x8765_4321]
    );
    let p = OriginalMutablePairPlan::inspect(&runtime)
        .unwrap()
        .facts()
        .backing_bytes() as u64;
    let mut a_scope = a.run.scope().unwrap();
    let before = pool.fixture_host_charge().unwrap();
    let mut a_publication = a.bank.claim_publication(&mut a_scope).unwrap();
    a_publication.publish(&a_scope, [&root], &[]).unwrap();
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        before,
        "original birth already belongs to prepaid P"
    );
    a_scope.certify().unwrap();
    drop(a_publication);
    let a_host = a.span.protected_host_bytes();
    // The external selection Arc is priced in A's Q, not in the baseline.
    // Bank/publication/registration custody retains the required shared origin.
    drop((a_mechanism, a));
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.fixture_host_charge().unwrap(), initial + a_host + p);

    let b_mechanism = PairMechanism::new(&runtime);
    let mut b = accept(&pool, &b_mechanism, 1, capacity).expect("same qualified component");
    let mut b_scope = b.run.scope().unwrap();
    let before = pool.fixture_host_charge().unwrap();
    let mut b_publication = b.bank.claim_publication(&mut b_scope).unwrap();
    b_publication.publish(&b_scope, [&root], &[]).unwrap();
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        before,
        "B validates its scope but cannot charge A's birth again"
    );
    b_scope.certify().unwrap();
    let b_host = b.span.protected_host_bytes();
    drop((b_mechanism, b_publication, b));
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        initial + a_host + b_host + p,
        "B metadata sidecar survives, but only A's physical partition remains"
    );
    drop(root);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.fixture_host_charge().unwrap(), initial);
}

#[test]
fn mutable_component_foreign_unpublished_birth_refuses_and_retains_actual_preparation() {
    let Some(runtime) = component_runtime() else {
        return;
    };
    let initial = baseline(&runtime);
    let pool = crate::memory_fixture::ledger(u64::MAX, initial).unwrap();
    let a_mechanism = PairMechanism::new(&runtime);
    let Some(capacity) = shared_capacity(&pool, &a_mechanism, [0, 1]) else {
        return;
    };
    let Some(a) = accept(&pool, &a_mechanism, 0, capacity) else {
        return;
    };
    let root = construct(&a, &a_mechanism, [103, 107]);
    let b_mechanism = PairMechanism::new(&runtime);
    let mut b = accept(&pool, &b_mechanism, 1, capacity).expect("same qualified component");
    let mut b_scope = b.run.scope().unwrap();
    let before = pool.fixture_host_charge().unwrap();
    let mut refused = b.bank.claim_publication(&mut b_scope).unwrap();
    assert!(matches!(
        refused.publish(&b_scope, [&root], &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), before);
    b_scope.certify().unwrap();
    drop((a_mechanism, b_mechanism, a, b, root));
    safemlx::reclaim_allocation_owners();
    assert!(
        pool.fixture_host_charge().unwrap() > initial,
        "failed prepared registration keeps actual B metadata"
    );
    drop(refused);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.fixture_host_charge().unwrap(), initial);
}

#[test]
fn completed_source_receipt_retires_repeated_request_controls_but_preserves_new_output_custody() {
    let Some(runtime) = component_runtime() else {
        return;
    };
    let initial = baseline(&runtime);
    let pool = crate::memory_fixture::ledger(u64::MAX, initial).unwrap();
    let owner = crate::backend::managed_memory::NativeMemoryOwner::acquire(&pool).unwrap();
    let source = Array::from_slice(&[37u32, 41], &[2]);
    let source_facts = source.allocation_info().unwrap().unwrap();
    let source_bytes = (source_facts.bytes() + source_facts.host_control_bytes()) as u64;
    let mut source_inventory =
        crate::backend::runtime::residency::storage::RetainedStorage::default();
    source_inventory.include_array(&source).unwrap();
    let receipt = source_inventory.publish_unquoted(&owner).unwrap();
    drop(owner);
    let baseline = initial + source_bytes;
    assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    let mechanism = || {
        let mut value = PairMechanism::new(&runtime);
        value.initial_publication = Some(receipt.clone());
        value
    };
    let Some(capacity) = shared_capacity(&pool, &mechanism(), [1, 1]) else {
        return;
    };
    for _ in 0..3 {
        let mechanism = mechanism();
        let mut accepted = accept(&pool, &mechanism, 1, capacity).unwrap();
        let mut scope = accepted.run.scope().unwrap();
        let mut publication = accepted.bank.claim_publication(&mut scope).unwrap();
        publication.publish(&scope, [&source], &[]).unwrap();
        scope.certify().unwrap();
        drop((publication, accepted, mechanism));
        safemlx::reclaim_allocation_owners();
        crate::backend::ordinary_retirement::reclaim_all();
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            baseline,
            "completed source already owns its attachment; no new request Q may remain on it"
        );
    }
    // A newly born output has no initial receipt. Its actual original P and Q
    // remain with an escaped alias after its publishing request is dropped.
    let mechanism = mechanism();
    let mut accepted = accept(&pool, &mechanism, 1, capacity).unwrap();
    let output = construct(&accepted, &mechanism, [43, 47]);
    let alias = output.clone();
    let mut scope = accepted.run.scope().unwrap();
    let mut publication = accepted.bank.claim_publication(&mut scope).unwrap();
    publication.publish(&scope, [&output], &[]).unwrap();
    scope.certify().unwrap();
    let output_host = accepted.span.protected_host_bytes();
    let output_bytes = output.allocation_info().unwrap().unwrap().bytes() as u64;
    drop((publication, accepted, output, mechanism));
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        baseline + output_host + output_bytes
    );
    assert_eq!(alias.evaluated().unwrap().as_slice::<u32>(), &[43, 47]);
    drop(alias);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    drop(source);
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        initial,
        "retained initial receipt must not pin the original physical source after cache eviction"
    );
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
