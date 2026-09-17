//! Actual source birth followed by the generic existing-immutable observation
//! and canonical publication worker. B emits no operation/new backing; its zero
//! native allowance cannot originate A's prepaid-host source row.
use super::*;
use crate::backend::runtime::residency::storage::{native_storage as mlx, StorageIdentity};
use eredu_nn::workspace::WorkspaceExistingStorage;
use eredu_runtime::working_memory::*;

#[derive(Clone, Debug)]
struct ExistingImmutable(mlx::MlxNativeStorage);
impl OriginalNativeStorageMechanism for ExistingImmutable {
    type Key = StorageIdentity;
    type Budget = safemlx::OriginalBufferBudget;
    type Root = Array;
    type Attachment = safemlx::PreparedAllocationOwner<NativeStorageRegistration<StorageIdentity>>;
    type Error = mlx::NativeStorageCause;
    type Observation<'a> = mlx::Observation<'a>;
    fn selection(&self) -> &NativeStorageSelection {
        self.0.selection()
    }
    fn key_clone_storage_bytes(&self) -> Option<u64> {
        // The closed observation below accepts only an actual immutable Native
        // generation. No checkpoint/source/String variant can be emitted.
        Some(0)
    }
    fn create_budget(
        &self,
        custody: OriginalNativeBudgetCustody,
    ) -> Result<Self::Budget, Self::Error> {
        self.0.create_budget(custody)
    }
    fn observe<'a>(
        &'a self,
        budget: &'a Self::Budget,
        root: &'a Array,
    ) -> Result<Self::Observation<'a>, Self::Error> {
        let actual = self.0.observe(budget, root)?;
        if matches!(
            &actual,
            mlx::Observation::Immutable(_) | mlx::Observation::Empty
        ) {
            Ok(actual)
        } else {
            Err(mlx::NativeStorageCause::Fixed(
                safemlx::OriginalBufferCause::UncertifiedBacking,
            ))
        }
    }
    fn describe(actual: &Self::Observation<'_>) -> NativeStorageObservation<StorageIdentity> {
        mlx::MlxNativeStorage::describe(actual)
    }
    fn prepare_attachment(
        &self,
        owner: NativeStorageRegistration<StorageIdentity>,
    ) -> Result<Self::Attachment, Self::Error> {
        self.0.prepare_attachment(owner)
    }
    fn registration(owner: &Self::Attachment) -> &NativeStorageRegistration<StorageIdentity> {
        mlx::MlxNativeStorage::registration(owner)
    }
    fn attach(
        actual: Self::Observation<'_>,
        owner: Self::Attachment,
    ) -> Result<(), (Self::Error, Self::Attachment)> {
        mlx::MlxNativeStorage::attach(actual, owner)
    }
}
struct LaterAlias {
    mechanism: ExistingImmutable,
    quote: IncrementalInferenceQuote,
    geometry: InferenceGeometry,
    request: AdmissionRequest,
    capabilities: ModelCapabilities,
}
impl LaterAlias {
    fn prepare(pool: &WorkingMemoryPool, runtime: Rc<safemlx::PreparedInputRuntime>) -> Self {
        let selection = NativeStorageSelection::default();
        let mechanism = ExistingImmutable(mlx::MlxNativeStorage::new(&Ok(runtime), &selection));
        let geometry = InferenceGeometry {
            batch_size: 1,
            input_positions: 1,
            cached_positions: 0,
            max_output_tokens: 1,
            prefill_chunk_positions: 1,
            output: OutputDemand::StateOnly,
        };
        let context = WorkspaceContext::new(EmptyWorkspace);
        let storage = RegisteredWorkspaceStorage::bind(
            pool,
            &context,
            std::iter::empty::<(StorageIdentity, WorkspaceExistingStorage)>(),
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
        let state_layout = StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let state = eredu_core::estimate_runtime_state(
            &state_layout,
            request.input,
            1,
            1,
            NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        let empty = || {
            WorkspaceBound::bounded(
                0,
                "existing immutable alias publication emits no native operation/backing",
            )
        };
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
        assert_eq!(
            quote.incremental_bytes(),
            0,
            "A's source is already protected; B cannot debit it again"
        );
        let provider = mechanism
            .0
            .publication_owner_layout(0)
            .unwrap()
            .unwrap()
            .control_bytes(1, 2)
            .unwrap()
            + size_of::<ExistingImmutable>() as u64;
        let plan = PreparedNativeStoragePlan::<ExistingImmutable>::prepare_qualified(
            quote.span_workspace(),
            &mechanism,
            Some(0),
            Some((1, 2)),
            (0..quote.span_workspace().plan().records().len()).map(|_| Some(0)),
            Some(provider),
        )
        .unwrap();
        let text = PreparedTextControlWorkspace::prepare_controls(
            geometry,
            quote.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap()
        .with_native_storage(plan)
        .unwrap();
        let quote = quote.with_span_workspace_and_text_controls(text).unwrap();
        let capabilities = ModelCapabilities {
            effective_model_type: "one existing immutable alias component".into(),
            native_max_context: Observed::exact(2, "empty component"),
            effective_max_context: Observed::exact(2, "empty component"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        drop((storage, context));
        Self {
            mechanism,
            quote,
            geometry,
            request,
            capabilities,
        }
    }
    fn publish(self, pool: &WorkingMemoryPool, array: &Array) -> u64 {
        let Self {
            mechanism,
            quote,
            geometry,
            request,
            capabilities,
        } = self;
        let before = pool.used_bytes().unwrap();
        let exact = before + quote.incremental_bytes();
        assert!(matches!(
            plan_prefill_incremental_with_capacity(
                &InferenceExecutionIdentity::default(),
                pool,
                &capabilities,
                request,
                geometry,
                exact - 1,
                |_| Ok(quote.clone())
            ),
            Err(PrefillPlanningError::Reservation(
                WorkingMemoryError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(pool.used_bytes().unwrap(), before);
        let (_, reservation, accepted) = plan_prefill_incremental_with_capacity(
            &InferenceExecutionIdentity::default(),
            pool,
            &capabilities,
            request,
            geometry,
            exact,
            |_| Ok(quote.clone()),
        )
        .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let (mut span, _) = accepted
            .into_funded_text_span_workspace(&run, &reservation)
            .unwrap();
        let host = span.protected_host_bytes();
        let mut bank = span
            .take_native_storage_bank::<ExistingImmutable>(&run, mechanism.selection())
            .unwrap()
            .unwrap();
        bank.install(mechanism.clone()).unwrap();
        let mut scope = run.scope().unwrap();
        let mut publication = bank.claim_publication(&mut scope).unwrap();
        let before = pool.used_bytes().unwrap();
        publication.publish(&scope, [array, array], &[]).unwrap();
        assert_eq!(
            pool.used_bytes().unwrap(),
            before,
            "duplicate actual roots preserve A's sole backing charge"
        );
        scope.certify().unwrap();
        drop((mechanism, publication, bank, span, reservation, run, quote));
        safemlx::reclaim_allocation_owners();
        host
    }
}

#[test]
fn immutable_source_birth_survives_cache_retirement_and_generic_b_alias_publication() {
    let fixture = OriginalGgufMissFixture::new();
    let layouts = fixture.source_storage_layouts();
    let runtime = fixture.host_runtime.clone();
    let mut later = None;
    let (array, pool) = component_destinations_with_ceiling(
        layouts.iter().sum::<u64>() * 3,
        layouts.len() * 3,
        3,
        None,
        |pool| {
            let prepared = LaterAlias::prepare(pool, runtime.clone());
            let bytes = prepared.quote.incremental_bytes();
            later = Some(prepared);
            bytes
        },
        |controls, observer, pool, mut host| {
            let bank = host.take_source_constructions().unwrap();
            (
                fixture.funded_source_miss_hit_miss(controls, observer, bank),
                pool.clone(),
            )
        },
    );
    drop(fixture);
    let original = pool.used_bytes().unwrap();
    assert!(original > 0);
    let facts = match array.inspect_immutable_source().unwrap() {
        safemlx::ImmutableSourceInspection::Allocation(witness) => witness.allocation(),
        _ => panic!("actual source output has immutable birth"),
    };
    let pin = pool
        .pin_registered_storage([(
            StorageIdentity::Native(facts.identity()),
            facts.bytes() as u64,
        )])
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), original);
    drop(pin);
    drop(runtime);
    let b_host = later.take().unwrap().publish(&pool, &array);
    assert_eq!(
        pool.used_bytes().unwrap(),
        original + b_host,
        "only B's sidecar controls are additionally protected"
    );
    std::thread::spawn(move || drop(array)).join().unwrap();
    settle_pool(&pool);
}
