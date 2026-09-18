//! Actual source birth followed by the generic existing-immutable observation
//! and canonical publication worker. B emits no operation/new backing; its zero
//! native allowance cannot originate A's prepaid-host source row.
use super::*;
use crate::backend::runtime::residency::storage::{StorageIdentity, native_storage as mlx};
use eredu_nn::workspace::WorkspaceExistingStorage;
use eredu_runtime::working_memory::*;

#[derive(Clone, Debug)]
struct ExistingImmutable(mlx::MlxNativeStorage);
impl OriginalNativeStorageMechanism for ExistingImmutable {
    type Key = StorageIdentity;
    type Budget = safemlx::OriginalBufferBudget;
    type Root<'a> = mlx::NativeStorageRoot<'a>;
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
    fn observe<'a, 'root: 'a>(
        &'a self,
        budget: &'a Self::Budget,
        root: mlx::NativeStorageRoot<'root>,
    ) -> Result<Self::Observation<'a>, Self::Error> {
        let actual = self.0.observe(budget, root)?;
        if matches!(
            &actual,
            mlx::Observation::Immutable(_)
                | mlx::Observation::Host(_)
                | mlx::Observation::HostBuffer(_, _)
                | mlx::Observation::Empty
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
    fn has_retained_attachment(
        &self,
        previous: &Self::Observation<'_>,
        current: &Self::Observation<'_>,
        pool: &WorkingMemoryPool,
    ) -> bool {
        self.0.has_retained_attachment(previous, current, pool)
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
    fn publish(self, pool: &WorkingMemoryPool, root: mlx::NativeStorageRoot<'_>) -> u64 {
        self.with_attempt(pool, None, |mut publication, scope| {
            let before = pool.used_bytes().unwrap();
            publication.publish(scope, [root, root], &[]).unwrap();
            assert_eq!(
                pool.used_bytes().unwrap(),
                before,
                "duplicate actual roots preserve A's sole backing charge"
            );
        })
        .1
    }
    fn with_attempt<T>(
        self,
        pool: &WorkingMemoryPool,
        ceiling: Option<u64>,
        operation: impl FnOnce(
            OriginalNativePublication<ExistingImmutable>,
            &WorkingMemoryFundingScope,
        ) -> T,
    ) -> (T, u64) {
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
        let (reservation, accepted) = plan_prefill_incremental_with_capacity(
            &InferenceExecutionIdentity::default(),
            pool,
            &capabilities,
            request,
            geometry,
            ceiling.unwrap_or(exact),
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
        let publication = bank.claim_publication(&mut scope).unwrap();
        let value = operation(publication, &scope);
        assert!(
            bank.claim_publication(&mut scope).is_err(),
            "one admitted attempt is never refunded"
        );
        scope.certify().unwrap();
        drop((mechanism, bank, span, reservation, run, quote));
        safemlx::reclaim_allocation_owners();
        (value, host)
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
    let b_host = later
        .take()
        .unwrap()
        .publish(&pool, mlx::NativeStorageRoot::Array(&array));
    assert_eq!(
        pool.used_bytes().unwrap(),
        original + b_host,
        "only B's sidecar controls are additionally protected"
    );
    std::thread::spawn(move || drop(array)).join().unwrap();
    settle_pool(&pool);
}

fn prepared_host_source_with<T>(
    runtime: &Rc<safemlx::PreparedInputRuntime>,
    caller_controls: u64,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
    retain: impl FnOnce(
        crate::backend::runtime::residency::storage::filled_host::PublishedHostSource,
    ) -> T,
) -> (T, WorkingMemoryPool) {
    use crate::backend::runtime::residency::storage::filled_host;
    let shape = [4];
    let plan = safemlx::PreparedHostTransferPlan::new(&runtime, &shape, safemlx::Dtype::Float32, 0)
        .unwrap();
    let source_bytes = filled_host::control_bytes::<filled_host::Cumulative>(&plan)
        .unwrap()
        .checked_add(plan.backing_bytes())
        .unwrap() as u64;
    component_destinations_with_ceiling(
        source_bytes.checked_add(caller_controls).unwrap(),
        1 + usize::from(caller_controls != 0),
        0,
        None,
        additional_ceiling,
        |controls, _, pool, mut host| {
            let mut bank = host.take_source_constructions().unwrap();
            let custody = controls.clone().into();
            let mut pending =
                filled_host::begin(&mut bank, plan, filled_host::Cumulative, &custody).unwrap();
            for (slot, value) in pending
                .completed_mut()
                .bytes_mut()
                .chunks_exact_mut(4)
                .zip([1.5f32, -2.0, 7.25, 19.0])
            {
                slot.copy_from_slice(&value.to_ne_bytes());
            }
            pending.completed_mut().freeze().unwrap();
            let source = filled_host::finish(pending).unwrap();
            let caller_receipt =
                (caller_controls != 0).then(|| bank.try_debit(caller_controls).unwrap());
            assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
            let retained = retain(source);
            drop(caller_receipt);
            (retained, pool.clone())
        },
    )
}

fn prepared_host_source(
    runtime: &Rc<safemlx::PreparedInputRuntime>,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
) -> (safemlx::ImmutableHostTransferBuffer, WorkingMemoryPool) {
    prepared_host_source_with(runtime, 0, additional_ceiling, |source| {
        let (buffer, facts, custody) = source.into_parts();
        assert_eq!(buffer.try_allocation_info().unwrap(), facts.expect("nonempty source attachment").allocation());
        drop(custody);
        buffer
    })
}

fn prepared_retained_host_source(
    runtime: &Rc<safemlx::PreparedInputRuntime>,
    additional_ceiling: impl FnOnce(&WorkingMemoryPool) -> u64,
) -> (
    crate::backend::runtime::residency::manager::RetainedHostBuffer,
    WorkingMemoryPool,
) {
    use crate::backend::runtime::residency::manager::RetainedHostBuffer;
    let controls = RetainedHostBuffer::storage_bytes().unwrap()
        + std::mem::size_of::<RetainedHostBuffer>() as u64;
    prepared_host_source_with(
        runtime,
        controls,
        additional_ceiling,
        RetainedHostBuffer::request,
    )
}

#[test]
fn immutable_host_source_survives_closed_donor_and_generic_b_alias_publication() {
    let runtime = Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap());
    let mut later = None;
    let (source, pool) = prepared_host_source(&runtime, |pool| {
        let prepared = LaterAlias::prepare(pool, runtime.clone());
        let bytes = prepared.quote.incremental_bytes();
        later = Some(prepared);
        bytes
    });
    // A's complete run, span, bank, and quote have retired. Only the actual
    // immutable source and its accounting attachments keep A's original hold.
    let original = pool.used_bytes().unwrap();
    assert!(original > 0);
    let observed = source.inspect_original_source().unwrap();
    assert!(observed.is_prepared_source());
    let physical = observed.allocation();
    drop(observed);
    let pin = pool
        .pin_registered_storage([(
            StorageIdentity::Native(physical.identity()),
            physical.bytes() as u64,
        )])
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), original);
    drop(pin);
    drop(runtime);
    let b_host = later
        .take()
        .unwrap()
        .publish(&pool, mlx::NativeStorageRoot::Host(&source, None));
    assert_eq!(pool.used_bytes().unwrap(), original + b_host);
    let values = source
        .as_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, [1.5f32, -2.0, 7.25, 19.0]);
    drop(source);
    settle_pool(&pool);
}

#[test]
fn immutable_host_source_refuses_foreign_pool_without_new_birth_or_attempt_refund() {
    for receipt_supplied in [false, true] {
        let runtime = Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap());
        let (source, donor) = prepared_retained_host_source(&runtime, |_| 0);
        let donor_before = donor.used_bytes().unwrap();
        assert!(
            source
                .inspect_original_source()
                .unwrap()
                .is_prepared_source()
        );
        let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        LaterAlias::prepare(&foreign, runtime).with_attempt(
            &foreign,
            None,
            |mut publication, scope| {
                let before = foreign.used_bytes().unwrap();
                assert!(matches!(
                    publication.publish(
                        scope,
                        [mlx::NativeStorageRoot::Host(
                            &source,
                            if receipt_supplied {
                                source.attachment_receipt()
                            } else {
                                None
                            }
                        )],
                        &[]
                    ),
                    Err(NativeStorageError::Memory(
                        WorkingMemoryError::IdentityMismatch
                    ))
                ));
                assert_eq!(
                    publication.failure_site(),
                    "registry physical alias missing"
                );
                assert_eq!(foreign.used_bytes().unwrap(), before);
                assert_eq!(donor.used_bytes().unwrap(), donor_before);
                assert!(matches!(
                    publication.publish(
                        scope,
                        [mlx::NativeStorageRoot::Host(
                            &source,
                            if receipt_supplied {
                                source.attachment_receipt()
                            } else {
                                None
                            }
                        )],
                        &[]
                    ),
                    Err(NativeStorageError::Memory(
                        WorkingMemoryError::PreparationAlreadyStarted
                    ))
                ));
            },
        );
        settle_pool(&foreign);
        assert_eq!(donor.used_bytes().unwrap(), donor_before);
        assert_eq!(source.as_bytes().unwrap().len(), 16);
        drop(source);
        settle_pool(&donor);
    }
}

#[test]
fn immutable_host_source_refuses_quarantined_donor_without_refunding_origin() {
    for receipt_supplied in [false, true] {
        let runtime = Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap());
        let mut aliases = None;
        let (source, pool) = prepared_retained_host_source(&runtime, |pool| {
            let first = LaterAlias::prepare(pool, runtime.clone());
            let next = LaterAlias::prepare(pool, runtime.clone());
            let bytes = first.quote.incremental_bytes() + next.quote.incremental_bytes();
            aliases = Some((first, next));
            bytes
        });
        let original = pool.used_bytes().unwrap();
        let (first, next) = aliases.unwrap();
        let ceiling = original + first.quote.incremental_bytes() + next.quote.incremental_bytes();
        // Both finite attempts are accepted before the real unwind. Quarantine
        // must reject reuse, rather than merely preventing a new request admission.
        next.with_attempt(&pool, Some(ceiling), |mut next_publication, next_scope| {
            first.with_attempt(
                &pool,
                Some(ceiling),
                |mut first_publication, first_scope| {
                    first_publication
                        .publish(
                            first_scope,
                            [mlx::NativeStorageRoot::Host(&source, None)],
                            &[],
                        )
                        .unwrap();
                    let caught =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                            let _published_alias = first_publication;
                            panic!("exercise actual published-origin unwind custody");
                        }));
                    assert!(caught.is_err());
                },
            );
            assert!(
                source
                    .inspect_original_source()
                    .unwrap()
                    .is_prepared_source()
            );
            assert!(
                !source
                    .attachment_receipt()
                    .unwrap()
                    .matches(source.try_allocation_info().unwrap(), &pool)
            );
            let before = pool.used_bytes().unwrap();
            assert!(matches!(
                next_publication.publish(
                    next_scope,
                    [mlx::NativeStorageRoot::Host(
                        &source,
                        if receipt_supplied {
                            source.attachment_receipt()
                        } else {
                            None
                        }
                    )],
                    &[]
                ),
                Err(NativeStorageError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert_eq!(pool.used_bytes().unwrap(), before);
            assert!(matches!(
                next_publication.publish(
                    next_scope,
                    [mlx::NativeStorageRoot::Host(
                        &source,
                        if receipt_supplied {
                            source.attachment_receipt()
                        } else {
                            None
                        }
                    )],
                    &[]
                ),
                Err(NativeStorageError::Memory(
                    WorkingMemoryError::PreparationAlreadyStarted
                ))
            ));
        });
        assert_eq!(source.as_bytes().unwrap().len(), 16);
        drop(source);
        safemlx::reclaim_allocation_owners();
        assert!(
            pool.used_bytes().unwrap() >= original,
            "quarantine cannot refund the donor's original source account"
        );
    }
}

#[test]
fn prepared_host_constructor_preserves_ordinary_origin_for_host_and_array_aliases() {
    use crate::backend::managed_memory::NativeMemoryOwner;
    use crate::backend::runtime::residency::storage::RetainedStorage;
    let runtime = Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap());
    for array_alias in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        // This is the same actual prepared constructor used by load-time Host
        // managers. Its quota pays native metadata; publication pays backing.
        let plan =
            safemlx::PreparedHostTransferPlan::new(&runtime, &[4], safemlx::Dtype::Float32, 1)
                .unwrap();
        let quota =
            safemlx::PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), ()).unwrap();
        let arena = safemlx::PreparedInputArena::try_allocate(quota).unwrap();
        let mut source = plan.construct(&arena).unwrap();
        for (bytes, value) in source
            .as_bytes_mut()
            .unwrap()
            .chunks_exact_mut(4)
            .zip([1.5f32, -2., 7.25, 19.])
        {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        let source = std::sync::Arc::new(source.freeze());
        assert!(
            source
                .inspect_original_source()
                .unwrap()
                .is_prepared_source()
        );
        let capacity = source.allocation_info().unwrap().bytes() as u64;
        let array = source.try_prepared_source_array().unwrap();
        array.evaluated().unwrap();
        assert_eq!(
            array.allocation_info().unwrap(),
            Some(source.allocation_info().unwrap())
        );
        let mut inventory = RetainedStorage::default();
        inventory.include_host(source.clone()).unwrap();
        let publication = inventory.publish_unquoted(&loading).unwrap();
        drop((publication, loading, arena));
        crate::backend::ordinary_retirement::reclaim_all();
        assert_eq!(pool.used_bytes().unwrap(), capacity);
        let root = if array_alias {
            mlx::NativeStorageRoot::Array(&array)
        } else {
            mlx::NativeStorageRoot::Host(&source, None)
        };
        let extra = LaterAlias::prepare(&pool, runtime.clone()).publish(&pool, root);
        assert_eq!(pool.used_bytes().unwrap(), capacity + extra);
        drop(source);
        assert_eq!(
            array.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
            [1.5, -2., 7.25, 19.]
        );
        safemlx::reclaim_allocation_owners();
        assert_eq!(
            pool.used_bytes().unwrap(),
            capacity + extra,
            "the escaped Array still owns the entire original Host backing and alias controls"
        );
        drop(array);
        settle_pool(&pool);
    }
}

#[test]
fn completed_host_receipt_releases_later_request_while_loaded_source_lives() {
    use crate::backend::managed_memory::NativeMemoryOwner;
    use crate::backend::runtime::residency::storage::RetainedStorage;
    let runtime = Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap());
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let plan =
        safemlx::PreparedHostTransferPlan::new(&runtime, &[4], safemlx::Dtype::Float32, 1).unwrap();
    let quota = safemlx::PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), ()).unwrap();
    let arena = safemlx::PreparedInputArena::try_allocate(quota).unwrap();
    let mut host = plan.construct(&arena).unwrap();
    for (bytes, value) in host
        .as_bytes_mut()
        .unwrap()
        .chunks_exact_mut(4)
        .zip([1.5f32, -2., 7.25, 19.])
    {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
    let host = std::sync::Arc::new(host.freeze());
    let facts = host.allocation_info().unwrap();
    let mut inventory = RetainedStorage::default();
    inventory.include_host(host.clone()).unwrap();
    let receipt = inventory.publish_unquoted(&loading).unwrap();
    drop((loading, arena));
    assert!(receipt.has_native_attachment(pool.shared_storage_domain(), facts));
    assert!(!receipt.has_native_attachment(foreign.shared_storage_domain(), facts));
    let baseline = pool.used_bytes().unwrap();
    assert_eq!(baseline, facts.bytes() as u64);
    // Both native observations must take the same positive complete receipt;
    // neither later Q is allowed to survive on this loaded physical source.
    let array = host.try_prepared_source_array().unwrap();
    array.evaluated().unwrap();
    for root in [
        mlx::NativeStorageRoot::Host(&host, None),
        mlx::NativeStorageRoot::Array(&array),
    ] {
        let mut later = LaterAlias::prepare(&pool, runtime.clone());
        later.mechanism.0 = later.mechanism.0.with_initial_publication(receipt.clone());
        later.publish(&pool, root);
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::reclaim_allocation_owners();
        assert_eq!(
            pool.used_bytes().unwrap(),
            baseline,
            "the loaded Host survives but the later request's complete Q retires"
        );
    }
    assert_eq!(
        array.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        [1.5, -2., 7.25, 19.]
    );
    drop((host, array));
    settle_pool(&pool);
    // The still-live receipt keeps only generation/capacity scalars.
    let next = safemlx::HostTransferBuffer::new(
        &[4],
        safemlx::Dtype::Float32,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap()
    .freeze();
    let next_facts = next.allocation_info().unwrap();
    assert_eq!(next_facts.bytes(), facts.bytes());
    assert_ne!(next_facts.identity(), facts.identity());
    assert!(!receipt.has_native_attachment(pool.shared_storage_domain(), next_facts));
    drop((next, receipt));
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn published_lazy_host_receipt_releases_every_later_request_while_source_survives() {
    use crate::backend::runtime::residency::manager::RetainedHostBuffer;
    let runtime = Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap());
    let mut later = Vec::new();
    let (host, pool) = prepared_retained_host_source(&runtime, |pool| {
        later.extend((0..2).map(|_| LaterAlias::prepare(pool, runtime.clone())));
        later
            .iter()
            .map(|value| value.quote.incremental_bytes())
            .max()
            .unwrap()
    });
    // The source constructor's complete run is closed. Its actual native owner
    // and final manager shell retain exactly A; no new registration pin exists.
    let original = pool.used_bytes().unwrap();
    assert!(original > 0);
    let facts = host.try_allocation_info().unwrap();
    let receipt = host
        .attachment_receipt()
        .expect("successful source attachment");
    assert!(receipt.matches(facts, &pool));
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    assert!(!receipt.matches(facts, &foreign));
    let unrelated = safemlx::HostTransferBuffer::new(
        &[4],
        safemlx::Dtype::Float32,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap()
    .freeze();
    assert!(!receipt.matches(unrelated.try_allocation_info().unwrap(), &pool));
    drop(unrelated);
    for request in later {
        request.publish(
            &pool,
            mlx::NativeStorageRoot::Host(&host, host.attachment_receipt()),
        );
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::reclaim_allocation_owners();
        assert_eq!(
            pool.used_bytes().unwrap(),
            original,
            "persistent lazy backing retains A but never a later request's Q"
        );
    }
    let values = host
        .as_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, [1.5, -2., 7.25, 19.]);
    drop(host);
    settle_pool(&pool);
}
