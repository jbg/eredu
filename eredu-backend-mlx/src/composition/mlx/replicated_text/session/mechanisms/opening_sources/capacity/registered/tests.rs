use super::super::tests as fixture;
use super::*;
use eredu_core::*;
use eredu_runtime::working_memory::*;

pub(in crate::composition::mlx::replicated_text) struct OpeningPinSetup {
    pub(in crate::composition::mlx::replicated_text) snapshot: PreparedOpeningPins,
    pub(in crate::composition::mlx::replicated_text) scope: WorkingMemoryFundingScope,
    pub(in crate::composition::mlx::replicated_text) owned: OwnedTextSpanWorkspace,
    pub(in crate::composition::mlx::replicated_text) reservation: WorkingMemoryReservation,
    pub(in crate::composition::mlx::replicated_text) run: WorkingMemoryFundingRun,
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    pub(in crate::composition::mlx::replicated_text) fn prepare_opening_pins_for_test(
        session: &eredu_runtime::ReplicatedTextSession<A, MlxNeuralBackend, Self>,
        source: &eredu_core::capture::SharedCapturePlan,
        q: IncrementalInferenceQuote,
        pool: &WorkingMemoryPool,
        caps: &ModelCapabilities,
        capacity: Option<u64>,
    ) -> Result<OpeningPinSetup, Error> {
        let selected = session
            .shared_observation_paths()
            .unwrap()
            .prepare_capture_selection(source)
            .map_err(|e| Error::Other(Box::new(e)))?;
        let plan = Self::prepare_session_fixed_opening_capacity(
            session,
            selected
                .bind_geometry(q.geometry())
                .map_err(|e| Error::Other(Box::new(e)))?,
        )
        .map_err(|e| Error::Other(Box::new(e)))?;
        // Initial publication covers model owners. The capture plan receives
        // its typed attachment through this original accepted quote, not the
        // legacy opaque inventory attachment.
        assert!(matches!(
            pool.pin_registered_storage([(
                Key::CapturePlan(source.storage_identity().clone()),
                source.capacity_bytes().unwrap(),
            )]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        let publication = PreparedCapturePlanPublication::prepare(
            pool,
            q.span_workspace().plan(),
            source,
            Key::CapturePlan(source.storage_identity().clone()),
            None,
        )
        .map_err(|e| Error::Other(Box::new(e)))?;
        assert_eq!(
            publication.new_source_bytes(),
            source.capacity_bytes().unwrap()
        );
        let proposal = plan
            .seal(
                q.span_workspace().plan(),
                TextHostControlFacts::new(Some(0), Some(0), Some(0)),
            )
            .and_then(|p| p.with_existing_opening_pins())
            .and_then(|p| p.with_capture_plan_publication(publication))
            .map_err(|e| Error::Other(Box::new(e)))?;
        let (sealed, q) = proposal.finish(q).map_err(|e| Error::Other(Box::new(e)))?;
        let geometry = q.geometry();
        let request = AdmissionRequest {
            input: InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
            max_output_tokens: geometry.max_output_tokens,
            batch_size: geometry.batch_size,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        let before = pool.used_bytes().unwrap();
        let exact = before.checked_add(q.incremental_bytes()).unwrap();
        let short = plan_prefill_incremental_with_capacity(
            session.inference_execution_identity(),
            pool,
            caps,
            request.clone(),
            geometry,
            exact - 1,
            |_| Ok(q.clone()),
        );
        assert!(
            short.is_err(),
            "one byte short must reject before native snapshot allocation"
        );
        assert_eq!(pool.used_bytes().unwrap(), before);
        let (_, reservation, accepted) = plan_prefill_incremental_with_capacity(
            session.inference_execution_identity(),
            pool,
            caps,
            request,
            geometry,
            capacity.unwrap_or(exact),
            |_| Ok(q.clone()),
        )
        .map_err(|e| Error::Other(Box::new(e)))?;
        drop(q);
        let (reservation, run) = reservation
            .into_funding()
            .map_err(|e| Error::Other(Box::new(e)))?;
        let scope = run.scope().map_err(|e| Error::Other(Box::new(e)))?;
        let pending = accepted
            .begin_capture_plan_publication::<Key>(&run, &reservation, source)
            .map_err(|e| Error::Other(Box::new(e)))?;
        let (mut owned, witness) = pending
            .publish_and_finish(&scope)
            .map_err(|e| Error::Other(Box::new(e)))?;
        drop(witness);
        let mut capsule = sealed
            .allocate(&owned)
            .map_err(|e| Error::Other(Box::new(e.cause)))?;
        capsule.collect().map_err(|e| Error::Other(Box::new(e)))?;
        let snapshot = capsule
            .prepare_registered_pins(&mut owned)
            .map_err(|e| Error::Other(Box::new(e.cause)))?;
        assert!(
            owned.take_prefill_storage_pins::<Key>().is_err(),
            "same original bank cannot be minted twice"
        );
        Ok(OpeningPinSetup {
            snapshot,
            scope,
            owned,
            reservation,
            run,
        })
    }
}

#[test]
fn derived_slots_are_checked_and_native_zero_does_not_erase_real_source_origin() {
    let n = OpeningSlots {
        arrays: 2,
        hosts: 3,
        bytes: 5,
        sources: 7,
        layouts: 11,
        tables: 13,
    };
    assert_eq!(descriptor_count(n).unwrap(), 41);
    assert!(matches!(
        descriptor_count(OpeningSlots {
            arrays: usize::MAX,
            hosts: 1,
            ..Default::default()
        }),
        Err(OpeningError::Overflow)
    ));
    let empty = Array::from_slice::<f32>(&[], &[0]);
    let _ = empty.evaluated().unwrap();
    assert!(storage_entry(OpeningEntry::Array(
        &empty,
        empty.try_allocation_info().unwrap().unwrap()
    ))
    .unwrap()
    .is_none());
    let source = Arc::new(Vec::<u8>::new());
    let entry = eredu_checkpoint::store::SourceStorageRef::new(&source, 0);
    let (key, bytes) = storage_entry(OpeningEntry::Source(entry)).unwrap().unwrap();
    assert_eq!(bytes, 0);
    assert_eq!(key, Key::Source(entry.identity()));
    let bytes: Arc<[u8]> = Arc::from([]);
    let first = storage_entry(OpeningEntry::Bytes(&bytes, 0))
        .unwrap()
        .unwrap();
    let alias = bytes.clone();
    assert_eq!(
        first,
        storage_entry(OpeningEntry::Bytes(&alias, 0))
            .unwrap()
            .unwrap()
    );
}

#[test]
fn arbitrary_pin_layout_cannot_replace_derived_native_proposal() {
    let stream = fixture::stream();
    let execution = fixture::execution(&stream);
    let state = fixture::state();
    let source = fixture::source();
    let paths = execution.prepare_observation_paths().unwrap();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let store = eredu_checkpoint::store::MemoryWeightStore::default();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let q = fixture::quote(&pool);
    let plan = NativeOpeningCapacityPlan::prepare(
        &state,
        &execution,
        &store,
        None,
        selected.bind_geometry(fixture::geometry()).unwrap(),
        &paths,
    )
    .unwrap();
    let arbitrary =
        PreparedPrefillStoragePinPlan::<Key>::prepare(q.span_workspace().plan(), |_| Some(999))
            .unwrap();
    let proposal = plan
        .seal(
            q.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap()
        .with_prefill_storage_pins(arbitrary)
        .unwrap();
    let (sealed, q) = proposal.finish(q).unwrap();
    let (run, mut owned) = fixture::accept(&pool, q);
    let mut capsule = sealed
        .allocate(&owned)
        .unwrap_or_else(|e| panic!("{}", e.cause));
    capsule.collect().unwrap();
    let failure = match capsule.prepare_registered_pins(&mut owned) {
        Err(e) => e,
        Ok(_) => panic!("arbitrary bank accepted"),
    };
    assert!(matches!(failure.cause, OpeningError::Identity));
    assert!(
        owned.take_prefill_storage_pins::<Key>().is_ok(),
        "rejection precedes bank extraction"
    );
    assert!(failure.inventory.owners.retained_counts().arrays > 0);
    drop((failure, owned, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn incomplete_handoff_retains_capsule_and_foreign_binding_does_not_take_bank() {
    let stream = fixture::stream();
    let execution = fixture::execution(&stream);
    let state = fixture::state();
    let source = fixture::source();
    let paths = execution.prepare_observation_paths().unwrap();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let store = eredu_checkpoint::store::MemoryWeightStore::default();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let q = fixture::quote(&pool);
    let plan = NativeOpeningCapacityPlan::prepare(
        &state,
        &execution,
        &store,
        None,
        selected.bind_geometry(fixture::geometry()).unwrap(),
        &paths,
    )
    .unwrap();
    let (sealed, q) = plan
        .seal(
            q.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap()
        .with_existing_opening_pins()
        .unwrap()
        .finish(q)
        .unwrap();
    let (run, mut owned) = fixture::accept(&pool, q);
    let capsule = sealed
        .allocate(&owned)
        .unwrap_or_else(|e| panic!("{}", e.cause));
    let failure = match capsule.prepare_registered_pins(&mut owned) {
        Err(e) => e,
        Ok(_) => panic!("incomplete accepted"),
    };
    assert!(matches!(failure.cause, OpeningError::Incomplete));
    let mut capsule = failure.inventory;
    capsule.collect().unwrap();
    let foreign_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let other = fixture::quote(&foreign_pool);
    let other_plan = NativeOpeningCapacityPlan::prepare(
        &state,
        &execution,
        &store,
        None,
        selected.bind_geometry(fixture::geometry()).unwrap(),
        &paths,
    )
    .unwrap();
    let (_, other) = other_plan
        .seal(
            other.span_workspace().plan(),
            TextHostControlFacts::new(Some(0), Some(0), Some(0)),
        )
        .unwrap()
        .with_existing_opening_pins()
        .unwrap()
        .finish(other)
        .unwrap();
    let (other_run, mut other_owned) = fixture::accept(&foreign_pool, other);
    let failure = match capsule.prepare_registered_pins(&mut other_owned) {
        Err(e) => e,
        Ok(_) => panic!("foreign accepted"),
    };
    assert!(matches!(failure.cause, OpeningError::Identity));
    assert!(other_owned.take_prefill_storage_pins::<Key>().is_ok());
    let snapshot = failure
        .inventory
        .prepare_registered_pins(&mut owned)
        .unwrap_or_else(|e| panic!("{}", e.cause));
    assert!(snapshot.paths.same_storage(paths.source()));
    drop((snapshot, owned, run, other_owned, other_run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
}

impl PreparedOpeningPins {
    pub(in crate::composition::mlx::replicated_text) fn captured_keys_for_test(
        &self,
    ) -> Vec<(Key, u64)> {
        let mut keys = Vec::new();
        self.owners
            .visit(&mut |entry| {
                if let Some(pair) = storage_entry(entry)? {
                    keys.push(pair);
                }
                Ok::<_, OpeningError>(())
            })
            .unwrap();
        keys
    }
    pub(in crate::composition::mlx::replicated_text) fn issued_rows_for_test(&self) -> usize {
        self.bank.spent_rows()
    }
}
