use super::*;
#[path = "prepared_control/prediction_loan.rs"]
mod prediction_loan;
use eredu_runtime::{
    working_memory::{
        InferenceExecutionIdentity, InferenceRequest, InferenceStateRetention, MemoryLedger,
    },
    HostSlotAttachmentError, HostSlotMetadata, SharedPreparedInputCacheIdentity,
};

type State = DeviceState<FakeBackend, FakeLayerState>;
type Session =
    eredu_runtime::ReplicatedTextSession<OrdinaryTextFixture, FakeBackend, ReferenceTextMechanisms>;

fn session() -> (Session, ReplicatedSessionCounters) {
    session_with_checkpoint_failure(true)
}

fn session_with_checkpoint_failure(fail_checkpoint: bool) -> (Session, ReplicatedSessionCounters) {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let expected = selected.requirements().architecture_identity().to_owned();
    let contract = prepare_replicated_text_contract::<_, FakeBackend, State>(
        &architecture,
        None,
        selected,
        &expected,
        &(),
    )
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Default::default(),
        completions: Default::default(),
        counters: counters.clone(),
        fail_completion: Rc::new(Cell::new(false)),
        // Any accidental use of the old independent-copy callback must fail.
        fail_checkpoint,
        fail_construction_report: false,
        prepared_partition: None,
        prompt_cache: None,
    };
    let session = construct_replicated_text_session::<_, FakeBackend, _>(
        architecture,
        None,
        contract,
        mechanisms,
        &(),
    )
    .unwrap();
    (session, counters)
}

fn state(position: i32, layers: usize, head_dim: i32) -> State {
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, head_dim).unwrap();
    let layout =
        StateLayout::new(LayerSchedule::new(layers, vec![policy; layers]).unwrap()).unwrap();
    State::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState::new(position))
    })
    .unwrap()
}

fn retain_unquoted(state: &mut State, pool: &MemoryLedger) {
    let owner = pool.acquire_unquoted().unwrap();
    state.inference_retention_mut().retain_unquoted(&owner);
}

fn request(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    position: u64,
) -> InferenceRequest {
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: position,
        input_positions: 1,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::LastPosition,
    };
    pool.reserve(execution, &mock_inference_admission(geometry))
        .unwrap()
        .into()
}

fn assert_retired(token: &HostSlotMetadata) {
    assert!(matches!(
        token.try_attach(&eredu_core::SharedStorageAccountingId::default(), || {
            panic!("retired table cannot acquire new custody");
            #[allow(unreachable_code)]
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
        }),
        Err(HostSlotAttachmentError::Retired)
    ));
}

fn prompt_identity() -> SharedPreparedInputCacheIdentity {
    let description =
        eredu_core::PreparedInputIdentity::new(vec![eredu_core::InputPartDescriptor::new(
            InputModality::Text,
            eredu_core::InputPayloadKind::TokenIds,
            InputTensorIdentity::new(TensorDtype::U32, vec![1, 3]).unwrap(),
            [],
        )
        .unwrap()])
        .unwrap();
    SharedPreparedInputCacheIdentity::new(
        eredu_runtime::PreparedInputCacheIdentity::new(description, "prepared prefix").unwrap(),
    )
}

#[test]
fn prepared_binding_moves_exact_table_and_prompt_without_old_retention_or_copy() {
    let (mut session, counters) = session();
    assert!(session.capture_control_state(&()).is_err());
    let old_pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let new_pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let old_request = request(&old_pool, &execution, 3);
    let new_request = request(&new_pool, &execution, 7);
    let mut installed = state(3, 1, 1);
    installed.inference_retention_mut().admit(&old_request);
    drop(old_request);
    session
        .exchange_prediction_target_state(&mut installed, &())
        .unwrap();
    drop(installed);
    let old_revision = session
        .inspect_runtime_state(|state| Ok(state.inference_retention().revision().clone()))
        .unwrap();
    let origin = session.control_state_origin().unwrap();
    session.validate_control_state_origin(&origin).unwrap();
    let mut prepared = state(7, 1, 1);
    prepared.inference_retention_mut().admit(&new_request);
    let table = prepared.layer_slot_metadata().unwrap().clone();
    let pointer = prepared.as_ref().as_ptr();
    let prepared_revision = prepared.inference_retention().revision().clone();
    let prompt = prompt_identity();
    let prompt_pointer = prompt.as_ref() as *const _;
    let before = counters.snapshot();
    let mut slot = session
        .bind_prepared_control_state(&origin, prepared, Some(prompt.clone()))
        .unwrap();
    assert_eq!(
        slot.shared_prompt_input_identity().unwrap().as_ref() as *const _,
        prompt_pointer
    );
    assert!(slot
        .shared_prompt_input_identity()
        .unwrap()
        .same_storage(&prompt));
    assert_eq!(session.report().unwrap().state_report(), &[3]);
    assert_eq!(
        counters.snapshot().state_allocations,
        before.state_allocations
    );
    assert_eq!(counters.snapshot().forward_calls, before.forward_calls);
    assert_eq!(
        counters.snapshot().completion_attempts,
        before.completion_attempts
    );
    session
        .inspect_runtime_state(|state| {
            assert_eq!(state.inference_retention().revision(), &old_revision);
            Ok(())
        })
        .unwrap();

    session.exchange_control_state(&mut slot, &()).unwrap();
    session
        .inspect_runtime_state(|state| {
            assert_eq!(state.as_ref().as_ptr(), pointer);
            assert!(state.layer_slot_metadata().unwrap().same_storage(&table));
            assert_ne!(state.inference_retention().revision(), &prepared_revision);
            assert_eq!(state.inference_retention().requests().len(), 1);
            let admission = state.inference_retention().admission().unwrap();
            admission
                .request()
                .validate_same_request(&new_request)
                .unwrap();
            assert_eq!(admission.position(), 7);
            Ok(())
        })
        .unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[7]);
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&prompt));
    drop(new_request);
    assert_eq!(
        old_pool.live_charge_bytes().unwrap(),
        mock_reservation_bytes()
    );
    assert_eq!(
        new_pool.live_charge_bytes().unwrap(),
        mock_reservation_bytes()
    );
    // The displaced state alone owns the old request; binding did not inherit it.
    drop(slot);
    assert_eq!(old_pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(
        new_pool.live_charge_bytes().unwrap(),
        mock_reservation_bytes()
    );
    session.validate_control_state_origin(&origin).unwrap();
    drop(session);
    assert_eq!(new_pool.payload_used_bytes().unwrap(), 0);
    assert_retired(&table);
}

#[test]
fn foreign_and_invalidated_origins_reject_cold_then_drop_consumed_state() {
    let (mut target, counters) = session();
    let (foreign, _) = session();
    let foreign_origin = foreign.control_state_origin().unwrap();
    let stale_origin = target.control_state_origin().unwrap();
    target.validate_control_state_origin(&stale_origin).unwrap();
    let publication_pool = crate::memory::host_ledger(u64::MAX, 0).unwrap();
    super::original_resident_reset::publication::publish_generation(&mut target, &publication_pool);
    let pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let before = counters.snapshot();
    for origin in [&foreign_origin, &stale_origin] {
        let error = target.validate_control_state_origin(origin).unwrap_err();
        assert!(
            matches!(error, eredu_runtime::ReplicatedTextSessionError::Contract(ref message)
            if message == "control state belongs to a different executable")
        );
        let mut prepared = state(11, 1, 1);
        retain_unquoted(&mut prepared, &pool);
        let token = prepared.layer_slot_metadata().unwrap().clone();
        let bytes = token.capacity_bytes().unwrap();
        token
            .try_attach(&eredu_core::SharedStorageAccountingId::default(), || {
                pool.register_host_storage([(token.identity().clone(), bytes)])
                    .map(|charge| Box::new(charge) as Box<dyn Send + Sync>)
            })
            .unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        assert!(target
            .bind_prepared_control_state(origin, prepared, Some(prompt_identity()))
            .is_err());
        assert_retired(&token);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        // Escaped metadata retains conservative charge but no retired payload.
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
        drop(token);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    assert_eq!(target.report().unwrap().state_report(), &[0]);
    assert_eq!(
        counters.snapshot().state_allocations,
        before.state_allocations
    );
    assert_eq!(counters.snapshot().forward_calls, before.forward_calls);
    assert_eq!(
        counters.snapshot().completion_attempts,
        before.completion_attempts
    );
    let current = target.control_state_origin().unwrap();
    let accepted = target
        .bind_prepared_control_state(&current, state(9, 1, 1), None)
        .unwrap();
    target.validate_control_state(&accepted).unwrap();
}

#[test]
fn prepared_binding_rejects_wrong_layer_geometry_and_absent_state_without_panicking() {
    let (session, counters) = session();
    let origin = session.control_state_origin().unwrap();
    let pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let before = counters.snapshot();
    for mut prepared in [state(4, 2, 1), state(4, 1, 2), State::stateless()] {
        retain_unquoted(&mut prepared, &pool);
        let token = prepared.layer_slot_metadata().cloned();
        let error = session
            .bind_prepared_control_state(&origin, prepared, None)
            .err()
            .unwrap();
        assert!(
            matches!(error, eredu_runtime::ReplicatedTextSessionError::Contract(ref message)
            if message == "realized state layout differs from selection")
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        if let Some(token) = token {
            assert_retired(&token);
        }
    }
    assert_eq!(session.report().unwrap().state_report(), &[0]);
    assert_eq!(
        counters.snapshot().state_allocations,
        before.state_allocations
    );
    assert_eq!(
        counters.snapshot().completion_attempts,
        before.completion_attempts
    );
}

#[test]
fn origin_clones_do_not_retain_installed_payload_or_accounting_custody() {
    let (mut session, _) = session();
    let pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let mut installed = state(5, 1, 1);
    retain_unquoted(&mut installed, &pool);
    let table = installed.layer_slot_metadata().unwrap().clone();
    session
        .exchange_prediction_target_state(&mut installed, &())
        .unwrap();
    drop(installed);
    let origin = session.control_state_origin().unwrap();
    let alias = origin.clone();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(session);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_retired(&table);
    let (other, _) = self::session();
    assert!(other.validate_control_state_origin(&origin).is_err());
    assert!(other.validate_control_state_origin(&alias).is_err());
}

#[test]
fn prepared_partitioned_binding_defers_agreement_and_revision_to_existing_exchange() {
    let (mut session, _, _, calls, counters) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PromptCacheLoadPreflight,
        None,
        true,
    );
    session.decode(&FakeTensor(vec![3]), &()).unwrap();
    let origin = session.control_state_origin().unwrap();
    let before_calls = calls.borrow().len();
    let before = counters.snapshot();
    let before_epoch = session
        .report()
        .unwrap()
        .distributed_commit()
        .unwrap()
        .epoch()
        .value();
    session.validate_control_state_origin(&origin).unwrap();
    let mut slot = session
        .bind_prepared_control_state(&origin, state(8, 1, 1), None)
        .unwrap();
    assert_eq!(calls.borrow().len(), before_calls);
    assert_eq!(
        counters.snapshot().state_allocations,
        before.state_allocations
    );
    assert_eq!(
        counters.snapshot().completion_attempts,
        before.completion_attempts
    );
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    session.exchange_control_state(&mut slot, &()).unwrap();
    assert_eq!(
        &calls.borrow()[before_calls..],
        &[(DistributedExecutionPhase::ControlExchangePreparation, true)]
    );
    assert_eq!(session.report().unwrap().state_report(), &[8]);
    assert_eq!(
        session
            .report()
            .unwrap()
            .distributed_commit()
            .unwrap()
            .epoch()
            .value(),
        before_epoch
    );
    session.decode(&FakeTensor(vec![4]), &()).unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[9]);
    assert_eq!(
        session
            .report()
            .unwrap()
            .distributed_commit()
            .unwrap()
            .epoch()
            .value(),
        before_epoch + 1
    );
    session.exchange_control_state(&mut slot, &()).unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[1]);
}

#[test]
fn failed_control_agreement_fences_origin_validation_and_consumes_rejected_candidate() {
    let (mut session, _, _, calls, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::ControlCapturePreparation,
        None,
        true,
    );
    let origin = session.control_state_origin().unwrap();
    assert!(session.capture_control_state(&()).is_err());
    let before_calls = calls.borrow().len();
    assert!(session.control_state_origin().is_err());
    assert!(session.validate_control_state_origin(&origin).is_err());
    let pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let mut prepared = state(6, 1, 1);
    retain_unquoted(&mut prepared, &pool);
    let token = prepared.layer_slot_metadata().unwrap().clone();
    assert!(session
        .bind_prepared_control_state(&origin, prepared, None)
        .is_err());
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_retired(&token);
    assert_eq!(calls.borrow().len(), before_calls);
    assert_eq!(session.report().unwrap().state_report(), &[0]);
}
