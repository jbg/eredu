use super::*;
use eredu_runtime::{
    prefill::{PrefillDriver, PrefillOutcome, PrefillProgress},
    replicated_session::{PreparedPrefillSource, SessionPrefill},
    working_memory::{InferenceRequest, MemoryLedger},
    PreparedInputCacheIdentity, SharedPreparedInputCacheIdentity,
};

type Session =
    eredu_runtime::ReplicatedTextSession<OrdinaryTextFixture, FakeBackend, ReferenceTextMechanisms>;

fn session() -> (Session, Rc<Cell<bool>>) {
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
    let contract = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(&architecture, None, selected, &expected, &())
    .unwrap();
    let fail_completion = Rc::new(Cell::new(false));
    let mechanisms = ReferenceTextMechanisms {
        tasks: Default::default(),
        completions: Default::default(),
        counters,
        fail_completion: fail_completion.clone(),
        fail_checkpoint: false,
        fail_construction_report: false,
        prepared_partition: None,
        prompt_cache: None,
    };
    (
        construct_replicated_text_session::<_, FakeBackend, _>(
            architecture,
            None,
            contract,
            mechanisms,
            &(),
        )
        .unwrap(),
        fail_completion,
    )
}

fn charged_identity(pool: &MemoryLedger, label: &str) -> SharedPreparedInputCacheIdentity {
    let description =
        eredu_core::PreparedInputIdentity::new(vec![eredu_core::InputPartDescriptor::new(
            InputModality::Text,
            eredu_core::InputPayloadKind::TokenIds,
            InputTensorIdentity::new(TensorDtype::U32, vec![1, 2]).unwrap(),
            [],
        )
        .unwrap()])
        .unwrap();
    let mut content = String::with_capacity(127);
    content.push_str(label);
    let owner = SharedPreparedInputCacheIdentity::new(
        PreparedInputCacheIdentity::new(description, content).unwrap(),
    );
    let bytes = owner.capacity_bytes().unwrap();
    let identity = owner.identity().clone();
    owner
        .try_attach(&eredu_core::SharedStorageAccountingId::default(), || {
            pool.register_host_storage([(identity, bytes)])
                .map(|charge| Box::new(charge) as Box<dyn Send + Sync>)
        })
        .unwrap();
    owner
}

#[test]
fn shared_identity_survives_prefill_rollback_checkpoint_and_independent_snapshots() {
    let pool = crate::memory::host_ledger(u64::MAX, 0).unwrap();
    let first = charged_identity(&pool, "first prompt");
    let first_bytes = first.capacity_bytes().unwrap();
    let original_key = first.identity().clone();
    let payload_pointer = first.as_ref() as *const PreparedInputCacheIdentity;
    let (mut session, fail_completion) = session();
    assert_eq!(
        session
            .prefill_input_with_shared_cache_identity(&FakeTensor(vec![3, 7]), first.clone(), &(),)
            .unwrap(),
        FakeTensor(vec![5])
    );
    assert_eq!(
        session.committed_prompt_input_identity().unwrap() as *const _,
        payload_pointer
    );
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    let checkpoint = session.checkpoint_complete(&()).unwrap();
    let saved = session.capture_control_state(&()).unwrap();
    let mut branch = session.copy_control_state(&saved, &()).unwrap();
    assert!(saved
        .shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    assert!(branch
        .shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);

    let rejected = charged_identity(&pool, "failed prompt");
    fail_completion.set(true);
    assert!(session
        .prefill_input_with_observer_and_shared_cache_identity(
            &FakeTensor(vec![11, 13]),
            rejected.clone(),
            &(),
            &mut eredu_runtime::NoopObserver,
        )
        .is_err());
    fail_completion.set(false);
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    drop(rejected);
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);

    let second = charged_identity(&pool, "second prompt");
    let second_bytes = second.capacity_bytes().unwrap();
    session
        .prefill_input_result_with_shared_identity(
            Ok(&FakeTensor(vec![17, 19])),
            Some(second.clone()),
            &(),
            &mut eredu_runtime::NoopObserver,
        )
        .unwrap();
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&second));
    session.rollback_complete(checkpoint, &()).unwrap();
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);
    assert!(second_bytes > 0);

    // The independent slot shares immutable metadata while its state advances
    // independently. Exchange must move the same owner, not rebuild it.
    session.decode(&FakeTensor(vec![23]), &()).unwrap();
    session.exchange_control_state(&mut branch, &()).unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    assert_eq!(
        session
            .committed_shared_prompt_input_identity()
            .unwrap()
            .identity(),
        &original_key
    );
    drop(first);
    drop(session);
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);
    drop(saved);
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);
    drop(branch);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop(original_key);
}

#[test]
fn distributed_complete_checkpoint_and_parameter_exchange_keep_identity_custody() {
    let pool = crate::memory::host_ledger(u64::MAX, 0).unwrap();
    let first = charged_identity(&pool, "distributed first");
    let first_bytes = first.capacity_bytes().unwrap();
    let (mut session, _, _, _, _) = partitioned_parameter_control_session(
        0,
        DistributedExecutionPhase::PromptCacheLoadPreflight,
        None,
        true,
    );
    session
        .prefill_input_result_with_shared_identity(
            Ok(&FakeTensor(vec![3, 7])),
            Some(first.clone()),
            &(),
            &mut eredu_runtime::NoopObserver,
        )
        .unwrap();
    let checkpoint = session.checkpoint_complete_distributed(&()).unwrap();
    let second = charged_identity(&pool, "distributed second");
    session
        .prefill_input_result_with_shared_identity_and_readout(
            Ok(&FakeTensor(vec![11, 13])),
            Some(second.clone()),
            eredu_core::OutputDemand::LastPosition,
            &(),
            &mut eredu_runtime::NoopObserver,
        )
        .unwrap();
    session
        .rollback_complete_distributed(checkpoint, &())
        .unwrap();
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);
    let (installation, custody) =
        super::original_resident_reset::publication::construct_reset(&session, &pool);
    let mut empty = session
        .prepare_parameter_state_reset(installation)
        .map_err(|(cause, _)| cause)
        .unwrap();
    assert!(empty.shared_prompt_input_identity().is_none());
    session.exchange_parameter_state_reset(&mut empty).unwrap();
    assert!(session.committed_shared_prompt_input_identity().is_none());
    assert!(empty
        .shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    session.exchange_parameter_state_reset(&mut empty).unwrap();
    assert!(session
        .committed_shared_prompt_input_identity()
        .unwrap()
        .same_storage(&first));
    assert!(empty.shared_prompt_input_identity().is_none());
    drop(first);
    drop(empty);
    drop(custody);
    assert_eq!(pool.payload_used_bytes().unwrap(), first_bytes);
    drop(session);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

struct SharedSource {
    geometry: eredu_core::InferenceGeometry,
    identity: SharedPreparedInputCacheIdentity,
    shared_calls: Rc<Cell<usize>>,
    fail_final: bool,
}

impl
    PreparedPrefillSource<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    > for SharedSource
{
    type Chunk = FakeTensor;
    fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(
        &self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        if self.fail_final && chunk.input.end == self.geometry.input_positions {
            return Err(Error::backend("injected final chunk failure"));
        }
        Ok(FakeTensor(vec![7, 11]))
    }
    fn input<'a>(&'a self, chunk: &'a FakeTensor) -> &'a FakeTensor {
        chunk
    }
    fn shared_cache_identity(&self) -> Option<SharedPreparedInputCacheIdentity> {
        self.shared_calls.set(self.shared_calls.get() + 1);
        Some(self.identity.clone())
    }
}

#[test]
fn scheduled_prefill_uses_shared_source_only_at_final_commit_and_retires_failed_candidate() {
    for fail_final in [false, true] {
        let pool = crate::memory::host_ledger(u64::MAX, 0).unwrap();
        let identity = charged_identity(&pool, "scheduled shared prompt");
        let bytes = identity.capacity_bytes().unwrap();
        let key = identity.identity().clone();
        let (mut session, _) = session();
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 2,
            max_output_tokens: 1,
            prefill_chunk_positions: 1,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let execution = session.inference_execution_identity().clone();
        let request: InferenceRequest = pool
            .reserve(&execution, &mock_inference_admission(geometry))
            .unwrap()
            .into();
        let shared_calls = Rc::new(Cell::new(0));
        let source = SharedSource {
            geometry,
            identity,
            shared_calls: shared_calls.clone(),
            fail_final,
        };
        let mut observer = eredu_runtime::NoopObserver;
        let mut executor =
            SessionPrefill::new(&mut session, source, &request, &(), &mut observer).unwrap();
        let mut driver = PrefillDriver::new(
            &execution,
            &request,
            geometry,
            eredu_core::GenerationCancellationToken::new(),
        )
        .unwrap();
        assert!(matches!(
            driver.step(&mut executor).unwrap(),
            PrefillProgress::Chunk { output: None, .. }
        ));
        assert_eq!(shared_calls.get(), 0);
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes + 384);
        let result = driver.run(&mut executor, |_, output| {
            if let Some(output) = output {
                assert_eq!(output, FakeTensor(vec![7, 11, 5]));
            }
        });
        assert_eq!(shared_calls.get(), 1);
        if fail_final {
            assert!(result.is_err());
        } else {
            assert!(matches!(result.unwrap(), PrefillOutcome::Complete));
        }
        drop(executor);
        drop(driver);
        if fail_final {
            assert!(session.committed_shared_prompt_input_identity().is_none());
            assert_eq!(session.report().unwrap().state_report(), &[1]);
            assert_eq!(pool.live_charge_bytes().unwrap(), mock_reservation_bytes());
        } else {
            assert_eq!(
                session
                    .committed_shared_prompt_input_identity()
                    .unwrap()
                    .identity(),
                &key
            );
            assert_eq!(session.report().unwrap().state_report(), &[2]);
            assert_eq!(pool.payload_used_bytes().unwrap(), bytes + 384);
        }
        drop((session, request));
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}
