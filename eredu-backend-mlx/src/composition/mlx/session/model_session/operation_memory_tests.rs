use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError};

fn zero_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless operation admission fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(0),
    })
}

fn zero_payload_ledger() -> MemoryLedger {
    let probe = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let controls = probe
        .reservation_requirements(&zero_admission(), None)
        .unwrap()
        .get(probe.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    crate::memory_fixture::ledger(controls, 0).unwrap()
}

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn fixture(
    stream: &Stream,
    operation_pool: &MemoryLedger,
) -> (
    ModelRuntime<MlxBackend<'static>>,
    MlxBackend<'static>,
    tempfile::TempDir,
) {
    // Keep all existing model/input payloads in their original domain. Only
    // admission of the new ordinary operation belongs to the reserved pool.
    let model_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let source = MlxBackend::new(stream, stream).with_memory_ledger(model_pool);
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&source, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let backend = MlxBackend::new(stream, stream).with_memory_ledger(operation_pool.clone());
    (
        ModelRuntime::from_prepared(backend, model).unwrap(),
        source,
        root,
    )
}

fn populate(runtime: &mut ModelRuntime<MlxBackend<'_>>, stream: &Stream) -> MlxModelOutput {
    // The test-only accessor acquires authority in the original model domain.
    // Populating through ordinary ingress would also retain a context-domain
    // owner and prevent installation of our isolated reservation there.
    let logits = runtime
        .session_mut()
        .neutral_prediction_target_mut()
        .unwrap()
        .decode(&Array::from_slice(&[1_u32], &[1, 1]), stream)
        .unwrap();
    let values = logits.evaluated().unwrap().as_slice::<f32>().to_vec();
    assert!(values.iter().any(|value| value.abs() > 1e-6));
    MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
}

fn assert_reserved(error: &(dyn std::error::Error + 'static)) {
    assert_memory(error, WorkingMemoryError::ReservedWorkActive);
}
fn assert_missing_admission(error: &(dyn std::error::Error + 'static)) {
    assert_memory(error, WorkingMemoryError::UnknownBound);
}
fn assert_memory(error: &(dyn std::error::Error + 'static), expected: WorkingMemoryError) {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            assert_eq!(*memory, expected);
            return;
        }
        current = current
            .source()
            .expect("preserve original domain admission error");
    }
}

fn settle(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &MemoryLedger) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        runtime.session().ensure_no_submission_in_flight().is_ok()
            && pool.unquoted_owner_count().unwrap() == 0
    });
}

fn unchanged(
    pool: &MemoryLedger,
    before: (
        paths::Counts,
        eredu_runtime::working_memory::MemoryLedgerSnapshot,
    ),
) {
    assert_eq!(paths::snapshot(), before.0);
    assert_eq!(pool.snapshot().unwrap(), before.1);
}

#[derive(Default)]
struct Observer(usize);
impl RuntimeActivationObserver<MlxTensor, Error> for Observer {
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.0 += 1;
        Ok(())
    }
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        self.0 += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.0 += 1;
        Ok(())
    }
    fn finish_transaction(&mut self, _: eredu_core::DistributedCommitEpoch, _: bool) {
        self.0 += 1;
    }
}

fn observation_request() -> ObservationRequest {
    ObservationRequest::selected([eredu_core::ObservationSelector::Exact(
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
    )])
}

#[test]
fn ordinary_reset_rejects_reserved_backend_domain_without_clearing_populated_state() {
    let stream = stream();
    let pool = zero_payload_ledger();
    let (mut runtime, _source, _root) = fixture(&stream, &pool);
    let output = populate(&mut runtime, &stream);
    settle(&runtime, &pool);
    let state = runtime.session().payload.model.erased().state_snapshot();
    assert!(state.iter().any(|(offset, _)| *offset > 0));
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let before = (paths::snapshot(), pool.snapshot().unwrap());
    let resets = paths::session_reset_attempts();
    assert_reserved(&runtime.reset().unwrap_err());
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        state
    );
    assert_eq!(paths::session_reset_attempts(), resets);
    unchanged(&pool, before);
    assert!(runtime.session().ensure_no_submission_in_flight().is_ok());
    drop((reservation, output));
}

#[test]
fn ordinary_prefill_variants_reject_before_forward_and_observer_callbacks() {
    let stream = stream();
    let pool = zero_payload_ledger();
    let (mut runtime, source, _root) = fixture(&stream, &pool);
    let output = populate(&mut runtime, &stream);
    let prompt = MlxBackend::prepare_text_prompt(&source, vec![2, 3, 4]).unwrap();
    settle(&runtime, &pool);
    let state = runtime.session().payload.model.erased().state_snapshot();
    let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let before = (paths::snapshot(), pool.snapshot().unwrap());
    assert_missing_admission(&runtime.prefill(prompt.clone()).err().unwrap());
    assert_missing_admission(
        &runtime
            .prefill_cancellable(
                prompt.clone(),
                &eredu_core::GenerationCancellationToken::new(),
            )
            .err()
            .unwrap(),
    );
    let mut observer = Observer::default();
    assert_missing_admission(
        &runtime
            .session_mut()
            .submit_prefill_with_observer(&backend, prompt.clone(), &mut observer)
            .err()
            .unwrap(),
    );
    assert_missing_admission(
        &runtime
            .inspect_prefill(prompt, &observation_request())
            .err()
            .unwrap(),
    );
    assert_eq!(observer.0, 0);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        state
    );
    unchanged(&pool, before);
    assert!(runtime.session().ensure_no_submission_in_flight().is_ok());
    drop((reservation, output));
}

#[test]
fn ordinary_decode_variants_reject_before_input_construction_forward_and_observation() {
    let stream = stream();
    let pool = zero_payload_ledger();
    let (mut runtime, _source, _root) = fixture(&stream, &pool);
    let output = populate(&mut runtime, &stream);
    let input = Array::from_slice(&[2_u32], &[1, 1]);
    settle(&runtime, &pool);
    let state = runtime.session().payload.model.erased().state_snapshot();
    let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let before = (paths::snapshot(), pool.snapshot().unwrap());
    let inputs = paths::session_input_creation_attempts();
    assert_missing_admission(&runtime.decode(input.clone()).err().unwrap());
    assert_missing_admission(
        &runtime
            .session_mut()
            .submit_token_decode(&backend, 2)
            .err()
            .unwrap(),
    );
    let mut observer = Observer::default();
    let mut input_calls = 0;
    assert_missing_admission(
        &runtime
            .session_mut()
            .submit_decode_with_observer(
                &backend,
                || {
                    input_calls += 1;
                    Ok(input.clone())
                },
                &mut observer,
            )
            .err()
            .unwrap(),
    );
    assert_missing_admission(
        &runtime
            .inspect_decode(input, &observation_request())
            .err()
            .unwrap(),
    );
    assert_eq!(input_calls, 0);
    assert_eq!(observer.0, 0);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        state
    );
    unchanged(&pool, before);
    assert!(runtime.session().ensure_no_submission_in_flight().is_ok());
    drop((reservation, output));
}

#[test]
fn completed_output_observation_rejects_reserved_backend_domain_without_touching_output() {
    let stream = stream();
    let pool = zero_payload_ledger();
    let (mut runtime, _source, _root) = fixture(&stream, &pool);
    let output = populate(&mut runtime, &stream);
    settle(&runtime, &pool);
    let array = output.logits().unwrap().as_array();
    let allocation = array.allocation_info().unwrap();
    let values = array.evaluated().unwrap().as_slice::<f32>().to_vec();
    let state = runtime.session().payload.model.erased().state_snapshot();
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let before = (paths::snapshot(), pool.snapshot().unwrap());
    assert_reserved(&runtime.observe_output(&output).unwrap_err());
    assert_eq!(array.allocation_info().unwrap(), allocation);
    assert_eq!(array.evaluated().unwrap().as_slice::<f32>(), values);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        state
    );
    unchanged(&pool, before);
    assert!(runtime.session().ensure_no_submission_in_flight().is_ok());
    drop(reservation);
    let observed = runtime.observe_output(&output).unwrap();
    assert!(!observed.is_empty());
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
