use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host, text_funding, text_quote,
};
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, ControlledTextGenerationError, GenerationSequenceRequest,
    TextGeneration, TextGenerationBackend, TextGenerationDriver, TextGenerationInput,
    TextPreparationOptions, TokenTerminalSignals, capture::*,
};
use std::error::Error as _;

type Runtime = ModelRuntime<MlxBackend<'static>>;
mod prepared_residency;
pub(in crate::composition::mlx::session::model_session) use prepared_residency::PreparedResidencyFixture;
thread_local! { static PROCESS_SOURCE_BASELINE: Cell<u64> = const {Cell::new(0)}; }
pub(in crate::composition::mlx::session::model_session::text_quote) fn stream() -> Stream {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let streams =
        crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_factory(&pool)
            .unwrap()
            .expect("original native stream factory");
    let stream = streams.execution().clone();
    disk::reclaim();
    PROCESS_SOURCE_BASELINE.with(|baseline| {
        if baseline.get() == 0 {
            baseline.set(pool.fixture_host_charge().unwrap());
        }
    });
    stream
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn load(
    _stream: &Stream,
    pool: &MemoryLedger,
    route: usize,
) -> (Runtime, tempfile::TempDir) {
    assert!(pool.same_ledger(&crate::backend::managed_memory::ledger()));
    disk::reclaim();
    let before = pool.fixture_host_charge().unwrap();
    let (target, artifact) = prepared_residency::target(route, None);
    let after = pool.fixture_host_charge().unwrap();
    let birth = after
        .checked_sub(before)
        .expect("factory source birth before model materialization");
    PROCESS_SOURCE_BASELINE
        .with(|baseline| baseline.set(baseline.get().checked_add(birth).unwrap()));
    let runtime = target.into_runtime().unwrap();
    runtime.backend().validate_original_stream_owners().unwrap();
    (runtime, artifact)
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn config(
    maximum: usize,
    capacity: u64,
) -> TextGenerationConfig {
    let original = disk::config(0.0, 1, capacity);
    let mut sampling = original.sampling();
    sampling.max_new_tokens = Some(maximum);
    TextGenerationConfig::new(sampling)
        .with_seed(19)
        .with_inference_policy(original.inference_policy().clone())
}
/// Explicit component ceilings shared by genuine C1/C2/C3 prefill fixtures.
/// As in the existing native graph fixture, 1 MiB is an enforced caller Record
/// capacity, not a claim that every native producer has a complete fit proof.
pub(in crate::composition::mlx::session::model_session::text_quote) fn chunked_original(
    config: TextGenerationConfig,
) -> TextGenerationConfig {
    let mut policy = config.inference_policy().clone();
    policy.prefill_chunk_positions = std::num::NonZeroU64::new(2);
    policy.graph_metadata_capacity_bytes = std::num::NonZeroU64::new(4 << 20);
    policy.submission_tracking_capacity_bytes = std::num::NonZeroU64::new(1 << 20);
    config.with_inference_policy(policy.clone())
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn source(
    runtime: &Runtime,
) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "original-r".into(),
        path: "readout.embedding".into(),
        schedule: CaptureSchedule {
            prefill: true,
            ..Default::default()
        },
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    plan.limits.per_step = CaptureUsage {
        captures: 1,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    plan.limits.cumulative = CaptureUsage {
        captures: 4,
        ..plan.limits.per_step
    };
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    )
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn finish(
    runtime: Runtime,
    stream: &Stream,
) {
    runtime.synchronize().unwrap();
    drop(runtime);
    stream.synchronize().unwrap();
}
#[track_caller]
pub(in crate::composition::mlx::session::model_session::text_quote) fn settle(
    pool: &MemoryLedger,
    bytes: u64,
) {
    let bytes = if pool.same_ledger(&crate::backend::managed_memory::ledger()) {
        bytes.checked_add(PROCESS_SOURCE_BASELINE.get()).unwrap()
    } else {
        bytes
    };
    let caller = std::panic::Location::caller();
    let started = std::time::Instant::now();
    let mut reported = false;
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        safemlx::memory::clear_cache();
        disk::reclaim();
        let used = pool.fixture_host_charge().unwrap();
        let unquoted = pool.unquoted_owner_count().unwrap();
        let complete = used == bytes && unquoted == 0;
        if !complete && !reported && started.elapsed().as_secs() >= 9 {
            eprintln!(
                "native fixture retirement at {caller}: expected={bytes}, used={used}, unquoted={unquoted}"
            );
            reported = true;
        }
        complete
    });
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> &'a T {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return value;
        }
        error = error.source().expect("original concrete source");
    }
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn start(
    runtime: &mut Runtime,
    maximum: usize,
    eos: &[u32],
    route: usize,
    capacity: u64,
    capture: Option<&SharedCapturePlan>,
) -> Result<RetainedGenerationSequence, BackendFailure> {
    start_with_decoder(runtime, maximum, eos, route, capacity, capture, None)
}
fn start_with_decoder(
    runtime: &mut Runtime,
    maximum: usize,
    eos: &[u32],
    route: usize,
    capacity: u64,
    capture: Option<&SharedCapturePlan>,
    decoder: Option<&dyn eredu_core::GenerationDecoderInput>,
) -> Result<RetainedGenerationSequence, BackendFailure> {
    let input = eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap();
    let options = capture.map(|source| TextPreparationOptions {
        interventions: None,
        capture: Some(source.clone()),
    });
    let claim = GenerationSequenceRequest::new(maximum, eos);
    let claim = if let Some(decoder) = decoder {
        claim.with_decoder(decoder)
    } else {
        claim
    };
    match route {
        0 => {
            let mut run = TextGeneration::from_token_ids_with_sequence(
                runtime,
                input,
                config(maximum, capacity),
                TokenFilter::All,
                options,
                claim,
            )?;
            let sequence = run.take_prepared_sequence().unwrap();
            assert!(run.take_prepared_sequence().is_none());
            Ok(sequence)
        }
        1 => {
            let mut run = ControlledTextGeneration::from_token_ids_with_sequence(
                runtime,
                input,
                config(maximum, capacity),
                disk::Controller::default(),
                options,
                claim,
            )
            .map_err(|cause| match cause {
                ControlledTextGenerationError::Preparation(cause) => cause,
                other => panic!("unexpected controlled error: {other}"),
            })?;
            let sequence = run.take_prepared_sequence().unwrap();
            assert!(run.take_prepared_sequence().is_none());
            Ok(sequence)
        }
        _ => {
            let mut driver = TextGenerationDriver::new(runtime);
            let mut run = driver
                .start_token_ids_with_sequence(
                    input,
                    config(maximum, capacity),
                    disk::Controller::default(),
                    options,
                    claim,
                )
                .map_err(|cause| match cause {
                    ControlledTextGenerationError::Preparation(cause) => cause,
                    other => panic!("unexpected detached error: {other}"),
                })?;
            let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
            assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
            Ok(sequence)
        }
    }
}

#[test]
fn original_native_sequence_admission_uses_actual_core_drivers_and_residency() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for residency in 0..3 {
        for driver in 0..3 {
            for capture in [false, true] {
                let pool = crate::tests::support::test_utils::initialize_original_sources();
                let (mut runtime, _artifact) = load(&stream, &pool, residency);
                let source = capture.then(|| source(&runtime));
                let probe = Probe::new(&runtime, source.as_ref(), false);
                let sequence = start(
                    &mut runtime,
                    4,
                    &[99, 97, 99],
                    driver,
                    u64::MAX,
                    source.as_ref(),
                )
                .unwrap();
                let preparation = probe.take();
                let facts = probe.facts();
                assert_eq!(facts.capture, capture);
                assert_eq!(
                    facts.source, capture,
                    "R-only must not invent a capture source"
                );
                assert!(facts.r > 0 && facts.held >= facts.r);
                assert_eq!(
                    facts.work,
                    text_funding::text_work_control_bytes(4).unwrap()
                );
                assert!(facts.admission >= text_quote::owner_control_bytes().unwrap());
                assert_eq!(probe.0.calls.get(), 1);
                let quote = preparation.quote.as_ref().unwrap();
                assert!(quote.sequence.as_ref().unwrap().pending.borrow().is_none());
                assert!(sequence.tokens().is_empty());
                let mut sequence = sequence;
                assert!(sequence.cancel());
                let ids = sequence.into_token_ids();
                assert!(ids.is_empty());
                let alias = ids.clone();
                drop((preparation, probe));
                finish(runtime, &stream);
                let c = source
                    .as_ref()
                    .map_or(0, |source| source.capacity_bytes().unwrap());
                settle(&pool, facts.held + c);
                drop(source);
                settle(&pool, facts.held);
                drop(ids);
                settle(&pool, facts.held);
                drop(alias);
                settle(&pool, 0);
            }
        }
    }
}

#[test]
fn original_native_sequence_exact_quote_and_short_admission_precede_prompt() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for (capture, rows) in [(false, false), (true, false), (true, true)] {
        let mut required = None;
        let mut r = None;
        for pass in 0..3 {
            let pool = crate::tests::support::test_utils::initialize_original_sources();
            let (mut runtime, _artifact) = load(&stream, &pool, 0);
            let source = capture.then(|| source(&runtime));
            let probe = Probe::new(&runtime, source.as_ref(), rows);
            let baseline = pool.fixture_host_charge().unwrap();
            let inputs = paths::session_input_creation_attempts();
            let native = paths::snapshot();
            let capacity = required.map_or(u64::MAX, |required| {
                baseline + required - u64::from(pass == 1)
            });
            let result = start(&mut runtime, 4, &[99, 97, 99], 2, capacity, source.as_ref());
            if pass == 1 {
                let error = result.unwrap_err();
                assert!(matches!(
                    cause::<WorkingMemoryError>(&error),
                    WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    )
                ));
                assert!(probe.0.admitted.borrow().is_none());
                assert_eq!(probe.0.calls.get(), 0);
                assert_eq!(paths::session_input_creation_attempts(), inputs);
                assert_eq!(paths::snapshot(), native);
                assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
            } else {
                let sequence = result.unwrap();
                let facts = probe.facts();
                if pass == 0 {
                    required = Some(facts.required);
                    r = Some(facts.r);
                } else {
                    assert_eq!(Some(facts.required), required);
                    assert_eq!(Some(facts.r), r);
                }
                let preparation = probe.take();
                drop((sequence, preparation));
            }
            drop((probe, source));
            finish(runtime, &stream);
            settle(&pool, 0);
        }
    }
}

#[test]
fn native_sequence_replays_busy_and_foreign_preflight_retain_no_owner() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    let sequence = start(&mut runtime, 4, &[], 0, u64::MAX, None).unwrap();
    let preparation = probe.take();
    let alias = preparation.clone();
    let quote = preparation.quote.as_ref().unwrap();
    let errors =
        std::array::from_fn::<_, 128, _>(|_| quote.sequence.as_ref().unwrap().take().unwrap_err());
    assert!(
        errors
            .iter()
            .all(|error| error.source().unwrap().is::<Rejection>())
    );
    let other_pool = crate::tests::support::test_utils::initialize_original_sources();
    let (other, _other_artifact) = load(&stream, &other_pool, 0);
    let foreign = std::array::from_fn::<_, 128, _>(|_| {
        super::super::preflight(&other, preparation.request.as_ref().unwrap(), quote)
            .unwrap_err()
            .into_backend_failure()
    });
    assert!(
        foreign
            .iter()
            .all(|error| error.source().unwrap().downcast_ref::<Rejection>()
                == Some(&Rejection::IdentityMismatch))
    );
    drop((sequence, preparation, alias, probe));
    finish(runtime, &stream);
    finish(other, &stream);
    settle(&pool, 0);
    settle(&other_pool, 0);
    drop((errors, foreign));
    let busy_pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut busy_runtime, _busy_artifact) = load(&stream, &busy_pool, 0);
    let busy_probe = Probe::new(&busy_runtime, None, false);
    busy_probe.mode(Mode::Busy);
    let error = start(&mut busy_runtime, 4, &[], 0, u64::MAX, None).unwrap_err();
    assert_eq!(
        error.source().unwrap().downcast_ref::<Rejection>(),
        Some(&Rejection::Busy)
    );
    let preparation = busy_probe.take();
    let bank = preparation
        .quote
        .as_ref()
        .unwrap()
        .sequence
        .as_ref()
        .unwrap()
        .take()
        .unwrap();
    drop((bank, preparation, busy_probe));
    finish(busy_runtime, &stream);
    settle(&busy_pool, 0);
    drop(error);
}

#[test]
fn native_sequence_foreign_genuine_claim_consumes_only_the_original_bank() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::Defer);
    let first = start(&mut runtime, 4, &[97], 0, u64::MAX, None).unwrap_err();
    assert!(first.source().unwrap().is::<Rejection>());
    let original = probe.take();
    let held = probe.facts().held;
    probe.mode(Mode::Normal);
    probe.target(original.clone());
    let failure = start(&mut runtime, 4, &[97], 1, u64::MAX, None).unwrap_err();
    assert_eq!(failure.kind(), BackendFailureKind::InvalidSession);
    assert!(!failure.source().unwrap().is::<Error>());
    assert!(matches!(
        cause::<WorkingMemoryError>(&failure),
        WorkingMemoryError::IdentityMismatch
    ));
    let later = probe.take();
    let errors = std::array::from_fn::<_, 128, _>(|_| {
        original
            .quote
            .as_ref()
            .unwrap()
            .sequence
            .as_ref()
            .unwrap()
            .take()
            .unwrap_err()
    });
    drop((original, later, probe));
    finish(runtime, &stream);
    settle(&pool, held);
    drop(failure);
    settle(&pool, 0);
    drop((first, errors));
}

#[test]
fn native_sequence_fenced_consumed_bank_and_provider_error_keep_original_hold() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for consumed in [false, true] {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let probe = Probe::new(&runtime, None, false);
        probe.mode(if consumed {
            Mode::Fence
        } else {
            Mode::FenceProvider
        });
        let inputs = paths::session_input_creation_attempts();
        let result = start(&mut runtime, 4, &[], 0, u64::MAX, None);
        assert_eq!(paths::session_input_creation_attempts(), inputs);
        let preparation = probe.take();
        let held = probe.facts().held;
        let failure = result.unwrap_err();
        if !consumed {
            let _ = cause::<eredu_core::RetainedSequencePreparationError>(&failure);
        }
        assert!(
            preparation
                .quote
                .as_ref()
                .unwrap()
                .sequence
                .as_ref()
                .unwrap()
                .pending
                .borrow()
                .is_none()
        );
        assert!(matches!(
            cause::<WorkingMemoryError>(&failure),
            WorkingMemoryError::ExecutionFenced
        ));
        assert!(!failure.source().unwrap().is::<Error>());
        drop((preparation, probe));
        finish(runtime, &stream);
        settle(&pool, held);
        drop(failure);
        settle(&pool, 0);
    }
}

#[test]
fn native_sequence_zero_and_dormant_cancel_do_not_refund_original_result_storage() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for maximum in [0, 4] {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let probe = Probe::new(&runtime, None, false);
        let mut sequence = start(&mut runtime, maximum, &[], 0, u64::MAX, None).unwrap();
        let held = probe.facts().held;
        let preparation = probe.take();
        assert_eq!(
            sequence.finish_reason(),
            (maximum == 0).then_some(eredu_core::FinishReason::MaxTokens)
        );
        assert!(sequence.cancel());
        assert!(!sequence.cancel());
        let ids = sequence.into_token_ids();
        assert!(ids.is_empty());
        drop((preparation, probe));
        finish(runtime, &stream);
        settle(&pool, held);
        drop(ids);
        settle(&pool, 0);
    }
}

#[test]
fn native_sequence_real_predictions_and_capture_rows_share_original_result_bank() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    run_real_prediction(false);
}
fn run_real_prediction(with_decoder: bool) {
    use eredu_core::observation::TensorObservationData;
    let stream = stream();
    let mut reference = None;
    for residency in 0..3 {
        for capture in [false, true] {
            let pool = crate::tests::support::test_utils::initialize_original_sources();
            let (mut runtime, _artifact) = load(&stream, &pool, residency);
            let source = capture.then(|| source(&runtime));
            let probe = Probe::new(&runtime, source.as_ref(), capture && residency == 0);
            let options = source.as_ref().map(|source| TextPreparationOptions {
                interventions: None,
                capture: Some(source.clone()),
            });
            let decoder = with_decoder.then(|| decoder::input(4));
            let mut claim = GenerationSequenceRequest::new(4, &[]);
            if let Some(decoder) = &decoder {
                claim = claim.with_decoder(decoder);
            }
            let mut decoded = String::new();
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut run = driver
                .start_token_ids_with_sequence(
                    eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
                    config(4, u64::MAX).with_inference_policy(eredu_core::TextInferencePolicy {
                        prefill_chunk_positions: std::num::NonZeroU64::new(2),
                        memory_limits: eredu_core::MemoryLimitDeclarations::new([(
                            "host".into(),
                            eredu_core::MemoryLimit::Finite(u64::MAX),
                        )]),
                        submission_tracking_capacity_bytes: None,
                        graph_metadata_capacity_bytes: None,
                    }),
                    disk::Controller::default(),
                    options,
                    claim,
                )
                .unwrap();
            let preparation = probe.take();
            let facts = probe.facts();
            let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
            assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
            let mut sequence = sequence.prepare_storage().unwrap();
            let pointer = sequence.tokens().as_ptr();
            if capture {
                let installed = probe.installed();
                assert!(installed.same_source);
                assert_eq!(installed.held, facts.held);
                assert_eq!(installed.r, facts.r);
                assert_eq!(
                    preparation
                        .quote
                        .as_ref()
                        .unwrap()
                        .capture
                        .as_ref()
                        .unwrap()
                        .opening_rows()
                        .is_some(),
                    residency == 0
                );
            }
            let mut actual = Vec::new();
            for _ in 0..4 {
                let token = driver.advance(&mut run).unwrap().unwrap();
                let token_id = token.token_id();
                actual.push(token_id);
                if with_decoder {
                    if let Some(text) = sequence.decode_token(token_id).unwrap() {
                        decoded.push_str(text);
                    }
                }
                sequence
                    .commit(token_id, TokenTerminalSignals::default())
                    .unwrap();
                drop(token);
                let delivery = driver.take_completed_delivery(&mut run).unwrap();
                if capture {
                    let frame = delivery.unwrap();
                    let Some(CapturePayload::SharedTensor(tensor)) = &frame.records()[0].payload
                    else {
                        panic!("actual full capture value");
                    };
                    let TensorObservationData::F32(values) = tensor.data() else {
                        panic!("F32 fixture");
                    };
                    assert!(!values.is_empty());
                    assert!(values.iter().all(|v| v.is_finite()));
                    assert!(values.iter().any(|v| *v != 0.0));
                } else {
                    assert!(delivery.is_none());
                }
            }
            if with_decoder {
                sequence.finish_decoder().unwrap();
                assert_eq!(
                    decoded,
                    decoder::tokenizer()
                        .snapshot()
                        .decode(&actual, false)
                        .unwrap()
                );
                assert!(!decoded.is_empty());
                assert_eq!(probe.0.decoder_takes.get(), 1);
            }
            assert_eq!(sequence.tokens(), actual);
            assert_eq!(sequence.tokens().as_ptr(), pointer);
            let ids = sequence.into_token_ids();
            assert_eq!(ids.as_ptr(), pointer);
            if let Some(expected) = &reference {
                assert_eq!(&actual, expected);
            } else {
                reference = Some(actual);
            }
            let mut iter = ids.into_iter();
            assert!(iter.next().is_some());
            drop((run, driver, preparation, probe));
            finish(runtime, &stream);
            let c = source.as_ref().map_or(0, |s| s.capacity_bytes().unwrap());
            settle(&pool, facts.held + c);
            drop(source);
            settle(&pool, facts.held);
            assert_eq!(iter.len(), 3);
            drop(iter);
            settle(&pool, 0);
        }
    }
}

#[test]
fn native_sequence_taken_bank_stays_spent_after_unwind() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::PanicAfterTake);
    let inputs = paths::session_input_creation_attempts();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = start(&mut runtime, 4, &[], 0, u64::MAX, None);
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"native original bank taken before unwind")
    );
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    let preparation = probe.take();
    let rejected = preparation
        .quote
        .as_ref()
        .unwrap()
        .sequence
        .as_ref()
        .unwrap()
        .take()
        .unwrap_err();
    assert_eq!(
        rejected.source().unwrap().downcast_ref::<Rejection>(),
        Some(&Rejection::Unavailable)
    );
    drop((preparation, probe));
    finish(runtime, &stream);
    settle(&pool, 0);
    drop(rejected);
}

#[test]
fn native_sequence_actual_claim_prices_exact_token_and_eos_requested_layout_delta() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let mut small = None;
    for (maximum, eos) in [(2, vec![99, 97]), (5, vec![99, 97, 99, 101])] {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let probe = Probe::new(&runtime, None, false);
        let mut sequence = start(&mut runtime, maximum, &eos, 0, u64::MAX, None).unwrap();
        let facts = probe.facts();
        if let Some(small) = small {
            assert_eq!(facts.r - small, 4 * (3 + 2));
        } else {
            small = Some(facts.r);
        }
        sequence.cancel();
        let ids = sequence.into_token_ids();
        let preparation = probe.take();
        drop((preparation, probe));
        finish(runtime, &stream);
        settle(&pool, facts.held);
        drop(ids);
        settle(&pool, 0);
    }
}

#[test]
fn native_sequence_saved_copy_rejects_before_source_or_destination_work() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    // A stopped genuine admission still owns its unextracted original bank.
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::Defer);
    let rejected = start(&mut runtime, 4, &[], 0, u64::MAX, None).unwrap_err();
    let preparation = probe.take();
    let quote = preparation.quote.as_ref().unwrap();
    assert!(quote.sequence.as_ref().unwrap().pending.borrow().is_some());
    let before = (pool.fixture_host_charge().unwrap(), paths::snapshot());
    let failure = quote.validate_capture_copy().unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&failure),
        WorkingMemoryError::UnknownBound
    ));
    assert!(quote.sequence.as_ref().unwrap().pending.borrow().is_some());
    assert_eq!(
        (pool.fixture_host_charge().unwrap(), paths::snapshot()),
        before
    );
    drop((preparation, probe));
    finish(runtime, &stream);
    settle(&pool, 0);
    drop((rejected, failure));

    // The historical retained-sequence claim still rejects the public copy
    // boundary after its one result bank has moved out successfully.
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut run = driver
        .start_token_ids_with_sequence(
            eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
            config(4, u64::MAX),
            disk::Controller::default(),
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap();
    let mut sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
    let preparation = probe.take();
    let held = probe.facts().held;
    let before = (pool.fixture_host_charge().unwrap(), paths::snapshot());
    let rejected = driver
        .quiescent(&mut run)
        .err()
        .expect("retained result requires destination admission");
    assert!(matches!(
        cause::<eredu_core::GenerationSequenceAdmissionError>(&rejected),
        eredu_core::GenerationSequenceAdmissionError::CopyNotAdmitted
    ));
    assert_eq!(
        (pool.fixture_host_charge().unwrap(), paths::snapshot()),
        before
    );
    assert!(
        preparation
            .quote
            .as_ref()
            .unwrap()
            .sequence
            .as_ref()
            .unwrap()
            .pending
            .borrow()
            .is_none()
    );
    assert!(sequence.cancel());
    let ids = sequence.into_token_ids();
    assert!(ids.is_empty());
    drop((run, driver, preparation, probe));
    finish(runtime, &stream);
    settle(&pool, held);
    drop(ids);
    settle(&pool, 0);
    drop(rejected);

    // Intercept the actual completed native Sampling value at its return into
    // core. The typed setup failure ends the core borrow; it grants no live
    // continuation or copy permission. No replacement state is constructed.
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::TakeSampling);
    let initial = config(4, u64::MAX);
    let mut sampling_config = initial.sampling();
    sampling_config.temperature = 0.7;
    let config = TextGenerationConfig::new(sampling_config)
        .with_seed(initial.seed())
        .with_inference_policy(initial.inference_policy().clone());
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let rejected = driver
        .start_token_ids_with_sequence(
            eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
            config,
            disk::Controller::default(),
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .err()
        .expect("scoped Sampling interception");
    let _ = cause::<SamplingIntercepted>(&rejected);
    drop(driver);
    let state = probe.take_sampling();
    assert!(probe.0.sampling_taken.get());
    assert!(probe.0.sampling.borrow().is_none());
    let preparation = probe.take();
    let held = probe.facts().held;
    {
        let sampling = &state.sampling;
        let key = sampling
            .prng
            .as_ref()
            .unwrap()
            .as_array()
            .allocation_info()
            .unwrap();
        let before = (
            pool.fixture_host_charge().unwrap(),
            pool.fixture_host_peak().unwrap(),
            paths::snapshot(),
            sampling.next_prediction,
        );
        for _ in 0..16 {
            let failure = preparation
                .quote
                .as_ref()
                .unwrap()
                .validate_capture_copy()
                .unwrap_err();
            assert!(matches!(
                cause::<WorkingMemoryError>(&failure),
                WorkingMemoryError::UnknownBound
            ));
        }
        assert_eq!(
            (
                pool.fixture_host_charge().unwrap(),
                pool.fixture_host_peak().unwrap(),
                paths::snapshot(),
                sampling.next_prediction
            ),
            before
        );
        assert_eq!(
            sampling
                .prng
                .as_ref()
                .unwrap()
                .as_array()
                .allocation_info()
                .unwrap(),
            key
        );
        assert!(
            preparation
                .quote
                .as_ref()
                .unwrap()
                .sequence
                .as_ref()
                .unwrap()
                .pending
                .borrow()
                .is_none()
        );
    }
    drop((state, probe));
    finish(runtime, &stream);
    settle(&pool, held);
    drop(preparation);
    settle(&pool, 0);
    drop(rejected);
}

#[test]
fn claimed_work_rejection_retains_original_error_and_quarantines_only_foreign_scope() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    // The fault swaps two real admitted scopes only after the actual native
    // permit claim. This is internal fault injection, not a public input route.
    let stream = stream();
    let foreign_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut foreign_runtime, _foreign_artifact) = load(&stream, &foreign_pool, 0);
    let foreign_probe = Probe::new(&foreign_runtime, None, false);
    foreign_probe.mode(Mode::Defer);
    let rejected = start(&mut foreign_runtime, 4, &[], 0, u64::MAX, None).unwrap_err();
    let foreign_preparation = foreign_probe.take();
    let foreign_quote = foreign_preparation.quote.as_ref().unwrap();
    let foreign_run = foreign_quote.take_funding_run().unwrap();
    let foreign_scope = foreign_run.scope().unwrap();
    drop((foreign_probe, rejected));

    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::ReplaceWorkScope);
    probe.foreign_scope(foreign_scope);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut run = driver
        .start_token_ids_with_sequence(
            eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
            config(4, u64::MAX),
            disk::Controller::default(),
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap();
    let preparation = probe.take();
    let held = probe.facts().held;
    let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
    assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
    let before = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let failure = driver
        .advance(&mut run)
        .err()
        .expect("actual claimed Work rejects the foreign scope");
    assert_eq!(probe.operation_faults(), 1);
    assert_eq!(
        paths::snapshot(),
        before,
        "Work fails before model/native input work"
    );
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    let original = cause::<Error>(&failure);
    assert!(matches!(original, Error::OriginalControl(_)));
    assert!(
        !original.model_state_preserved(),
        "do not invent a rollback witness"
    );
    assert!(matches!(
        cause::<WorkingMemoryError>(&failure),
        WorkingMemoryError::IdentityMismatch
    ));
    assert!(matches!(
        foreign_run.scope(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(matches!(
        foreign_run.validate_reservation(foreign_quote.request().memory_reservation()),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(
        preparation
            .quote
            .as_ref()
            .unwrap()
            .sequence
            .as_ref()
            .unwrap()
            .pending
            .borrow()
            .is_none()
    );
    drop((sequence, run, driver, preparation, probe));
    finish(runtime, &stream);
    settle(&pool, held);
    let foreign_retained = foreign_pool.fixture_host_charge().unwrap();
    drop(failure);
    settle(&pool, 0);
    assert_eq!(
        foreign_pool.fixture_host_charge().unwrap(),
        foreign_retained,
        "original error never refunds or adopts the foreign quarantined account"
    );
    // Keep its intentional quarantine; no manufactured completion/refund.
    drop((foreign_run, foreign_preparation));
    finish(foreign_runtime, &stream);
}

#[test]
fn consumed_installation_error_keeps_original_source_and_replays_add_no_custody() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime);
    let source_bytes = source.capacity_bytes().unwrap();
    // CaptureAdmission joins this actual registered source before sealing.
    // The detached identity contains no paths payload or accounting attachment.
    let (paths_key, paths_bytes) = {
        let paths = runtime
            .session()
            .payload
            .model
            .erased()
            .shared_observation_paths()
            .unwrap();
        (
            crate::backend::runtime::residency::storage::StorageIdentity::HostMetadata(
                paths.identity().registry_key().clone(),
            ),
            paths.capacity_bytes().unwrap(),
        )
    };
    assert!(paths_bytes > 0);
    let probe = Probe::new(&runtime, Some(&source), true);
    probe.mode(Mode::InstallRowsTwice);
    let before = paths::snapshot();
    let failure = start(&mut runtime, 4, &[], 0, u64::MAX, Some(&source)).unwrap_err();
    assert_eq!(probe.installation_faults(), 1);
    assert_eq!(paths::snapshot().forwards, before.forwards);
    assert_eq!(
        paths::snapshot().state_publications,
        before.state_publications
    );
    let preparation = probe.take();
    let held = probe.facts().held;
    // Original complete native source remains reachable through the direct
    // core source owner; the leaf is not replaced by a replay diagnostic.
    let _ = cause::<Error>(&failure);
    let mut replays = Vec::new();
    for _ in 0..32 {
        let error = preparation
            .quote
            .as_ref()
            .unwrap()
            .take_capture_installation(runtime.session(), &source)
            .err()
            .unwrap();
        assert!(matches!(
            cause::<WorkingMemoryError>(&error),
            WorkingMemoryError::PreparationAlreadyStarted
        ));
        assert!(!matches!(error, Error::OriginalControl(_)));
        replays.push(error);
    }
    assert_eq!(probe.installation_faults(), 1);
    drop((preparation, probe, source));
    finish(runtime, &stream);
    let protected = held
        .checked_add(source_bytes)
        .and_then(|bytes| bytes.checked_add(paths_bytes))
        .unwrap();
    settle(&pool, protected);
    // Prove the escaped original guard retains the declared source itself.
    // The temporary verification pin is dropped before the actual error.
    drop(
        pool.pin_registered_storage([(paths_key.clone(), paths_bytes)])
            .unwrap(),
    );
    assert_eq!(pool.fixture_host_charge().unwrap(), protected);
    drop(failure);
    settle(&pool, 0);
    assert!(matches!(
        pool.pin_registered_storage([(paths_key, paths_bytes)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        replays.len(),
        32,
        "retained replay errors cannot prolong original custody"
    );
    drop(replays);
}

#[test]
fn expired_rows_release_only_at_explicit_idle_boundary_and_reuse_original_admission() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    // Genuine core claim + private row proposal + original seal/installation.
    // No prediction is submitted: isolate the weak's accounting tail without
    // an independently surviving newly published KV backing or captured frame.
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    runtime.synchronize().unwrap();
    let baseline = pool.fixture_host_charge().unwrap();
    // Retain only the detached key, never a second paths/source payload owner.
    // This registration precedes the original run and is already in baseline.
    let (paths_key, paths_bytes) = {
        let paths = runtime
            .session()
            .payload
            .model
            .erased()
            .shared_observation_paths()
            .unwrap();
        (
            crate::backend::runtime::residency::storage::StorageIdentity::HostMetadata(
                paths.identity().registry_key().clone(),
            ),
            paths.capacity_bytes().unwrap(),
        )
    };
    assert!(paths_bytes > 0);
    assert!(baseline >= paths_bytes);
    drop(
        pool.pin_registered_storage([(paths_key.clone(), paths_bytes)])
            .unwrap(),
    );
    assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    let mut required = None;
    for _ in 0..2 {
        let source = source(&runtime);
        let source_bytes = source.capacity_bytes().unwrap();
        let source_key = crate::backend::runtime::residency::storage::StorageIdentity::CapturePlan(
            source.storage_identity().clone(),
        );
        let probe = Probe::new(&runtime, Some(&source), true);
        let sequence = start(
            &mut runtime,
            4,
            &[],
            2,
            required.map_or(u64::MAX, |bytes| baseline + bytes),
            Some(&source),
        )
        .unwrap();
        let preparation = probe.take();
        let facts = probe.facts();
        if let Some(required) = required {
            assert_eq!(facts.required, required);
        }
        required = Some(facts.required);
        let rows = preparation
            .quote
            .as_ref()
            .unwrap()
            .capture
            .as_ref()
            .unwrap()
            .opening_rows()
            .expect("actual installed original row bank");
        let semantic_drops = Rc::new(Cell::new(0));
        rows.observe_retirement_for_test(semantic_drops.clone());
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, true)
        );
        assert!(probe.installed().same_source);
        // One prior owner remains. Existing synchronize must not clear it or
        // change its source/binding/once-only flag.
        drop((sequence, preparation, probe, source));
        runtime.synchronize().unwrap();
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            rows.strong_count_for_test() == 1
        });
        assert_eq!(semantic_drops.get(), 0);
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, true)
        );
        assert!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .install_opening_rows(&rows)
                .is_err()
        );
        drop(rows);
        // No ordinary-idle cleanup inside this source-only status inspection.
        // The semantic rows are already gone, but their Rc block and original
        // full accounting/source witness remain under the expired weak.
        assert_eq!(semantic_drops.get(), 1);
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, false)
        );
        let weak_tail = baseline
            .checked_add(facts.held)
            .and_then(|bytes| bytes.checked_add(source_bytes))
            .unwrap();
        settle(&pool, weak_tail);
        drop(
            pool.pin_registered_storage([
                (paths_key.clone(), paths_bytes),
                (source_key.clone(), source_bytes),
            ])
            .unwrap(),
        );
        assert_eq!(pool.fixture_host_charge().unwrap(), weak_tail);
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, false)
        );
        // A real active authority is distinct from a nested slot loan. This
        // exclusion-only lease submits no native work and needs no completion.
        let lease = runtime
            .session()
            .authority
            .borrow_mut()
            .begin_submission()
            .unwrap();
        let authority_busy = runtime.synchronize().unwrap_err();
        assert_eq!(authority_busy.kind(), BackendFailureKind::Busy);
        assert_eq!(
            runtime.session().authority.borrow().require_idle(),
            Err(eredu_core::SessionAuthorityError::Busy)
        );
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, false)
        );
        assert_eq!(pool.fixture_host_charge().unwrap(), weak_tail);
        // No native allocation/execution occurred under this actual lease.
        drop(lease);
        runtime.session().authority.borrow().require_idle().unwrap();
        drop(authority_busy);
        let busy = runtime
            .session()
            .payload
            .model
            .erased()
            .busy_rows_retirement_for_test()
            .unwrap_err();
        let _ = cause::<std::cell::BorrowMutError>(&busy);
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, false)
        );
        assert_eq!(pool.fixture_host_charge().unwrap(), weak_tail);
        drop(busy);
        // Inject only the existing outer poison condition, with no failed
        // native submission or completion claim. The ordinary boundary must
        // reject without touching even an expired slot or its original charge.
        struct RestorePoison(Rc<Cell<bool>>);
        impl Drop for RestorePoison {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        let restore = RestorePoison(runtime.session().poison.clone());
        runtime.session().poison.set(true);
        assert!(runtime.synchronize().is_err());
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (true, false)
        );
        assert_eq!(pool.fixture_host_charge().unwrap(), weak_tail);
        drop(restore);
        runtime.synchronize().unwrap();
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .opening_rows_status_for_test()
                .unwrap(),
            (false, false)
        );
        settle(&pool, baseline);
        // Cleanup releases the prior original C, while the loaded model still
        // owns the exact registered paths already counted in baseline.
        assert!(matches!(
            pool.pin_registered_storage([(source_key, source_bytes)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        drop(
            pool.pin_registered_storage([(paths_key.clone(), paths_bytes)])
                .unwrap(),
        );
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
        assert_eq!(semantic_drops.get(), 1);
    }
    finish(runtime, &stream);
    settle(&pool, 0);
    assert!(matches!(
        pool.pin_registered_storage([(paths_key, paths_bytes)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}

mod decoder;

pub(in crate::composition::mlx::session::model_session::text_quote) mod token_input;

mod scoped_completion;

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
