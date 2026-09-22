//! Actual original admission and local ownership only. These tests deliberately
//! do not claim successful authenticated install without the core options hook.
use super::text_quote::{CaptureFundingProbe, CapturedFundingQuote};
use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host,
};
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{capture::*, TextGenerationBackend};
use eredu_runtime::working_memory::CaptureRunHostPlan;

type Runtime = ModelRuntime<MlxBackend<'static>>;
fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0))
}
fn source(runtime: &Runtime) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "original-logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule {
            prefill: false,
            ..Default::default()
        },
        slices: vec![],
        transform: CaptureTransform::Preview { max_elements: 3 },
    });
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        raw.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
        )
        .unwrap(),
    )
}
fn admit(runtime: &Runtime, source: &SharedCapturePlan) -> MlxTextPreparation {
    let ids = vec![2, 5, 7];
    let (request, quote) = text_quote::admit_with_capture(
        runtime,
        &disk::evidence(&ids),
        disk::config(0.0, 1, u64::MAX),
        &disk::Controller::default(),
        source,
    )
    .unwrap();
    MlxTextPreparation {
        request: Some(request),
        chunk: std::num::NonZeroU64::new(1),
        quote: Some(quote),
    }
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> &'a T {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return found;
        }
        error = error.source().expect("original typed failure source");
    }
}
fn finish_runtime(runtime: Runtime, stream: &Stream) {
    runtime.synchronize().unwrap();
    drop(runtime);
    stream.synchronize().unwrap();
    safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
        .unwrap()
        .synchronize()
        .unwrap();
}

fn settle_terminal(pool: &MemoryLedger, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        // Terminal native graph owners retire in bounded batches. Each record
        // needs its own completion proof; this empty event submits no tensor work.
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        disk::reclaim();
        pool.fixture_host_charge().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}

#[test]
fn one_use_owner_moves_actual_source_and_keeps_host_hold_after_quote_retirement() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = host::runtime(&stream, &pool, None);
    let source = source(&runtime);
    let alias = source.clone();
    let independent = self::source(&runtime);
    let baseline = pool.fixture_host_charge().unwrap();
    let h = CaptureRunHostPlan::prepare(&source)
        .unwrap()
        .initialization_peak_bytes();
    let c = source.capacity_bytes().unwrap();
    let preparation = admit(&runtime, &source);
    let quote = preparation.quote.as_ref().unwrap().clone();
    let used = pool.fixture_host_charge().unwrap();
    let native = paths::snapshot();
    let error = quote
        .take_capture_installation(runtime.session(), &independent)
        .err()
        .unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), used);
    let mut installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    assert!(installed.source().same_storage(&alias));
    assert!(std::ptr::eq(
        installed.source().admission(),
        alias.admission()
    ));
    assert_eq!(installed.collector().spent_steps(), 0);
    assert!(!installed.collector_mut().has_pending_step());
    installed.validate_sources(&pool).unwrap();
    let foreign = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&installed.validate_sources(&foreign).unwrap_err()),
        WorkingMemoryError::IdentityMismatch
    ));
    let error = quote
        .take_capture_installation(runtime.session(), &source)
        .err()
        .unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::PreparationAlreadyStarted
    ));
    assert_eq!(paths::snapshot(), native);
    assert_eq!(pool.fixture_host_charge().unwrap(), used);
    let protected = installed.span_workspace().protected_host_bytes();
    drop((quote, preparation));
    // The original bank and aggregate controls remain, while transient native
    // work is closed. Exact values come from this actual installed owner.
    disk::settle(&pool, baseline + h + protected + c);
    installed.validate_sources(&pool).unwrap();
    drop(installed);
    disk::settle(&pool, baseline + protected + c);
    drop((source, independent));
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, protected + c);
    drop(alias);
    settle_terminal(&pool, 0);
}

#[test]
fn installed_witness_rechecks_inherited_origin_after_bank_moves_and_quote_drops() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = host::runtime(&stream, &pool, None);
    let source = source(&runtime);
    let old = admit(&runtime, &source);
    let abandoned = old.quote.as_ref().unwrap().funding_scope().unwrap();
    let current = admit(&runtime, &source);
    let installed = current
        .quote
        .as_ref()
        .unwrap()
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    installed.validate_sources(&pool).unwrap();
    drop(current);
    installed.validate_sources(&pool).unwrap();
    drop(abandoned);
    let used = pool.fixture_host_charge().unwrap();
    let native = paths::snapshot();
    for _ in 0..2 {
        assert!(matches!(
            cause::<WorkingMemoryError>(&installed.validate_sources(&pool).unwrap_err()),
            WorkingMemoryError::ExecutionFenced
        ));
        assert_eq!(installed.collector().spent_steps(), 0);
        assert!(installed.source().same_storage(&source));
        assert_eq!(pool.fixture_host_charge().unwrap(), used);
    }
    assert_eq!(paths::snapshot(), native);
    drop((installed, old, source));
    finish_runtime(runtime, &stream);
    assert!(
        pool.fixture_host_charge().unwrap() > 0,
        "unresolved inherited source remains quarantined"
    );
}

#[test]
fn actual_funded_sampler_copy_rejects_capture_before_work_and_collector_slots_are_exclusive() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = host::runtime(&stream, &pool, None);
    let source = source(&runtime);
    // The genuine core options path binds and prepares this capture-only run.
    // Intercept its actual Sampling return, before Instrumentation; no R bank
    // or substitute state can account for the copy rejection below.
    let (preparation, mut state) = text_quote::capture_only_sampling_for_test(
        &mut runtime,
        &source,
        disk::config(0.0, 1, u64::MAX),
    );
    let quote = preparation.quote.as_ref().unwrap();
    assert!(state.sampling.sampler.is_funded());
    assert!(quote.has_capture());
    require_empty(&state).unwrap();
    let native = paths::snapshot();
    let before = pool.fixture_host_charge().unwrap();
    let revision = runtime.session().payload.model.erased().state_snapshot();
    let error = quote.validate_capture_copy().unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::UnknownBound
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), before);
    assert_eq!(paths::snapshot(), native);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        revision
    );
    runtime.session().authority.borrow().require_idle().unwrap();
    // Ownership-only setup: no core context is fabricated or install success claimed.
    state.funded_capture = Some(
        quote
            .take_capture_installation(runtime.session(), &source)
            .unwrap(),
    );
    assert!(matches!(
        cause::<WorkingMemoryError>(&require_empty(&state).unwrap_err()),
        WorkingMemoryError::PreparationAlreadyStarted
    ));
    drop(state.funded_capture.take());
    state.capture = Some(eredu_runtime::capture::CaptureSession::new(source.clone()));
    assert!(matches!(
        cause::<WorkingMemoryError>(&require_empty(&state).unwrap_err()),
        WorkingMemoryError::PreparationAlreadyStarted
    ));
    drop(state.capture.take());
    require_empty(&state).unwrap();
    drop((state, preparation, source));
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, 0);
}

mod candidates;
mod integration;

mod failure;

mod original_native;
