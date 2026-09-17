//! Original native facts and actual owners; no new native activation route.
use super::span_install::quoted;
use super::*;
use crate::composition::mlx::session::model_session::text_funding::FundedWorkOwner;
use eredu_core::{ControlledTextGeneration, TextPreparationOptions};

// Deferred model/native owners can retire while the control alias remains live.
// Check the actual protected lower bound on every cleanup pass, not a snapshot
// of unrelated bytes taken before those owners had a chance to retire.
fn settle_with_controls(pool: &WorkingMemoryPool, protected_and_source: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        assert!(pool.used_bytes().unwrap() >= protected_and_source);
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        disk::reclaim();
        assert!(pool.used_bytes().unwrap() >= protected_and_source);
        pool.unquoted_owner_count().unwrap() == 0
    });
}

#[test]
fn named_native_controls_are_bound_once_to_the_actual_candidate_on_all_routes() {
    let stream = stream();
    for route in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = load(&stream, &pool, route);
        let source = source(&runtime, 0);
        let ids = vec![2, 5, 7];
        let controller = disk::Controller::default();
        let baseline = pool.used_bytes().unwrap();
        let capture = CaptureAdmission::new(runtime.session(), geometry(), &source).unwrap();
        let quote = quoted(&runtime, &source, &ids, &controller);
        let controls = quote.span_workspace().text_controls().unwrap();
        assert!(controls.plan().same_plan(quote.span_workspace().plan()));
        assert_eq!(controls.source_identity(), Some(source.storage_identity()));
        assert_eq!(controls.geometry(), quote.geometry());
        let base_facts = capture.control_facts(quote.geometry()).unwrap();
        let pair_facts = super::super::super::preparation::facts().unwrap();
        let expected = PreparedTextControlWorkspace::prepare(
            &source,
            quote.geometry(),
            quote.span_workspace().plan(),
            base_facts,
        )
        .unwrap()
        .with_preparation_scopes(pair_facts)
        .unwrap();
        let pair_delta =
            expected.facts().admission_bytes().unwrap() - base_facts.admission_bytes().unwrap();
        let prediction_facts = super::super::super::prediction::facts().unwrap();
        let expected = expected.with_prediction_scopes(prediction_facts).unwrap();
        let prediction_delta = expected.facts().admission_bytes().unwrap()
            - base_facts.admission_bytes().unwrap()
            - pair_delta;
        assert_eq!(controls.facts(), expected.facts());
        assert!(
            prediction_delta
                > quote.geometry().max_output_tokens
                    * prediction_facts.total_bytes().unwrap().unwrap(),
            "original prediction enrichment includes its exact bank and issue controls"
        );
        assert!(
            pair_delta > pair_facts.total_bytes().unwrap().unwrap(),
            "original pair enrichment also includes its exact neutral bank controls"
        );
        let q = controls.facts().total_bytes().unwrap().unwrap();
        let publication = controls.capture_publication_control_bytes();
        let p = quote.span_workspace().retention_peak_bytes().unwrap();
        assert!(q > 0 && p > 0);
        assert_eq!(
            controls.facts().admission_bytes(),
            Some(CaptureAdmission::control_peak_bytes().unwrap() + pair_delta + prediction_delta)
        );
        assert_eq!(controls.facts().work_bytes(), Some(crate::composition::mlx::session::model_session::text_funding::text_work_control_bytes(4).unwrap()));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        let required = quote.incremental_bytes();
        drop(quote);
        drop(capture);
        let short = admit(
            &runtime,
            &source,
            &ids,
            &controller,
            baseline + required - 1,
        )
        .unwrap_err();
        assert!(matches!(
            cause::<WorkingMemoryError>(&short),
            WorkingMemoryError::BudgetExceeded { .. }
        ));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        let (preparation, quote) =
            admit(&runtime, &source, &ids, &controller, baseline + required).unwrap();
        let installed = quote
            .take_capture_installation(runtime.session(), &source)
            .unwrap();
        assert_eq!(
            installed.span_workspace().protected_host_bytes(),
            p + q + publication
        );
        assert_eq!(
            preparation.request().memory_reservation().unwrap().bytes(),
            required
        );
        drop((installed, quote, preparation, source));
        finish_runtime(runtime, &stream);
        settle_terminal(&pool, 0);
    }
}

#[test]
fn historical_quote_retains_controls_after_pending_installation_and_run_retire() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let c = source.capacity_bytes().unwrap();
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &vec![2, 5, 7],
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let protected = installed.span_workspace().protected_host_bytes();
    let run = quote.take_funding_run().unwrap();
    drop(installed);
    drop(run);
    drop(preparation);
    // The pending slot is empty and no plan payload is retained by this quote.
    assert!(quote.capture.as_ref().unwrap().pending.borrow().is_none());
    finish_runtime(runtime, &stream);
    let held = pool.used_bytes().unwrap();
    assert!(held >= protected + c);
    let alias = quote.clone();
    drop(quote);
    settle_with_controls(&pool, protected + c);
    drop(alias);
    // Native work and the run are closed. The source retains only its
    // original protected controls plus the independently registered C.
    settle_terminal(&pool, protected + c);
    drop(source);
    settle_terminal(&pool, 0);
}

#[test]
fn original_work_keeps_aggregate_custody_after_certification_and_last_quote_drop() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let c = source.capacity_bytes().unwrap();
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &vec![2, 5, 7],
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let protected = installed.span_workspace().protected_host_bytes();
    // Same original factory as Prompt/Sampling, with no submitted native work.
    let work = quote.preparation_work().unwrap();
    work.publish(RetainedStorage::default()).unwrap();
    work.certify().unwrap();
    let alias = work.clone();
    drop((installed, quote, preparation));
    finish_runtime(runtime, &stream);
    let held = pool.used_bytes().unwrap();
    assert!(held >= protected + c);
    work.certify().unwrap(); // Repeated certification cannot release Q.
    drop(work);
    settle_with_controls(&pool, protected + c);
    drop(alias);
    // Native work and the run are closed. The source retains only its
    // original protected controls plus the independently registered C.
    settle_terminal(&pool, protected + c);
    drop(source);
    settle_terminal(&pool, 0);
}

#[test]
fn original_control_guard_rejects_another_account_without_touching_either_scope() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let ids = vec![2, 5, 7];
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &ids,
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let guard = installed.span_workspace().control_guard();
    let (other_preparation, other_quote) = admit(
        &runtime,
        &source,
        &ids,
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let scope = other_quote.funding_scope().unwrap();
    let before = (pool.used_bytes().unwrap(), path_instrumentation::snapshot());
    assert!(matches!(
        guard.validate_native_scope(&scope),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        guard.validate_reservation(other_preparation.request().memory_reservation().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        (pool.used_bytes().unwrap(), path_instrumentation::snapshot()),
        before
    );
    scope.certify().unwrap();
    drop((
        guard,
        installed,
        quote,
        preparation,
        other_quote,
        other_preparation,
        source,
    ));
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, 0);
}

#[test]
fn actual_controlled_inference_work_alias_keeps_controls_after_native_and_frame_retirement() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let c = source.capacity_bytes().unwrap();
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &vec![2, 5, 7],
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let protected = installed.span_workspace().protected_host_bytes();
    drop((installed, quote, preparation));
    let controller = disk::Controller::default();
    let mut run = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![2, 5, 7],
        config(u64::MAX),
        controller.clone(),
        TextPreparationOptions {
            interventions: None, capture: Some(source.clone()),
        },
    )
    .unwrap();
    let token = run.next().unwrap().unwrap();
    let work: FundedWorkOwner = token
        .output()
        .owner
        .funding
        .borrow()
        .as_ref()
        .unwrap()
        .clone();
    drop(token);
    drop(run.take_captured_delivery().unwrap());
    for _ in 1..4 {
        drop(run.next().unwrap().unwrap());
        drop(run.take_captured_delivery().unwrap());
    }
    assert!(run.next().is_none());
    assert_eq!(controller.0.get(), (4, 4));
    drop(run);
    // Core has settled the actual producing operation; this repeats only its
    // ordinary closed certification, and must not take the retained guard.
    work.certify().unwrap();
    finish_runtime(runtime, &stream);
    let held = pool.used_bytes().unwrap();
    assert!(held >= protected + c);
    let alias = work.clone();
    drop(work);
    settle_with_controls(&pool, protected + c);
    drop(alias);
    // Native work and the run are closed. The source retains only its
    // original protected controls plus the independently registered C.
    settle_terminal(&pool, protected + c);
    drop(source);
    settle_terminal(&pool, 0);
}

#[test]
fn closed_quote_and_work_aliases_keep_original_custody_through_unwind_without_source_alias() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let c = source.capacity_bytes().unwrap();
    // Unlike residual-only decoder pins, CaptureAdmission explicitly joins the
    // loaded path storage into the full original control witness.
    let paths_bytes = runtime
        .session()
        .payload
        .model
        .erased()
        .shared_observation_paths()
        .unwrap()
        .capacity_bytes()
        .unwrap();
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &vec![2, 5, 7],
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let protected = installed.span_workspace().protected_host_bytes();
    let work = quote.preparation_work().unwrap();
    work.publish(RetainedStorage::default()).unwrap();
    work.certify().unwrap();
    let quote_alias = quote.clone();
    assert!(quote.same_owner(&quote_alias));
    let work_alias = work.clone();
    let original_geometry = quote.request().geometry();
    let run = quote.take_funding_run().unwrap();
    drop((installed, run, preparation, source));
    finish_runtime(runtime, &stream);
    // Only the closed quote/work populations now retain this full original
    // control witness. No source-plan payload alias masks a missing guard.
    settle_terminal(&pool, protected + c + paths_bytes);
    let marker = std::sync::Arc::new(());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let marker = marker.clone();
        move || {
            let _owners = (quote, work);
            std::panic::panic_any(marker);
        }
    }))
    .unwrap_err();
    assert!(std::sync::Arc::ptr_eq(
        panic.downcast_ref::<std::sync::Arc<()>>().unwrap(),
        &marker
    ));
    assert_eq!(quote_alias.request().geometry(), original_geometry);
    assert!(
        quote_alias
            .capture
            .as_ref()
            .unwrap()
            .pending
            .borrow()
            .is_none()
    );
    assert!(work_alias.test_scope_is_retired());
    settle_terminal(&pool, protected + c + paths_bytes);
    drop(quote_alias);
    settle_terminal(&pool, protected + c + paths_bytes);
    drop(work_alias);
    settle_terminal(&pool, 0);
}

// The actual original admission supplies the same work/custody subsequently
// installed in the concrete submission owner. No fabricated byte guard enters.
fn original_submission_resources() -> (
    WorkingMemoryPool,
    crate::composition::mlx::session::model_session::SubmissionResourcesOwner,
    eredu_core::SessionAuthority,
    u64,
) {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime, 0);
    let c = source.capacity_bytes().unwrap();
    let paths = runtime
        .session()
        .payload
        .model
        .erased()
        .shared_observation_paths()
        .unwrap()
        .capacity_bytes()
        .unwrap();
    let (preparation, quote) = admit(
        &runtime,
        &source,
        &vec![2, 5, 7],
        &disk::Controller::default(),
        u64::MAX,
    )
    .unwrap();
    let installed = quote
        .take_capture_installation(runtime.session(), &source)
        .unwrap();
    let expected = installed.span_workspace().protected_host_bytes() + c + paths;
    let work = quote.preparation_work().unwrap();
    work.publish(RetainedStorage::default()).unwrap();
    work.certify().unwrap();
    let mut authority = eredu_core::SessionAuthority::new();
    let resources = crate::composition::mlx::session::model_session::SubmissionResources::new(
        authority.begin_submission().unwrap(),
        Rc::new(Cell::new(false)),
    );
    resources.funding.replace(Some(work));
    let run = quote.take_funding_run().unwrap();
    drop((installed, run, preparation, source, quote));
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, expected);
    (pool, resources, authority, expected)
}

#[test]
fn closed_submission_token_aliases_retain_original_controls_after_owner_unwind() {
    use eredu_core::TokenOutput as _;
    let (pool, resources, authority, expected) = original_submission_resources();
    let stream = stream();
    let mut recovery = resources.recovery().unwrap();
    let value = Array::from_slice(&[7_u32], &[1]);
    safemlx::transforms::eval([&value]).unwrap();
    recovery.seal();
    let status = recovery.finish();
    assert!(status.settled && !status.failed && !status.blocked);
    resources.request_release();
    assert!(authority.require_idle().is_ok());
    let token =
        crate::composition::mlx::session::MlxTextToken::new(value, stream, resources.clone());
    let token_alias = token.clone();
    let owner_alias = resources.clone();
    let marker = std::sync::Arc::new(());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let marker = marker.clone();
        move || {
            let _owners = (resources, token);
            std::panic::panic_any(marker);
        }
    }))
    .unwrap_err();
    assert!(std::sync::Arc::ptr_eq(
        panic.downcast_ref::<std::sync::Arc<()>>().unwrap(),
        &marker
    ));
    assert_eq!(token_alias.token_id().unwrap(), 7);
    assert!(owner_alias.resources_releasable());
    assert!(
        owner_alias
            .funding
            .borrow()
            .as_ref()
            .unwrap()
            .test_scope_is_retired()
    );
    settle_terminal(&pool, expected);
    drop(owner_alias);
    settle_terminal(&pool, expected);
    drop(token_alias);
    settle_terminal(&pool, 0);
}

#[test]
fn closed_submission_original_custody_survives_unwind_in_real_descendant_quarantine() {
    let (pool, resources, authority, expected) = original_submission_resources();
    let mut recovery = resources.recovery().unwrap();
    // A real live descendant prevents parent lifetime retirement even without
    // pending records. No record observation is used as completion evidence.
    let child = safemlx::SubmissionScope::begin().unwrap();
    recovery.seal();
    assert!(!recovery.progress().settled);
    resources.request_release();
    assert!(authority.require_idle().is_err());
    let marker = std::sync::Arc::new(());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let marker = marker.clone();
        move || {
            let _owners = (resources, recovery);
            std::panic::panic_any(marker);
        }
    }))
    .unwrap_err();
    assert!(std::sync::Arc::ptr_eq(
        panic.downcast_ref::<std::sync::Arc<()>>().unwrap(),
        &marker
    ));
    assert!(authority.require_idle().is_err());
    assert_eq!(pool.used_bytes().unwrap(), expected);
    crate::backend::submission_recovery::reap();
    assert!(authority.require_idle().is_err());
    assert_eq!(pool.used_bytes().unwrap(), expected);
    drop(child);
    crate::backend::submission_recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    settle_terminal(&pool, 0);
}

#[path = "controls/recovery_nodes.rs"]
mod recovery_nodes;
