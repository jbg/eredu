//! Actual generation consumes the one original table after explicitly ordinary
//! reset setup. This does not provide native readiness or a public reset gate.
use super::super::super::{text_funding::FundedWorkOwner, text_quote};
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use eredu_runtime::working_memory::OriginalResidentResetSource;

struct Recorded {
    key: eredu_runtime::HostMetadataKey,
    work: Vec<FundedWorkOwner>,
}
thread_local! { static WORK: RefCell<Option<Recorded>> = const { RefCell::new(None) }; }
struct WorkProbe;
impl WorkProbe {
    fn new(source: &OriginalResidentResetSource) -> Self {
        WORK.with_borrow_mut(|slot| {
            assert!(slot.is_none());
            *slot = Some(Recorded {
                key: source.metadata().identity().registry_key().clone(),
                work: Vec::new(),
            });
        });
        Self
    }
    fn take(&self) -> Vec<FundedWorkOwner> {
        WORK.with_borrow_mut(|slot| std::mem::take(&mut slot.as_mut().unwrap().work))
    }
}
impl Drop for WorkProbe {
    fn drop(&mut self) {
        let old = WORK.with_borrow_mut(Option::take);
        drop(old);
    }
}
pub(in crate::composition::mlx::session::model_session) fn record(
    work: &FundedWorkOwner,
    source: &OriginalResidentResetSource,
) {
    WORK.with_borrow_mut(|slot| {
        if let Some(slot) = slot {
            if &slot.key == source.metadata().identity().registry_key() {
                slot.work.push(work.clone());
            }
        }
    });
}
fn source(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    pool: &WorkingMemoryPool,
) -> OriginalResidentResetSource {
    pool.pin_original_reset_slots(
        runtime
            .session()
            .payload
            .model
            .erased()
            .resident_reset_source()
            .unwrap()
            .state()
            .resident_reset_layers()
            .metadata(),
    )
    .unwrap()
}
fn original(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    pool: &WorkingMemoryPool,
) -> OriginalResidentResetSource {
    runtime.synchronize().unwrap(); // ordinary external readiness preparation
    let scope = Scope::enter(runtime.session());
    runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap();
    scope.retired(1);
    source(runtime, pool)
}
fn generation(
    route: usize,
    controlled: bool,
    reset: bool,
) -> (Vec<u32>, Vec<(Vec<i32>, Vec<f32>)>) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, route);
    let original = reset.then(|| original(&mut runtime, &pool));
    let probe = original.as_ref().map(WorkProbe::new);
    let tokens = disk::outputs(
        &mut runtime,
        disk::tokens(),
        disk::config(0.0, 2, u64::MAX),
        controlled,
    );
    let ids = disk::token_ids(&tokens);
    runtime.synchronize().unwrap();
    let values = numeric(&runtime);
    assert_eq!(values.len(), 6);
    assert!(values
        .iter()
        .any(|(_, values)| values.iter().any(|v| *v != 0.0)));
    if let Some(original) = &original {
        assert!(
            source(&runtime, &pool)
                .metadata()
                .same_storage(original.metadata()),
            "normal generation preserves the authenticated table"
        );
    }
    drop(tokens);
    runtime.synchronize().unwrap();
    let work = probe.as_ref().map(WorkProbe::take).unwrap_or_default();
    if reset {
        assert_eq!(
            work.len(),
            4,
            "only the four actual model Work owners carry the slot"
        );
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            work.iter().all(|work| work.test_scope_is_retired())
        });
        assert!(
            work.iter().all(|work| work.test_is_published()),
            "nonstate and decoder publication both succeeded"
        );
        assert!(
            work.iter().all(|work| work.test_scope_is_retired()),
            "exact settlement and token retirement certified all model scopes"
        );
    }
    drop(runtime);
    reclaim();
    if reset {
        assert!(
            pool.used_bytes().unwrap() > 0,
            "real source/Work aliases retain custody"
        );
    }
    drop((work, probe, original));
    settle(&pool, 0);
    (ids, values)
}

#[test]
fn original_table_generation_publishes_and_certifies_with_full_kv_parity_all_weight_routes() {
    for route in 0..3 {
        let ordinary = generation(route, false, false);
        for controlled in [false, true] {
            assert_eq!(generation(route, controlled, true), ordinary);
        }
    }
}

#[test]
fn original_table_slot_is_in_the_first_control_only_quote_exact_and_one_short() {
    for route in 0..3 {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = runtime(&stream, &pool, route);
        let source = original(&mut runtime, &pool);
        let baseline = pool.used_bytes().unwrap();
        let ids = disk::tokens();
        let input = disk::evidence(&ids);
        let controller = disk::Controller::default();
        let (preparation, quote) = text_quote::admit(
            &runtime,
            &input,
            disk::config(0.0, 1, u64::MAX),
            &controller,
        )
        .unwrap();
        let required = preparation.request().memory_reservation().unwrap().bytes();
        assert!(quote.original_controls().is_some());
        assert!(quote
            .test_original_table()
            .unwrap()
            .metadata()
            .same_storage(source.metadata()));
        drop((preparation, quote));
        settle(&pool, baseline);
        let before_state = runtime.session().payload.model.erased().state_snapshot();
        let short = text_quote::admit(
            &runtime,
            &input,
            disk::config(0.0, 1, baseline + required - 1),
            &controller,
        )
        .unwrap_err();
        assert!(matches!(
            disk::cause::<WorkingMemoryError>(&short),
            Some(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before_state
        );
        drop(short);
        let (preparation, quote) = text_quote::admit(
            &runtime,
            &input,
            disk::config(0.0, 1, baseline + required),
            &controller,
        )
        .unwrap();
        assert_eq!(
            preparation.request().memory_reservation().unwrap().bytes(),
            required
        );
        assert_eq!(pool.used_bytes().unwrap(), baseline + required);
        assert!(quote
            .test_original_table()
            .unwrap()
            .metadata()
            .same_storage(source.metadata()));
        drop((preparation, quote, source, runtime));
        settle(&pool, 0);
    }
}

#[test]
fn complete_inventory_refusal_precedes_adoption_and_same_slot_retries_without_double_charge() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    let expected = original(&mut runtime, &pool);
    let foreign_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut foreign, _foreign_artifact) = runtime_for_foreign(&stream, &foreign_pool);
    let foreign_source = original(&mut foreign, &foreign_pool);
    let (mut other, _other_artifact) = runtime_for_foreign(&stream, &pool);
    let other_source = original(&mut other, &pool);
    let ids = disk::tokens();
    let input = disk::evidence(&ids);
    let controller = disk::Controller::default();
    // Cold accepted request; no fake execution authority and no native work.
    // Actual issued model Work is covered by the generation test above.
    let (preparation, quote) = text_quote::admit(
        &runtime,
        &input,
        disk::config(0.0, 2, u64::MAX),
        &controller,
    )
    .unwrap();
    let work = quote.funded_work(quote.funding_scope().unwrap()).unwrap();
    let baseline = pool.used_bytes().unwrap();
    for case in 0..6 {
        let (mut nonstate, mut decoder) = runtime
            .session()
            .payload
            .retained_idle_storage()
            .unwrap()
            .into_parts();
        match case {
            0 => nonstate = RetainedStorage::default(),
            1 => nonstate
                .include_slot_metadata(foreign_source.metadata().clone())
                .unwrap(),
            2 => decoder
                .include_slot_metadata(expected.metadata().clone())
                .unwrap(),
            3 => {
                nonstate = RetainedStorage::default();
                nonstate
                    .include_slot_metadata(foreign_source.metadata().clone())
                    .unwrap();
            }
            4 => nonstate
                .include_slot_metadata(other_source.metadata().clone())
                .unwrap(),
            5 => {
                nonstate = RetainedStorage::default();
                nonstate
                    .include_slot_metadata(other_source.metadata().clone())
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let error = work.publish_model(nonstate, decoder).unwrap_err();
        assert!(matches!(
            disk::cause::<WorkingMemoryError>(&error),
            Some(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        assert!(!work.test_is_published());
        assert!(!work.test_scope_is_retired());
        drop(error);
    }
    let (mut nonstate, decoder) = runtime
        .session()
        .payload
        .retained_idle_storage()
        .unwrap()
        .into_parts();
    // Canonical equal aliases occupy one inventory entry and no new table charge.
    nonstate
        .include_slot_metadata(expected.metadata().clone())
        .unwrap();
    nonstate
        .include_slot_metadata(expected.metadata().clone())
        .unwrap();
    let publication = work.publish_model(nonstate, decoder).unwrap().unwrap();
    runtime
        .session()
        .payload
        .nonstate_publication
        .replace(Some(publication));
    assert!(work.test_is_published());
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    work.certify().unwrap();
    assert!(work.test_scope_is_retired());
    let (nonstate, decoder) = runtime
        .session()
        .payload
        .retained_idle_storage()
        .unwrap()
        .into_parts();
    assert!(work.publish_model(nonstate, decoder).unwrap().is_none());
    let error = work
        .publish_model(RetainedStorage::default(), RetainedStorage::default())
        .unwrap_err();
    assert!(matches!(
        disk::cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    drop((
        error,
        work,
        preparation,
        quote,
        expected,
        foreign_source,
        other_source,
        runtime,
        foreign,
        other,
    ));
    settle(&pool, 0);
    settle(&foreign_pool, 0);
}
fn runtime_for_foreign(
    stream: &Stream,
    pool: &WorkingMemoryPool,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    runtime(stream, pool, 0)
}

#[test]
fn retired_original_table_and_replacement_never_refresh_the_fixed_quote_slot() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    let old = original(&mut runtime, &pool);
    let ids = disk::tokens();
    let input = disk::evidence(&ids);
    let controller = disk::Controller::default();
    let (preparation, quote) = text_quote::admit(
        &runtime,
        &input,
        disk::config(0.0, 2, u64::MAX),
        &controller,
    )
    .unwrap();
    // Discard the unbound request before a second real reset. Keep only the
    // old source token, which cannot make the displaced table live again.
    drop((preparation, quote));
    let new = original(&mut runtime, &pool);
    assert!(!new.metadata().same_storage(old.metadata()));
    assert!(matches!(
        pool.pin_original_reset_slots(old.metadata()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let (nonstate, _) = runtime
        .session()
        .payload
        .retained_idle_storage()
        .unwrap()
        .into_parts();
    let error = nonstate
        .validate_original_table(&pool, Some(&old))
        .unwrap_err();
    assert!(matches!(
        disk::cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    drop((error, nonstate, old, new, runtime));
    settle(&pool, 0);
}

#[test]
fn actual_checkpoint_rollback_preserves_original_table_and_every_kv_value() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    let expected = original(&mut runtime, &pool);
    warm(&mut runtime, &pool, false);
    let before = numeric(&runtime);
    let token = Array::try_from_slice(&[4_u32], &[1, 1]).unwrap();
    // Existing native checkpoint/forward/rollback fixture; all work and host
    // reads here are ordinary test observation, not additional quoted work.
    let (prior, advanced, restored, _, _, _, continuation) = runtime
        .session_mut()
        .payload
        .get_mut()
        .unwrap()
        .model
        .erased_mut()
        .checkpoint_restore_probe(&token, &stream)
        .unwrap();
    assert_ne!(prior, advanced);
    assert_eq!(prior, restored);
    assert!(continuation.iter().any(|v| *v != 0.0));
    assert_eq!(numeric(&runtime), before);
    assert!(source(&runtime, &pool)
        .metadata()
        .same_storage(expected.metadata()));
    drop((token, expected, runtime));
    settle(&pool, 0);
}
