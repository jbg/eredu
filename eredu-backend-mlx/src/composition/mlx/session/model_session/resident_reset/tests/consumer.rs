//! Actual generation consumes the one original table after explicitly ordinary
//! reset setup. This does not provide native readiness or a public reset gate.
use super::super::super::{text_funding::FundedWorkOwner, text_quote};
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::working_memory::OriginalResidentResetSource;

fn host_account(pool: &MemoryLedger) -> eredu_runtime::working_memory::MemoryDomainSnapshot {
    pool.snapshot()
        .unwrap()
        .domains
        .into_iter()
        .find(|domain| domain.domain == pool.topology().host_domain())
        .unwrap()
}

fn settle_current(pool: &MemoryLedger, expected: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.fixture_host_current().unwrap() == expected
            && pool.unquoted_owner_count().unwrap() == 0
    });
    assert_eq!(host_account(pool).current_charge_bytes, expected);
}

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
    pool: &MemoryLedger,
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
    pool: &MemoryLedger,
) -> OriginalResidentResetSource {
    runtime.synchronize().unwrap(); // ordinary external readiness preparation
    let scope = Scope::enter(runtime.session());
    runtime
        .reset_admitted(SessionResetLimits::new(crate::memory_fixture::limits(
            u64::MAX,
        )))
        .unwrap();
    scope.retired(1);
    source(runtime, pool)
}
fn generation(
    fixture: &text_quote::PreparedResidencyFixture,
    route: usize,
    controlled: bool,
    reset: bool,
) -> (Vec<u32>, Vec<(Vec<i32>, Vec<f32>)>) {
    let pool = fixture.pool.clone();
    let (mut runtime, _artifact, baseline) = fixture.load(route, Some(128));
    let original = reset.then(|| original(&mut runtime, &pool));
    let probe = original.as_ref().map(WorkProbe::new);
    let tokens = outputs(
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
            pool.fixture_host_charge().unwrap() > 0,
            "real source/Work aliases retain custody"
        );
    }
    drop((work, probe, original));
    settle(&pool, baseline);
    (ids, values)
}

#[test]
fn original_table_generation_publishes_and_certifies_with_full_kv_parity_all_weight_routes() {
    if !crate::tests::support::native_process::enter("physical reset generation") {
        return;
    }
    let fixture = text_quote::PreparedResidencyFixture::new();
    for route in 0..3 {
        let ordinary = generation(&fixture, route, false, false);
        for controlled in [false, true] {
            assert_eq!(generation(&fixture, route, controlled, true), ordinary);
        }
    }
}

#[test]
fn original_table_slot_is_in_the_first_control_only_quote_exact_and_one_short() {
    if !crate::tests::support::native_process::enter("physical reset quote") {
        return;
    }
    let fixture = text_quote::PreparedResidencyFixture::new();
    for route in 0..3 {
        let pool = fixture.pool.clone();
        let (mut runtime, _artifact, source_baseline) = fixture.load(route, Some(128));
        let source = original(&mut runtime, &pool);
        let baseline = pool.fixture_host_current().unwrap();
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
        let required = preparation
            .request()
            .memory_reservation()
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        assert!(quote.original_controls().is_some());
        assert!(quote
            .test_original_table()
            .unwrap()
            .metadata()
            .same_storage(source.metadata()));
        let accepted = pool.fixture_host_current().unwrap();
        let preparation_bytes = accepted.checked_sub(baseline).unwrap();
        assert!(
            preparation_bytes >= required,
            "the accepted preparation retains its source and planning accounts too"
        );
        drop((preparation, quote));
        settle_current(&pool, baseline);
        let before_state = runtime.session().payload.model.erased().state_snapshot();
        let short = text_quote::admit(
            &runtime,
            &input,
            disk::config(0.0, 1, accepted - 1),
            &controller,
        )
        .unwrap_err();
        assert!(matches!(
            disk::cause::<WorkingMemoryError>(&short),
            Some(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            before_state
        );
        drop(short);
        settle_current(&pool, baseline);
        let (preparation, quote) = text_quote::admit(
            &runtime,
            &input,
            disk::config(0.0, 1, accepted),
            &controller,
        )
        .unwrap();
        assert_eq!(
            preparation
                .request()
                .memory_reservation()
                .requirements()
                .get(crate::memory_fixture::topology().host_domain())
                .unwrap()
                .total()
                .unwrap(),
            required
        );
        assert_eq!(pool.fixture_host_current().unwrap(), accepted);
        assert!(quote
            .test_original_table()
            .unwrap()
            .metadata()
            .same_storage(source.metadata()));
        drop((preparation, quote, source, runtime));
        settle(&pool, source_baseline);
    }
}

#[test]
fn complete_inventory_refusal_precedes_adoption_and_same_slot_retries_without_double_charge() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    let expected = original(&mut runtime, &pool);
    let foreign_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
    let baseline = host_account(&pool);
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
        let refused = host_account(&pool);
        assert_eq!(refused.current_charge_bytes, baseline.current_charge_bytes);
        assert_eq!(
            refused.registered_storage_bytes,
            baseline.registered_storage_bytes
        );
        assert_eq!(
            refused.outstanding_reservation_bytes,
            baseline.outstanding_reservation_bytes
        );
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
    let published = host_account(&pool);
    assert_eq!(
        published.registered_storage_bytes - published.registry_metadata_bytes,
        baseline.registered_storage_bytes - baseline.registry_metadata_bytes,
        "publishing aliases must not charge the existing payload a second time"
    );
    assert!(
        published.current_charge_bytes <= baseline.current_charge_bytes,
        "publication uses the accepted allowance and may release overlapping metadata"
    );
    assert!(published.outstanding_reservation_bytes < baseline.outstanding_reservation_bytes);
    assert!(source(&runtime, &pool)
        .metadata()
        .same_storage(expected.metadata()));
    work.certify().unwrap();
    assert!(work.test_scope_is_retired());
    let (nonstate, decoder) = runtime
        .session()
        .payload
        .retained_idle_storage()
        .unwrap()
        .into_parts();
    let certified = host_account(&pool);
    assert!(work.publish_model(nonstate, decoder).unwrap().is_none());
    let repeated = host_account(&pool);
    assert_eq!(
        repeated.current_charge_bytes,
        certified.current_charge_bytes
    );
    assert_eq!(
        repeated.registered_storage_bytes,
        certified.registered_storage_bytes
    );
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
    settle_current(&pool, host_account(&pool).fixed_baseline.total().unwrap());
    settle_current(
        &foreign_pool,
        host_account(&foreign_pool).fixed_baseline.total().unwrap(),
    );
}
fn runtime_for_foreign(
    stream: &Stream,
    pool: &MemoryLedger,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    runtime(stream, pool, 0)
}

#[test]
fn retired_original_table_and_replacement_never_refresh_the_fixed_quote_slot() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
