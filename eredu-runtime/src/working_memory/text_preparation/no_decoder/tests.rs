use super::*;
use crate::working_memory::{InferenceExecutionIdentity, MemoryLedger};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig, StateMemoryLayout,
    TextGenerationConfig, WorkspaceBound, cache::LayerCachePolicy,
};

fn fresh(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> Result<(InferenceTextPreparation, WorkingMemoryFundingRun), WorkingMemoryError> {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |n| WorkspaceBound::bounded(n, "explicit neutral native-scope accounting fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let admission = crate::working_memory::memory_fixture::attribute_host_admission(
        pool,
        Admission {
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
            requested_positions: 1,
            state,
            incremental_required_bytes: Some(bytes),
        },
    );
    let quoted = pool
        .reservation_requirements(&admission, None)?
        .get(pool.topology().host_domain())?
        .total()?;
    let snapshot = pool.snapshot()?;
    let host = snapshot
        .domains
        .iter()
        .find(|domain| domain.domain == pool.topology().host_domain())
        .unwrap();
    // The fixture selects a payload ceiling; the actual physical limit also
    // includes its explicitly quoted registry and reservation construction.
    let existing_controls =
        host.current_charge_bytes - host.fixed_baseline.total()? - pool.payload_used_bytes()?;
    let physical_payload_limit = capacity
        .checked_add(existing_controls)
        .and_then(|n| n.checked_add(quoted - bytes))
        .ok_or(WorkingMemoryError::Overflow)?;
    let reservation = pool.reserve_with_capacity(
        &execution,
        &admission,
        crate::working_memory::memory_fixture::resolved_host_limits(pool, physical_payload_limit),
    )?;
    let (reservation, run) = reservation.into_funding()?;
    // The low-level preparation contract accepts zero outputs; no application
    // generation configuration or forward loop is being constructed here.
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: false,
        temperature: 0.0,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(0),
    });
    let preparation =
        InferenceRequest::from(reservation).prepare_text(&execution, geometry, config)?;
    Ok((preparation, run))
}

fn inventory(pool: &MemoryLedger, bytes: u64) -> WorkingMemoryStorage<u32> {
    pool.register_host_storage([(17, bytes)]).unwrap()
}
fn counters(pool: &MemoryLedger) -> (u64, u64, u64, usize, usize, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.reserved,
        usage.registered,
        usage.peak,
        usage.funding.len(),
        usage.funding.values().map(|a| a.scopes).sum(),
        usage
            .funding
            .values()
            .map(|a| a.host_held - a.control_floor)
            .sum(),
    )
}

#[test]
fn no_decoder_exact_reservation_opens_one_scope_without_host_charge_and_keeps_source_pin() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let source_payload = Box::new([9u8; 32]);
    let source = inventory(&pool, source_payload.len() as u64);
    let before = counters(&pool);
    assert!(matches!(
        fresh(&pool, 16, 47),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(counters(&pool), before);
    let (preparation, run) = fresh(&pool, 16, 48).unwrap();
    let before = counters(&pool);
    let (completion, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_without_decoder(&run, source.clone())
        .unwrap();
    let after = counters(&pool);
    assert_eq!(
        (after.0, after.1, after.2, after.3, after.5),
        (before.0, before.1, before.2, before.3, 0)
    );
    assert_eq!(after.4, before.4 + 1);
    assert!(matches!(
        preparation.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    assert!(matches!(
        preparation.bind_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    completion.finish().unwrap();
    preparation.bind_prompt().unwrap();
    drop((source_payload, source, preparation, run));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        48,
        "scope keeps reservation and complete-source accounting"
    );
    native.certify().unwrap(); // this neutral fixture submits no native work
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn no_decoder_wrong_request_foreign_source_and_sampling_stage_reject_without_scope() {
    for case in 0..3 {
        let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
        let foreign = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
        let (preparation, run) = fresh(&pool, 16, 1 << 20).unwrap();
        let (other_preparation, other_run) = fresh(&pool, 16, 1 << 20).unwrap();
        let source = inventory(if case == 1 { &foreign } else { &pool }, 8);
        let before = (counters(&pool), counters(&foreign));
        let stage = if case == 2 {
            preparation
                .claim_sampling(preparation.authority().config.clone())
                .unwrap()
        } else {
            preparation.claim_prompt().unwrap()
        };
        let error = stage
            .construct_without_decoder(if case == 0 { &other_run } else { &run }, source.clone())
            .unwrap_err();
        assert_eq!(
            error,
            if case == 2 {
                WorkingMemoryError::InvocationPhaseMismatch
            } else {
                WorkingMemoryError::IdentityMismatch
            }
        );
        assert_eq!((counters(&pool), counters(&foreign)), before);
        let no_work = run.scope().unwrap();
        no_work.certify().unwrap();
        drop((source, preparation, run, other_preparation, other_run));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn no_decoder_rechecks_every_source_origin_including_zero_byte_registration() {
    for bytes in [0, 8] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
        let (source_preparation, source_run) = fresh(&pool, bytes, 1 << 20).unwrap();
        let source_scope = source_run.scope().unwrap();
        let mut registered = source_scope
            .publish_host_storage_fixture([(31u32, bytes)])
            .unwrap();
        let source = registered.remove(&31).unwrap();
        let cold_pin = pool.pin_registered_storage([(31u32, bytes)]).unwrap();
        let (preparation, run) = fresh(&pool, 16, 1 << 20).unwrap();
        drop(source_scope); // original registered origin becomes quarantined
        let before = counters(&pool);
        let error = preparation
            .claim_prompt()
            .unwrap()
            .construct_without_decoder(&run, cold_pin.clone())
            .unwrap_err();
        assert_eq!(error, WorkingMemoryError::ExecutionFenced);
        assert_eq!(counters(&pool), before);
        run.scope().unwrap().certify().unwrap(); // destination remains healthy
        drop((preparation, run, source, source_preparation, source_run));
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            bytes,
            "original quarantine is not refunded"
        );
    }

    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let source = inventory(&pool, 8);
    let (preparation, run) = fresh(&pool, 16, 1 << 20).unwrap();
    drop(run.scope().unwrap()); // this exact destination account is fenced
    let before = counters(&pool);
    let error = preparation
        .claim_prompt()
        .unwrap()
        .construct_without_decoder(&run, source.clone())
        .unwrap_err();
    assert_eq!(error, WorkingMemoryError::ExecutionFenced);
    assert_eq!(counters(&pool), before);
    drop((source, preparation, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 16);
}

#[test]
fn no_decoder_cancellation_does_not_reissue_prompt_or_certify_native_work() {
    for unresolved in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
        let source = inventory(&pool, 8);
        let (preparation, run) = fresh(&pool, 16, 1 << 20).unwrap();
        let (completion, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_without_decoder(&run, source.clone())
            .unwrap();
        drop(completion);
        assert!(matches!(
            preparation.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        assert!(matches!(
            preparation.claim_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        drop((source, preparation, run));
        assert_eq!(pool.payload_used_bytes().unwrap(), 24);
        if unresolved {
            drop(native);
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                24,
                "unresolved scope pins source in quarantine"
            );
        } else {
            native.certify().unwrap(); // this branch proves no work occurred
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn no_decoder_empty_source_is_real_absence_and_scope_overflow_is_atomic() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let source = pool
        .pin_registered_storage(std::iter::empty::<(u32, u64)>())
        .unwrap();
    let (preparation, run) = fresh(&pool, 0, 0).unwrap();
    {
        let mut usage = pool.0.usage.lock().unwrap();
        usage.funding.values_mut().next().unwrap().scopes = usize::MAX;
    }
    let before = counters(&pool);
    assert_eq!(
        preparation
            .claim_prompt()
            .unwrap()
            .construct_without_decoder(&run, source.clone())
            .unwrap_err(),
        WorkingMemoryError::Overflow
    );
    assert_eq!(counters(&pool), before);
    pool.0
        .usage
        .lock()
        .unwrap()
        .funding
        .values_mut()
        .next()
        .unwrap()
        .scopes = 0;
    drop((preparation, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let (preparation, run) = fresh(&pool, 0, 0).unwrap();
    let before_native = counters(&pool);
    let (completion, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_without_decoder(&run, source)
        .unwrap();
    let mut expected = before_native;
    expected.4 += 1;
    assert_eq!(counters(&pool), expected);
    completion.finish().unwrap();
    native.certify().unwrap();
    drop((preparation, run));
    let retired = counters(&pool);
    assert_eq!(
        (retired.0, retired.1, retired.3, retired.4, retired.5),
        (0, 0, 0, 0, 0)
    );
}

#[test]
fn preparation_scope_identity_rejects_an_equal_account_with_a_different_preparing_arc() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let (original, run) = fresh(&pool, 1024, 4096).unwrap();
    let mut forged = original.clone();
    let original_authority = original.request.preparation.as_ref().unwrap();
    forged.request.preparation = Some(std::sync::Arc::new(
        super::super::TextPreparationAuthority {
            config: original_authority.config.clone(),
            state: crate::working_memory::control_mutex::ControlMutex::new(
                super::super::TextPreparationState::default(),
            ),
        },
    ));
    original
        .request
        .validate_same_request(&forged.request)
        .unwrap();
    assert!(matches!(
        forged.validate_scope_preparation(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        original.validate_scope_preparation(),
        Err(WorkingMemoryError::TextRunUnbound)
    ));
    // Rejection does not replace the original preparing Arc or consume a role.
    let prompt = original.claim_prompt().unwrap();
    drop((prompt, forged, original, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
