use super::*;
use crate::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryFundingRun, WorkingMemoryPool,
    WorkingMemoryReservation,
};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, SharedStorageDomain, StateMemoryLayout,
    TextControllerStorage, TokenFilter, WorkspaceBound,
};
use std::{
    cell::Cell,
    convert::Infallible,
    num::NonZeroU8,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

mod mixed_sources;

struct Controller {
    shared: Vec<SharedTokenFilter>,
    additional: u64,
    known: bool,
    decisions: Cell<usize>,
}
impl Controller {
    fn new(shared: Vec<SharedTokenFilter>, additional: u64) -> Self {
        Self {
            shared,
            additional,
            known: true,
            decisions: Cell::new(0),
        }
    }
}
impl TokenFilterController for Controller {
    type Error = Infallible;
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: self
                .shared
                .first()
                .map_or(&TokenFilter::All, SharedTokenFilter::as_ref)
                .into(),
            additional_host_bytes: self.additional,
        })
    }
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        if self.known {
            TextControllerStorage::RunOwnedWithSharedFilters(&self.shared)
        } else {
            TextControllerStorage::Unknown
        }
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.decisions.set(self.decisions.get() + 1);
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn mask(capacity: usize) -> SharedTokenFilter {
    SharedTokenFilter::new(token_filter(capacity))
}

fn token_filter(capacity: usize) -> TokenFilter {
    let mut bits = Vec::with_capacity(capacity);
    bits.extend_from_slice(&[true, false, true]);
    assert_eq!(bits.capacity(), capacity);
    TokenFilter::allowed(bits).unwrap()
}

// A stateless portable execution with an explicit enclosing host allowance.
// No native execution or allocation bound is inferred by these ledger tests.
fn funding(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    pool.reserve(&InferenceExecutionIdentity::default(), &admission(bytes))
        .unwrap()
        .into_funding()
        .unwrap()
}

fn admission(bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = |n| {
        WorkspaceBound::bounded(
            n,
            "portable controller fixture: only enclosing host storage",
        )
    };
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        1,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(0),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(bytes),
    })
    .unwrap();
    Admission {
        requested_positions: 2,
        state,
        incremental_required_bytes: bytes,
        available_memory_bytes: None,
    }
}

fn balances(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (usage.reserved, usage.registered, usage.peak)
}

#[test]
fn exact_spare_capacity_is_priced_once_beyond_the_emitted_filter() {
    let source = mask(37);
    let mut controller = Controller::new(vec![source.clone(), source.clone()], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    assert_eq!(source.allowed_mask().unwrap().len(), 3);
    assert_eq!(source.capacity_bytes(), Some(37));
    contract
        .validate_workspace(controller.inference_workspace(2).unwrap())
        .unwrap();
    controller.additional = 36;
    assert!(matches!(
        contract.validate_workspace(controller.inference_workspace(2).unwrap()),
        Err(ControllerStorageError::UnpricedSharedStorage {
            required_bytes: 37,
            available_bytes: 36
        })
    ));
    controller.additional = 3;
    assert!(matches!(
        contract.validate_workspace(controller.inference_workspace(2).unwrap()),
        Err(ControllerStorageError::UnpricedSharedStorage {
            required_bytes: 37,
            available_bytes: 3
        })
    ));
    assert_eq!(controller.decisions.get(), 0);
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    controller.additional = 37;
    contract
        .validate_workspace(controller.inference_workspace(2).unwrap())
        .unwrap();
    contract.adopt(&controller, &scope).unwrap();
    assert_eq!(balances(&pool), (91, 37, 128));
    scope.certify().unwrap();
    drop((run, controller));
    assert_eq!(pool.used_bytes().unwrap(), 37);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(metadata.bytes(), 128);
}

#[test]
fn repeated_full_runs_share_one_charge_and_preexisting_alias_outlives_both() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let source = mask(37);
    let escaped_before_attachment = source.clone();
    let controller = Controller::new(vec![source], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let metadata_contract = contract.clone();
    let (first_metadata, first) = funding(&pool, 128);
    let scope = first.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    scope.certify().unwrap();
    drop(first);
    assert_eq!(balances(&pool), (0, 37, 128));

    let (second_metadata, second) = funding(&pool, 128);
    assert_eq!(balances(&pool), (128, 37, 165));
    let scope = second.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    contract.adopt(&controller, &scope).unwrap();
    assert_eq!(balances(&pool), (128, 37, 165));
    scope.certify().unwrap();
    drop((second, controller, contract));
    assert_eq!(balances(&pool), (0, 37, 165));
    drop(escaped_before_attachment);
    // Neither historical request nor contract metadata retains the mask charge.
    assert_eq!(balances(&pool), (0, 0, 165));
    assert_eq!(first_metadata.bytes(), 128);
    assert_eq!(second_metadata.bytes(), 128);
    drop((metadata_contract, first_metadata, second_metadata));
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn independent_pool_domains_each_retain_the_shared_physical_payload() {
    let first_pool = WorkingMemoryPool::new(512, 0).unwrap();
    let second_pool = WorkingMemoryPool::new(512, 0).unwrap();
    let source = mask(37);
    let controller = Controller::new(vec![source.clone()], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    for pool in [&first_pool, &second_pool] {
        let (metadata, run) = funding(pool, 128);
        let scope = run.scope().unwrap();
        contract.adopt(&controller, &scope).unwrap();
        assert_eq!(balances(pool), (91, 37, 128));
        scope.certify().unwrap();
        drop((run, metadata));
        assert_eq!(pool.used_bytes().unwrap(), 37);
    }
    drop(controller);
    assert_eq!(first_pool.used_bytes().unwrap(), 37);
    assert_eq!(second_pool.used_bytes().unwrap(), 37);
    drop(source);
    assert_eq!(first_pool.used_bytes().unwrap(), 0);
    assert_eq!(second_pool.used_bytes().unwrap(), 0);
    drop(contract);
}

#[test]
fn unknown_lifetime_rejects_without_callbacks_or_accounting_changes() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let mut controller = Controller::new(vec![mask(37)], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    controller.known = false;
    assert!(matches!(
        ControllerStorageContract::inspect(&controller),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert!(matches!(
        contract.validate(&controller),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(balances(&pool), (0, 0, 0));
    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    assert!(matches!(
        contract.adopt(&controller, &scope),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(balances(&pool), (128, 0, 128));
    assert_eq!(controller.decisions.get(), 0);
    // No acquisition occurred, and all fixture payloads retire before scope certification.
    drop(controller);
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn equal_sized_replacements_and_erased_decision_provenance_cannot_substitute_storage() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let original = mask(37);
    let replacement = mask(37);
    assert_eq!(original, replacement);
    let mut controller = Controller::new(vec![original.clone()], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    contract.validate(&controller).unwrap();
    let good =
        TokenSamplingDecision::new(TokenFilter::All).with_shared_tokenizer_validity(&original);
    contract.validate_decision(&good).unwrap();
    let alias = original.clone();
    contract
        .validate_decision(
            &TokenSamplingDecision::new(TokenFilter::All).with_shared_tokenizer_validity(&alias),
        )
        .unwrap();
    let mut foreign =
        TokenSamplingDecision::new(TokenFilter::All).with_shared_tokenizer_validity(&replacement);
    foreign.override_filter(TokenFilter::allowed(vec![true, false, false]).unwrap());
    assert!(matches!(
        contract.validate_decision(&foreign),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let erased = TokenSamplingDecision::new(TokenFilter::All)
        .with_shared_tokenizer_validity(&original)
        .with_tokenizer_validity(original.as_ref());
    assert!(matches!(
        contract.validate_decision(&erased),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert!(matches!(
        contract.validate_decision(
            &TokenSamplingDecision::new(TokenFilter::All)
                .with_tokenizer_validity(replacement.as_ref())
        ),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    contract
        .validate_decision(&TokenSamplingDecision::new(TokenFilter::All))
        .unwrap();
    controller.shared[0] = replacement.clone();
    assert!(matches!(
        contract.validate(&controller),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    assert!(matches!(
        contract.adopt(&controller, &scope),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(balances(&pool), (128, 0, 128));
    assert_eq!(controller.decisions.get(), 0);
    drop((good, foreign, erased));
    drop((controller, original, replacement, alias));
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn completed_attachment_survives_later_preparation_rejection_without_retaining_unused_funding() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let escaped = mask(53);
    let controller = Controller::new(vec![escaped.clone()], 53);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let (metadata, run) = funding(&pool, 192);
    let scope = run.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    assert_eq!(balances(&pool), (139, 53, 192));
    // Simulate a later preparation agreement rejection before other work starts.
    // Complete shared publication permits safe certification after local payload retirement.
    drop(controller);
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(balances(&pool), (0, 53, 192));
    assert!(contract.shared_bytes > 0);
    drop(escaped);
    assert_eq!(balances(&pool), (0, 0, 192));
    drop(contract);
}

#[test]
fn partial_attachment_and_uncertified_retirement_quarantine_the_remaining_envelope() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let mut sources = vec![mask(37), mask(53)];
    sources.sort_by(|a, b| a.identity().cmp(b.identity()));
    let first_bytes = sources[0].capacity_bytes().unwrap();
    let poison_domain = SharedStorageDomain::default();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ =
                sources[1].try_attach::<Infallible>(&poison_domain, || panic!("provider failed"));
        }))
        .is_err()
    );
    let controller = Controller::new(sources, 90);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    contract
        .validate_workspace(controller.inference_workspace(1).unwrap())
        .unwrap();
    let (metadata, run) = funding(&pool, 192);
    let scope = run.scope().unwrap();
    assert!(matches!(
        contract.adopt(&controller, &scope),
        Err(ControllerStorageError::Attachment(
            SharedStorageAttachmentError::Poisoned
        ))
    ));
    assert_eq!(balances(&pool), (192 - first_bytes, first_bytes, 192));
    drop((scope, run, metadata));
    assert_eq!(balances(&pool), (192 - first_bytes, first_bytes, 192));
    drop(controller);
    // Retiring the partial publication returns credit to quarantine, never refunds it.
    assert_eq!(balances(&pool), (192, 0, 192));
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(contract);
}

#[test]
fn final_filter_retirement_allows_accounting_reentry() {
    struct Reenter {
        pool: WorkingMemoryPool,
        called: Arc<AtomicBool>,
    }
    impl Drop for Reenter {
        fn drop(&mut self) {
            assert!(self.pool.0.usage.try_lock().is_ok());
            let extra = self.pool.register_storage([(719_u32, 11)]).unwrap();
            assert!(self.pool.used_bytes().unwrap() >= 11);
            drop(extra);
            self.called.store(true, Ordering::SeqCst);
        }
    }
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let source = mask(37);
    let called = Arc::new(AtomicBool::new(false));
    source
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Reenter {
                pool: pool.clone(),
                called: called.clone(),
            }))
        })
        .unwrap();
    let controller = Controller::new(vec![source], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(pool.used_bytes().unwrap(), 37);
    drop(controller);
    assert!(called.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(contract);
}

#[test]
fn source_construction_rejects_live_reservations_before_invoking_factory() {
    for bytes in [0, 128] {
        let pool = WorkingMemoryPool::new(512, 0).unwrap();
        let reservation = pool
            .reserve(&InferenceExecutionIdentity::default(), &admission(bytes))
            .unwrap();
        let calls = Cell::new(0);
        let result = pool.prepare_shared_token_filter(|| {
            calls.set(calls.get() + 1);
            token_filter(37)
        });
        assert!(matches!(
            result,
            Err(ControllerStorageError::Storage(
                WorkingMemoryError::ReservedWorkActive
            ))
        ));
        assert_eq!(calls.get(), 0);
        assert_eq!(balances(&pool), (bytes, 0, bytes));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        drop(reservation);

        let source = pool
            .prepare_shared_token_filter(|| {
                calls.set(calls.get() + 1);
                assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
                assert!(matches!(
                    pool.reserve(&InferenceExecutionIdentity::default(), &admission(0)),
                    Err(WorkingMemoryError::UnknownBound)
                ));
                token_filter(37)
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(source.capacity_bytes(), Some(37));
        assert_eq!(source.allowed_mask(), Some(&[true, false, true][..]));
        assert_eq!(balances(&pool), (0, 37, bytes.max(37)));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        let alias = source.clone();
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), 37);
        drop(alias);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn never_admitted_sources_compete_with_the_next_request_for_domain_capacity() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let first = pool
        .prepare_shared_token_filter(|| token_filter(37))
        .unwrap();
    let second = pool
        .prepare_shared_token_filter(|| token_filter(53))
        .unwrap();
    let second_alias = second.clone();
    assert_eq!(balances(&pool), (0, 90, 90));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &admission(11)),
        Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: 11,
            available_bytes: 10
        })
    ));
    assert_eq!(balances(&pool), (0, 90, 90));
    let exact = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(10))
        .unwrap();
    assert_eq!(balances(&pool), (10, 90, 100));
    let called = Cell::new(false);
    assert!(
        pool.prepare_shared_token_filter(|| {
            called.set(true);
            token_filter(3)
        })
        .is_err()
    );
    assert!(!called.get());
    drop(exact);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 90);
    drop(second_alias);
    assert_eq!(pool.used_bytes().unwrap(), 37);
    drop(first);
    assert_eq!(balances(&pool), (0, 0, 100));
}

#[test]
fn prepared_source_reuses_its_existing_charge_during_full_funded_admission() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let source = pool
        .prepare_shared_token_filter(|| token_filter(37))
        .unwrap();
    let controller = Controller::new(vec![source.clone()], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    contract
        .validate_workspace(controller.inference_workspace(1).unwrap())
        .unwrap();
    assert_eq!(balances(&pool), (0, 37, 37));
    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    assert_eq!(balances(&pool), (128, 37, 165));
    contract.adopt(&controller, &scope).unwrap();
    // The source was already attached to this exact domain; no funding credit
    // transfers and the complete ordinary request envelope is still retained.
    assert_eq!(balances(&pool), (128, 37, 165));
    scope.certify().unwrap();
    drop((run, controller));
    assert_eq!(balances(&pool), (0, 37, 165));
    drop(source);
    assert_eq!(balances(&pool), (0, 0, 165));
    assert_eq!(metadata.bytes(), 128);
    drop((metadata, contract));
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn source_registration_rejection_and_factory_unwind_release_temporary_authority() {
    let pool = WorkingMemoryPool::new(36, 0).unwrap();
    let calls = Cell::new(0);
    let error = pool
        .prepare_shared_token_filter(|| {
            calls.set(calls.get() + 1);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            token_filter(37)
        })
        .unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    let capacity = loop {
        if let Some(error) = cause.downcast_ref::<WorkingMemoryError>() {
            break error;
        }
        cause = cause
            .source()
            .expect("registration must preserve its typed rejection");
    };
    assert!(matches!(
        capacity,
        WorkingMemoryError::BudgetExceeded {
            required_bytes: 37,
            available_bytes: 36
        }
    ));
    assert_eq!(calls.get(), 1);
    assert_eq!(balances(&pool), (0, 0, 0));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(error);

    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = pool.prepare_shared_token_filter(|| {
                calls.set(calls.get() + 1);
                assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
                let _payload = token_filter(17);
                panic!("source producer failed after allocating its local mask");
            });
        }))
        .is_err()
    );
    assert_eq!(calls.get(), 2);
    assert_eq!(balances(&pool), (0, 0, 0));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    // Both failure paths left the pool usable, including zero-payload owners.
    let all = pool
        .prepare_shared_token_filter(|| TokenFilter::All)
        .unwrap();
    assert_eq!(all.capacity_bytes(), Some(0));
    assert_eq!(balances(&pool), (0, 0, 0));
    let exact = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission(36))
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 36);
    drop((exact, all));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn retained_original_controller_accepts_empty_inventory_without_adopting_shared_sources() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let mut controller = Controller::new(vec![], 0);
    let workspace = controller.inference_workspace(2).unwrap();
    let contract =
        ControllerStorageContract::inspect_original_retained(&controller, workspace, &pool, &InferenceExecutionIdentity::default(), 2)
            .unwrap();
    assert!(!contract.has_original_domain());
    contract.validate(&controller).unwrap();
    assert_eq!(balances(&pool), (0, 0, 0));

    controller.known = false;
    assert!(
        ControllerStorageContract::inspect_original_retained(
            &controller,
            controller.inference_workspace(2).unwrap(),
            &pool,
            &InferenceExecutionIdentity::default(),
            2,
        )
        .is_err()
    );
    controller.known = true;
    controller.shared.push(mask(37));
    controller.additional = 37;
    assert!(
        ControllerStorageContract::inspect_original_retained(
            &controller,
            controller.inference_workspace(2).unwrap(),
            &pool,
            &InferenceExecutionIdentity::default(),
            2,
        )
        .is_err()
    );
    assert!(contract.validate(&controller).is_err());
    assert_eq!(controller.decisions.get(), 0);
    assert_eq!(balances(&pool), (0, 0, 0));
}

mod declarations;
