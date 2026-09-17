use super::*;

struct MixedController {
    filters: Vec<SharedTokenFilter>,
    bytes: Vec<SharedControllerBytes>,
    additional: u64,
    known: bool,
}

impl MixedController {
    fn new(
        filters: Vec<SharedTokenFilter>,
        bytes: Vec<SharedControllerBytes>,
        additional: u64,
    ) -> Self {
        Self {
            filters,
            bytes,
            additional,
            known: true,
        }
    }
}

impl TokenFilterController for MixedController {
    type Error = Infallible;

    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: self.additional,
        })
    }

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        if self.known {
            TextControllerStorage::RunOwnedWithSharedStorage {
                filters: &self.filters,
                bytes: &self.bytes,
            }
        } else {
            TextControllerStorage::Unknown
        }
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("cold source accounting must not advance the controller")
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("cold source accounting must not commit a token")
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("cold source accounting must not query completion")
    }
}

fn buffer(capacity: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend([7, 11, 13]);
    assert_eq!(bytes.capacity(), capacity);
    bytes
}

fn source(capacity: usize) -> SharedControllerBytes {
    SharedControllerBytes::new(buffer(capacity))
}

fn mismatch(error: ControllerStorageError) {
    assert!(matches!(
        error,
        ControllerStorageError::Storage(WorkingMemoryError::IdentityMismatch)
    ));
}

#[test]
fn mixed_inventory_deduplicates_aliases_but_preserves_zero_and_equal_size_identities() {
    let filter = mask(37);
    let bytes = source(53);
    let empty = SharedControllerBytes::new(Vec::new());
    let mut controller = MixedController::new(
        vec![filter.clone(), filter],
        vec![bytes.clone(), empty.clone(), bytes],
        90,
    );
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    assert_eq!(contract.shared.len(), 3);
    assert_eq!(contract.shared_bytes, 90);
    assert_eq!(controller.bytes[0].as_ref(), &[7, 11, 13]);
    assert_eq!(controller.bytes[0].capacity_bytes(), Some(53));
    contract
        .validate_workspace(controller.inference_workspace(1).unwrap())
        .unwrap();
    controller.additional = 89;
    assert!(matches!(
        contract.validate_workspace(controller.inference_workspace(1).unwrap()),
        Err(ControllerStorageError::UnpricedSharedStorage {
            required_bytes: 90,
            available_bytes: 89,
        })
    ));
    controller.bytes[1] = SharedControllerBytes::new(Vec::new());
    mismatch(contract.validate(&controller).unwrap_err());
    controller.bytes[1] = empty;
    contract.validate(&controller).unwrap();
    // Equal contents and exact capacities cannot replace an admitted allocation.
    let replacement = source(53);
    assert_eq!(replacement.as_ref(), controller.bytes[0].as_ref());
    assert!(!replacement.same_storage(&controller.bytes[0]));
    controller.bytes[0] = replacement.clone();
    controller.bytes[2] = replacement;
    mismatch(contract.validate(&controller).unwrap_err());
    controller.known = false;
    assert!(matches!(
        ControllerStorageContract::inspect(&controller),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
}

#[test]
fn mixed_exact_capacity_adopts_each_backing_and_final_aliases_retire_independently() {
    let filter = mask(37);
    let bytes = source(53);
    let controller = MixedController::new(vec![filter.clone()], vec![bytes.clone()], 90);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let short = WorkingMemoryPool::new(89, 0).unwrap();
    assert!(matches!(
        short.reserve(&InferenceExecutionIdentity::default(), &admission(90)),
        Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: 90,
            available_bytes: 89
        })
    ));
    assert_eq!(balances(&short), (0, 0, 0));

    let pool = WorkingMemoryPool::new(90, 0).unwrap();
    let (metadata, run) = funding(&pool, 90);
    let scope = run.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    contract.adopt(&controller, &scope).unwrap();
    assert_eq!(balances(&pool), (0, 90, 90));
    scope.certify().unwrap();
    drop((run, controller));
    assert_eq!(pool.used_bytes().unwrap(), 90);
    drop(filter);
    assert_eq!(pool.used_bytes().unwrap(), 53);
    drop(bytes);
    // Contract, source identity and request diagnostics never keep physical custody.
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(metadata.bytes(), 90);
    assert_eq!(contract.shared_bytes, 90);
    drop((metadata, contract));
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn mixed_existing_source_credit_changes_only_the_fixed_addition_and_pin_lifetime() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let filter = pool
        .prepare_shared_token_filter(|| token_filter(37))
        .unwrap();
    let bytes = pool.prepare_shared_controller_bytes(|| buffer(53)).unwrap();
    let controller = MixedController::new(
        vec![filter.clone()],
        vec![bytes.clone(), bytes.clone()],
        128,
    );
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let registered = contract.pin_registered(&controller, &pool).unwrap();
    assert_eq!(registered.source_bytes(), 90);
    let outside = admission(0).state.execution_workspace.unwrap();
    let contribution = ControllerWorkspaceContribution::new(
        outside.geometry,
        controller.inference_workspace(1).unwrap(),
        3,
        &contract,
        &pool,
        Some(&registered),
    )
    .unwrap();
    let estimate = contribution.compose(outside.clone()).unwrap();
    assert_eq!(estimate.full().retained.bytes(), Some(128));
    assert_eq!(estimate.incremental().retained.bytes(), Some(38));
    assert_eq!(estimate.full().activations, outside.activations);
    assert_eq!(estimate.full().vocabulary, outside.vocabulary);
    assert_eq!(estimate.incremental().vocabulary, outside.vocabulary);
    assert_eq!(estimate.controller_contract().additional_host_bytes(), 128);
    assert_eq!(balances(&pool), (0, 90, 90));

    let (metadata, run) = funding(&pool, 128);
    let scope = run.scope().unwrap();
    contract.adopt(&controller, &scope).unwrap();
    // Loading attached to the same identity namespace: no second charge or transfer.
    assert_eq!(balances(&pool), (128, 90, 218));
    scope.certify().unwrap();
    drop((run, controller, filter, bytes));
    // The proof pins charges, never payload; its removal releases the remaining custody.
    assert_eq!(pool.used_bytes().unwrap(), 90);
    drop((registered, contribution, estimate));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(contract.shared_bytes, 90);
    drop((metadata, contract));
}

#[test]
fn mixed_pinning_is_existing_only_atomic_and_bound_to_domain_and_capacity() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let foreign = WorkingMemoryPool::new(512, 0).unwrap();
    let filter = pool
        .prepare_shared_token_filter(|| token_filter(37))
        .unwrap();
    let bytes = source(53);
    let controller = MixedController::new(vec![filter.clone()], vec![bytes.clone()], 90);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    mismatch(contract.pin_registered(&controller, &pool).unwrap_err());
    mismatch(contract.pin_registered(&controller, &foreign).unwrap_err());
    assert_eq!(balances(&pool), (0, 37, 37));
    let conflict = pool
        .register_storage([(bytes.identity().clone(), 54)])
        .unwrap();
    mismatch(contract.pin_registered(&controller, &pool).unwrap_err());
    assert_eq!(balances(&pool), (0, 91, 91));
    drop((controller, filter, conflict));
    // Rejected complete pins did not retain any earlier registered source.
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(bytes.capacity_bytes(), Some(53));
    drop((bytes, contract));
}

#[test]
fn byte_only_sources_do_not_impose_or_satisfy_shared_filter_provenance() {
    let bytes = source(37);
    let filter = mask(37);
    let mut controller = MixedController::new(vec![], vec![bytes], 37);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    {
        let raw = TokenSamplingDecision::new(TokenFilter::All)
            .with_tokenizer_validity(filter.as_ref())
            .with_controller_storage(controller.inference_storage());
        contract.validate_decision(&raw).unwrap();
        let shared = TokenSamplingDecision::new(TokenFilter::All)
            .with_shared_tokenizer_validity(&filter)
            .with_controller_storage(controller.inference_storage());
        mismatch(contract.validate_decision(&shared).unwrap_err());
    }

    controller.filters.push(filter.clone());
    controller.additional = 74;
    let mixed = ControllerStorageContract::inspect(&controller).unwrap();
    let shared = TokenSamplingDecision::new(TokenFilter::All)
        .with_shared_tokenizer_validity(&filter)
        .with_controller_storage(controller.inference_storage());
    let raw = TokenSamplingDecision::new(TokenFilter::All)
        .with_tokenizer_validity(filter.as_ref())
        .with_controller_storage(controller.inference_storage());
    mixed.validate_decision(&shared).unwrap();
    mismatch(mixed.validate_decision(&raw).unwrap_err());
    let replacement = mask(37);
    mismatch(
        mixed
            .validate_decision(
                &TokenSamplingDecision::new(TokenFilter::All)
                    .with_shared_tokenizer_validity(&replacement)
                    .with_controller_storage(controller.inference_storage()),
            )
            .unwrap_err(),
    );
}

#[test]
fn postdecision_witness_requires_complete_exact_sources_and_survives_forced_override() {
    let filter = mask(37);
    let bytes = source(53);
    let mut controller = MixedController::new(vec![filter], vec![bytes], 90);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    mismatch(
        contract
            .validate_decision(&TokenSamplingDecision::new(TokenFilter::All))
            .unwrap_err(),
    );
    assert!(matches!(
        contract.validate_decision(
            &TokenSamplingDecision::new(TokenFilter::All)
                .with_controller_storage(TextControllerStorage::Unknown)
        ),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    mismatch(
        contract
            .validate_decision(
                &TokenSamplingDecision::new(TokenFilter::All)
                    .with_controller_storage(TextControllerStorage::RunOwned),
            )
            .unwrap_err(),
    );
    {
        let mut decision = TokenSamplingDecision::new(TokenFilter::All)
            .with_controller_storage(controller.inference_storage());
        contract.validate_decision(&decision).unwrap();
        decision.override_filter(TokenFilter::allowed(vec![false, true, false]).unwrap());
        contract.validate_decision(&decision).unwrap();
    }
    // The controller passed preflight, then substituted a same-sized source in
    // its callback. The returned current inventory must reject at this step.
    contract.validate(&controller).unwrap();
    controller.bytes[0] = source(53);
    mismatch(
        contract
            .validate_decision(
                &TokenSamplingDecision::new(TokenFilter::All)
                    .with_controller_storage(controller.inference_storage()),
            )
            .unwrap_err(),
    );
    let same_total_other_kind = MixedController::new(vec![mask(90)], vec![], 90);
    mismatch(
        contract
            .validate_decision(
                &TokenSamplingDecision::new(TokenFilter::All)
                    .with_controller_storage(same_total_other_kind.inference_storage()),
            )
            .unwrap_err(),
    );
}

#[test]
fn zero_byte_owners_require_witnesses_while_legacy_masks_keep_optional_witness() {
    let empty = MixedController::new(vec![], vec![SharedControllerBytes::new(Vec::new())], 0);
    let contract = ControllerStorageContract::inspect(&empty).unwrap();
    mismatch(
        contract
            .validate_decision(&TokenSamplingDecision::new(TokenFilter::All))
            .unwrap_err(),
    );
    contract
        .validate_decision(
            &TokenSamplingDecision::new(TokenFilter::All)
                .with_controller_storage(empty.inference_storage()),
        )
        .unwrap();
    let different_empty =
        MixedController::new(vec![], vec![SharedControllerBytes::new(Vec::new())], 0);
    mismatch(
        contract
            .validate_decision(
                &TokenSamplingDecision::new(TokenFilter::All)
                    .with_controller_storage(different_empty.inference_storage()),
            )
            .unwrap_err(),
    );

    let legacy = Controller::new(vec![mask(37)], 37);
    let contract = ControllerStorageContract::inspect(&legacy).unwrap();
    contract
        .validate_decision(&TokenSamplingDecision::new(TokenFilter::All))
        .unwrap();
    contract
        .validate_decision(
            &TokenSamplingDecision::new(TokenFilter::All)
                .with_controller_storage(legacy.inference_storage()),
        )
        .unwrap();
    mismatch(
        contract
            .validate_decision(
                &TokenSamplingDecision::new(TokenFilter::All)
                    .with_controller_storage(TextControllerStorage::RunOwned),
            )
            .unwrap_err(),
    );
}

#[test]
fn mixed_preexisting_aliases_keep_separate_domain_charges_after_preparation_rejection() {
    let first = WorkingMemoryPool::new(512, 0).unwrap();
    let second = WorkingMemoryPool::new(512, 0).unwrap();
    let filter = mask(37);
    let bytes = source(53);
    let controller = MixedController::new(vec![filter.clone()], vec![bytes.clone()], 90);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let mut requests = Vec::new();
    for pool in [&first, &second] {
        let (metadata, run) = funding(pool, 192);
        let scope = run.scope().unwrap();
        contract.adopt(&controller, &scope).unwrap();
        assert_eq!(balances(pool), (102, 90, 192));
        scope.certify().unwrap();
        // No other work began; a later preparation rejection closes unused funding.
        drop(run);
        requests.push(metadata);
        assert_eq!(pool.used_bytes().unwrap(), 90);
    }
    drop(controller);
    drop(filter);
    assert_eq!(first.used_bytes().unwrap(), 53);
    assert_eq!(second.used_bytes().unwrap(), 53);
    drop(bytes);
    assert_eq!(first.used_bytes().unwrap(), 0);
    assert_eq!(second.used_bytes().unwrap(), 0);
    drop((requests, contract));
}

#[test]
fn mixed_partial_attachment_keeps_published_source_and_quarantines_remaining_funding() {
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let filter = mask(37);
    let bytes = source(53);
    let (poisoned, first_bytes) = if filter.identity() < bytes.identity() {
        (SharedControllerSource::Bytes(&bytes), 37)
    } else {
        (SharedControllerSource::Filter(&filter), 53)
    };
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = poisoned.try_attach::<Infallible>(&SharedStorageDomain::default(), || {
            panic!("later source acquisition failed")
        });
    }))
    .is_err());
    let controller = MixedController::new(vec![filter], vec![bytes], 90);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
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
    assert_eq!(balances(&pool), (192, 0, 192));
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn byte_loading_excludes_live_plain_and_funded_reservations_before_factory() {
    for funded in [false, true] {
        for amount in [0, 128] {
            let pool = WorkingMemoryPool::new(512, 0).unwrap();
            let reservation = pool
                .reserve(&InferenceExecutionIdentity::default(), &admission(amount))
                .unwrap();
            let (reservation, run) = if funded {
                let (metadata, run) = reservation.into_funding().unwrap();
                (metadata, Some(run))
            } else {
                (reservation, None)
            };
            let calls = Cell::new(0);
            assert!(matches!(
                pool.prepare_shared_controller_bytes(|| {
                    calls.set(calls.get() + 1);
                    buffer(53)
                }),
                Err(ControllerStorageError::Storage(
                    WorkingMemoryError::ReservedWorkActive
                ))
            ));
            assert_eq!(calls.get(), 0);
            assert_eq!(balances(&pool), (amount, 0, amount));
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            drop((run, reservation));
            let bytes = pool
                .prepare_shared_controller_bytes(|| {
                    calls.set(calls.get() + 1);
                    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
                    assert!(matches!(
                        pool.reserve(&InferenceExecutionIdentity::default(), &admission(0)),
                        Err(WorkingMemoryError::UnknownBound)
                    ));
                    buffer(53)
                })
                .unwrap();
            assert_eq!(calls.get(), 1);
            assert_eq!(bytes.capacity_bytes(), Some(53));
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            let alias = bytes.clone();
            drop(bytes);
            assert_eq!(pool.used_bytes().unwrap(), 53);
            drop(alias);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn never_admitted_mixed_sources_compete_with_later_requests_at_exact_capacity() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let filter = pool
        .prepare_shared_token_filter(|| token_filter(37))
        .unwrap();
    let bytes = pool.prepare_shared_controller_bytes(|| buffer(53)).unwrap();
    assert_eq!(balances(&pool), (0, 90, 90));
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
    drop(exact);
    drop(bytes);
    assert_eq!(pool.used_bytes().unwrap(), 37);
    drop(filter);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn byte_registration_rejection_and_factory_unwind_retire_temporary_ownership() {
    let pool = WorkingMemoryPool::new(52, 0).unwrap();
    let error = pool
        .prepare_shared_controller_bytes(|| {
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            buffer(53)
        })
        .unwrap_err();
    assert!(matches!(
        error,
        ControllerStorageError::Attachment(SharedStorageAttachmentError::Provider(
            WorkingMemoryError::BudgetExceeded {
                required_bytes: 53,
                available_bytes: 52,
            }
        ))
    ));
    assert_eq!(balances(&pool), (0, 0, 0));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = pool.prepare_shared_controller_bytes(|| {
            let _payload = buffer(17);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            panic!("producer failed with local byte payload")
        });
    }))
    .is_err());
    assert_eq!(balances(&pool), (0, 0, 0));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let empty = pool.prepare_shared_controller_bytes(Vec::new).unwrap();
    assert_eq!(empty.capacity_bytes(), Some(0));
    assert_eq!(balances(&pool), (0, 0, 0));
}

#[test]
fn byte_payload_retirement_reenters_accounting_without_identity_metadata_pinning_it() {
    struct Reenter {
        pool: WorkingMemoryPool,
        retired: Arc<AtomicBool>,
    }
    impl Drop for Reenter {
        fn drop(&mut self) {
            assert!(self.pool.0.usage.try_lock().is_ok());
            let temporary = self.pool.register_storage([(811u32, 7)]).unwrap();
            assert!(self.pool.used_bytes().unwrap() >= 7);
            drop(temporary);
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let pool = WorkingMemoryPool::new(512, 0).unwrap();
    let bytes = pool.prepare_shared_controller_bytes(|| buffer(53)).unwrap();
    let identity = bytes.identity().clone();
    let retired = Arc::new(AtomicBool::new(false));
    bytes
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Reenter {
                pool: pool.clone(),
                retired: retired.clone(),
            }))
        })
        .unwrap();
    let controller = MixedController::new(vec![], vec![bytes.clone()], 53);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    drop(controller);
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 53);
    drop(bytes);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(contract.shared.get(&identity).unwrap().bytes, 53);
    drop((identity, contract));
}
