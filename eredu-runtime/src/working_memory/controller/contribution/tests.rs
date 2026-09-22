use super::*;
use eredu_core::{
    CapabilityError, SharedStorageAccountingId, SharedTokenFilter, TextControllerContractError,
    TextControllerStorage, TextFilterWorkspace, TokenFilter, TokenFilterController,
};
use std::{
    convert::Infallible,
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

struct Controller {
    filters: Vec<SharedTokenFilter>,
    known: bool,
}
impl Controller {
    fn new(filters: Vec<SharedTokenFilter>) -> Self {
        Self {
            filters,
            known: true,
        }
    }
}
impl TokenFilterController for Controller {
    type Error = Infallible;
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        if self.known {
            TextControllerStorage::RunOwnedWithSharedFilters(&self.filters)
        } else {
            TextControllerStorage::Unknown
        }
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        panic!("cold contribution must not query a decision")
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        panic!("cold contribution must not commit a token")
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        panic!("cold contribution must not query controller state")
    }
}

fn filter(capacity: usize) -> TokenFilter {
    let mut bits = Vec::with_capacity(capacity);
    bits.extend([true, false, true, false]);
    TokenFilter::allowed(bits).unwrap()
}
fn prepared(pool: &MemoryLedger, capacity: usize) -> SharedTokenFilter {
    pool.prepare_shared_token_filter(|| filter(capacity))
        .unwrap()
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 2,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: eredu_core::OutputDemand::LastPosition,
    }
}
fn workspace(additional_host_bytes: u64) -> TextControllerWorkspace<'static> {
    TextControllerWorkspace {
        filter: TextFilterWorkspace::OptionalMask {
            max_mask_positions: 4,
            mask_capacity_bytes: 8,
        },
        additional_host_bytes,
    }
}
fn outside() -> ExecutionWorkspaceEstimate {
    let bound = |bytes| WorkspaceBound::bounded(bytes, "independent enclosing fixture component");
    ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: geometry(),
        activations: bound(11),
        attention: bound(13),
        vocabulary: bound(17),
        state_update: bound(19),
        materialization: bound(23),
        retained: bound(29),
    }
}
fn unchanged_components(
    actual: &ExecutionWorkspaceEstimate,
    expected: &ExecutionWorkspaceEstimate,
) {
    assert_eq!(actual.geometry, expected.geometry);
    assert_eq!(actual.activations, expected.activations);
    assert_eq!(actual.attention, expected.attention);
    assert_eq!(actual.vocabulary, expected.vocabulary);
    assert_eq!(actual.state_update, expected.state_update);
    assert_eq!(actual.materialization, expected.materialization);
}
fn mismatch(error: ControllerStorageError) {
    assert!(matches!(
        &error,
        ControllerStorageError::Storage(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(error.source().unwrap().is::<WorkingMemoryError>());
}

#[test]
fn exact_unique_source_capacity_is_removed_only_from_fixed_controller_addition() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let source = prepared(&pool, 37);
    let controller = Controller::new(vec![source.clone(), source.clone()]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let pin = contract.pin_registered(&controller, &pool).unwrap();
    assert_eq!(source.allowed_mask().unwrap().len(), 4);
    assert_eq!(pin.source_bytes(), 37);
    assert!(pin.pool().same_ledger(&pool));
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
    let declaration = workspace(64);
    let contribution = ControllerWorkspaceContribution::new(
        geometry(),
        declaration,
        4,
        &contract,
        &pool,
        Some(&pin),
    )
    .unwrap();
    let base = outside();
    assert_eq!(base.peak_bytes().unwrap(), Some(112));
    let estimate = contribution.compose(base.clone()).unwrap();
    assert_eq!(
        estimate.full().retained.bytes(),
        Some(93 + contract.publication_control_bytes().unwrap())
    );
    assert_eq!(
        estimate.incremental().retained.bytes(),
        Some(56 + contract.publication_control_bytes().unwrap())
    );
    assert_eq!(
        estimate.full().peak_bytes().unwrap(),
        Some(176 + contract.publication_control_bytes().unwrap())
    );
    assert_eq!(
        estimate.incremental().peak_bytes().unwrap(),
        Some(139 + contract.publication_control_bytes().unwrap())
    );
    unchanged_components(estimate.full(), &base);
    unchanged_components(estimate.incremental(), &base);
    assert_eq!(estimate.geometry(), geometry());
    assert!(estimate.pool().same_ledger(&pool));
    assert_eq!(
        estimate.controller_contract(),
        contribution.controller_contract()
    );
    assert_eq!(estimate.controller_contract().additional_host_bytes(), 64);
    // Optional future masks have no source storage credit. Their bound remains eight bytes.
    assert_eq!(estimate.controller_contract().filter_capacity_bytes(), 8);
    assert_eq!(
        estimate.controller_contract(),
        &TextControllerContract::from_workspace(declaration, 4).unwrap()
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
}

#[test]
fn uncredited_contribution_preserves_the_full_allowance_and_pins_no_sources() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let source = prepared(&pool, 37);
    let controller = Controller::new(vec![source]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let contribution =
        ControllerWorkspaceContribution::new(geometry(), workspace(64), 4, &contract, &pool, None)
            .unwrap();
    let estimate = contribution.compose(outside()).unwrap();
    assert_eq!(estimate.full(), estimate.incremental());
    assert_eq!(
        estimate.full().peak_bytes().unwrap(),
        Some(176 + contract.publication_control_bytes().unwrap())
    );
    assert!(estimate.pin().is_none());
    drop(controller);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop((contract, contribution, estimate));
}

#[test]
fn complete_existing_only_pin_rejects_missing_members_without_partial_retention() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let registered = prepared(&pool, 37);
    let absent = SharedTokenFilter::new(filter(53));
    let controller = Controller::new(vec![registered.clone(), absent.clone()]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    mismatch(contract.pin_registered(&controller, &pool).unwrap_err());
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
    drop((controller, registered));
    // A staged first key must not survive rejection of the complete inventory.
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(absent.capacity_bytes(), Some(53));
    drop((absent, contract));
}

#[test]
fn existing_pin_rejects_conflicting_capacity_instead_of_registering_a_new_charge() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let source = SharedTokenFilter::new(filter(37));
    let conflicting = pool
        .register_host_storage([(source.identity().clone(), 38)])
        .unwrap();
    let controller = Controller::new(vec![source.clone()]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    mismatch(contract.pin_registered(&controller, &pool).unwrap_err());
    assert_eq!(pool.payload_used_bytes().unwrap(), 38);
    drop((controller, source, conflicting));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn foreign_pools_and_equal_sized_replacement_inventories_cannot_reuse_credit() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let foreign = super::super::tests::host_ledger(512, 0).unwrap();
    let first = prepared(&pool, 37);
    let replacement = prepared(&pool, 37);
    assert_eq!(first, replacement);
    let mut controller = Controller::new(vec![first]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let registered = contract.pin_registered(&controller, &pool).unwrap();
    mismatch(contract.pin_registered(&controller, &foreign).unwrap_err());
    mismatch(
        ControllerWorkspaceContribution::new(
            geometry(),
            workspace(64),
            4,
            &contract,
            &foreign,
            Some(&registered),
        )
        .unwrap_err(),
    );
    controller.filters[0] = replacement;
    mismatch(contract.pin_registered(&controller, &pool).unwrap_err());
    let replacement_contract = ControllerStorageContract::inspect(&controller).unwrap();
    assert_ne!(contract, replacement_contract);
    mismatch(
        ControllerWorkspaceContribution::new(
            geometry(),
            workspace(64),
            4,
            &replacement_contract,
            &pool,
            Some(&registered),
        )
        .unwrap_err(),
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 74);
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    drop((registered, controller));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn zero_byte_sources_still_bind_identity_without_requiring_registry_entries() {
    let pool = super::super::tests::host_ledger(0, 0).unwrap();
    let source = SharedTokenFilter::new(TokenFilter::All);
    let mut controller = Controller::new(vec![source.clone(), source.clone()]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let registered = contract.pin_registered(&controller, &pool).unwrap();
    assert_eq!(registered.source_bytes(), 0);
    let contribution = ControllerWorkspaceContribution::new(
        geometry(),
        TextControllerWorkspace {
            filter: source.as_ref().into(),
            additional_host_bytes: 0,
        },
        4,
        &contract,
        &pool,
        Some(&registered),
    )
    .unwrap();
    let estimate = contribution.compose(outside()).unwrap();
    assert_eq!(estimate.full(), estimate.incremental());
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    controller.filters = vec![SharedTokenFilter::new(TokenFilter::All)];
    mismatch(contract.pin_registered(&controller, &pool).unwrap_err());
}

#[test]
fn unknown_lifetime_invalid_metadata_and_unpriced_sources_preserve_typed_causes() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let mut controller = Controller::new(vec![prepared(&pool, 37)]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let registered = contract.pin_registered(&controller, &pool).unwrap();
    controller.known = false;
    assert!(matches!(
        contract.pin_registered(&controller, &pool),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert!(matches!(
        ControllerStorageContract::inspect(&controller),
        Err(ControllerStorageError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    controller.known = true;
    assert!(matches!(
        ControllerWorkspaceContribution::new(
            geometry(),
            workspace(36),
            4,
            &contract,
            &pool,
            Some(&registered),
        ),
        Err(ControllerStorageError::UnpricedSharedStorage {
            required_bytes: 37,
            available_bytes: 36
        })
    ));
    let mut invalid_geometry = geometry();
    invalid_geometry.prefill_chunk_positions = 0;
    let error = ControllerWorkspaceContribution::new(
        invalid_geometry,
        workspace(64),
        4,
        &contract,
        &pool,
        Some(&registered),
    )
    .unwrap_err();
    assert!(matches!(&error, ControllerStorageError::Estimate(_)));
    assert!(error.source().unwrap().is::<CapabilityError>());
    let mut invalid = workspace(64);
    invalid.filter = TextFilterWorkspace::OptionalMask {
        max_mask_positions: 0,
        mask_capacity_bytes: 8,
    };
    let error = ControllerWorkspaceContribution::new(
        geometry(),
        invalid,
        4,
        &contract,
        &pool,
        Some(&registered),
    )
    .unwrap_err();
    assert!(matches!(
        &error,
        ControllerStorageError::Contract(TextControllerContractError::InvalidOptionalMask { .. })
    ));
    assert!(error.source().unwrap().is::<TextControllerContractError>());
    assert!(matches!(
        ControllerWorkspaceContribution::new(
            geometry(),
            workspace(64),
            0,
            &contract,
            &pool,
            Some(&registered),
        ),
        Err(ControllerStorageError::Contract(_))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
}

#[test]
fn composition_checks_geometry_and_full_overflow_without_upgrading_unknown_components() {
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let controller = Controller::new(vec![prepared(&pool, 37)]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let registered = contract.pin_registered(&controller, &pool).unwrap();
    let contribution = ControllerWorkspaceContribution::new(
        geometry(),
        workspace(64),
        4,
        &contract,
        &pool,
        Some(&registered),
    )
    .unwrap();
    let mut wrong = outside();
    wrong.geometry.cached_positions = 1;
    mismatch(contribution.compose(wrong).unwrap_err());
    let mut overflowing = outside();
    overflowing.retained =
        WorkspaceBound::bounded(u64::MAX - 40, "full controller addition overflows");
    let error = contribution.compose(overflowing).unwrap_err();
    assert!(matches!(
        &error,
        ControllerStorageError::Estimate(CapabilityError::ArithmeticOverflow { .. })
    ));
    assert!(error.source().unwrap().is::<CapabilityError>());
    for retained_unknown in [false, true] {
        let mut missing = outside();
        let unknown = WorkspaceBound::Unknown {
            reason: "missing complete fixture bound".into(),
        };
        if retained_unknown {
            missing.retained = unknown;
        } else {
            missing.attention = unknown;
        }
        let estimate = contribution.compose(missing.clone()).unwrap();
        assert_eq!(estimate.full().peak_bytes().unwrap(), None);
        assert_eq!(estimate.incremental().peak_bytes().unwrap(), None);
        unchanged_components(estimate.full(), &missing);
        unchanged_components(estimate.incremental(), &missing);
        if retained_unknown {
            assert_eq!(estimate.full().retained, missing.retained);
            assert_eq!(estimate.incremental().retained, missing.retained);
        } else {
            assert_eq!(
                estimate.full().retained.bytes(),
                Some(93 + contract.publication_control_bytes().unwrap())
            );
            assert_eq!(
                estimate.incremental().retained.bytes(),
                Some(56 + contract.publication_control_bytes().unwrap())
            );
        }
    }
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
}

#[test]
fn accounting_proofs_outlive_mask_payload_without_owner_cycles_and_release_at_last_pin() {
    struct Retired(Arc<AtomicBool>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let pool = super::super::tests::host_ledger(512, 0).unwrap();
    let source = prepared(&pool, 37);
    let retired = Arc::new(AtomicBool::new(false));
    source
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Retired(retired.clone())))
        })
        .unwrap();
    let controller = Controller::new(vec![source.clone()]);
    let contract = ControllerStorageContract::inspect(&controller).unwrap();
    let registered = contract.pin_registered(&controller, &pool).unwrap();
    let contribution = ControllerWorkspaceContribution::new(
        geometry(),
        workspace(64),
        4,
        &contract,
        &pool,
        Some(&registered),
    )
    .unwrap();
    let estimate = contribution.compose(outside()).unwrap();
    let independent_estimate = estimate.clone();
    let scope_pin = estimate.pin().unwrap();
    drop((controller, source));
    assert!(
        retired.load(Ordering::SeqCst),
        "proofs must retain no SharedTokenFilter owner"
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
    drop((registered, contribution, estimate, independent_estimate));
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
    drop(scope_pin);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    // Identity-only contract metadata remains alive after all physical custody retires.
    assert_eq!(contract, contract.clone());
}
