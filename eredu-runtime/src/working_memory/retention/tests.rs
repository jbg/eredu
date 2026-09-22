use super::*;
use crate::{DeviceState, StateLayout};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, StateMemoryLayout, WorkspaceBound, cache::LayerCachePolicy,
};
use eredu_nn::workspace::WorkspaceBackend;
use std::{cell::Cell, rc::Rc};

fn zero_admission(pool: &crate::working_memory::MemoryLedger) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless fixture without managed payload");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    crate::working_memory::memory_fixture::attribute_host_admission(
        &pool,
        Admission {
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
            requested_positions: 1,
            state,
            incremental_required_bytes: Some(0),
        },
    )
}

fn zero_request_ledger() -> crate::working_memory::MemoryLedger {
    let probe = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let bytes =
        crate::working_memory::memory_fixture::reservation_bytes(&probe, &zero_admission(&probe));
    crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap()
}

#[test]
fn revision_is_stable_for_cold_clones_and_charge_only_retention() {
    const EMPTY: InferenceRetention = InferenceRetention::new();
    let mut original = EMPTY;
    let revision = original.revision().clone();
    let mut copied = original.clone();
    original.validate_revision(&revision).unwrap();
    copied.validate_revision(&revision).unwrap();
    assert!(matches!(
        InferenceRetention::new().validate_revision(&revision),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let pool = crate::working_memory::memory_fixture::host_ledger(65536, 0).unwrap();
    let lease = pool.acquire_unquoted().unwrap();
    copied.retain_unquoted(&lease);
    original.extend_from(&copied);
    original.validate_revision(&revision).unwrap();
    copied.validate_revision(&revision).unwrap();
    copied.commit_span(1);
    assert!(copied.validate_revision(&revision).is_err());
    original.validate_revision(&revision).unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop((lease, copied, original));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn admission_commit_and_equal_frontier_restore_never_revive_saved_revision() {
    let pool = crate::working_memory::memory_fixture::host_ledger(65536, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = |reserve: Option<u64>| {
        let mut admission = zero_admission(&pool);
        // A nonzero safety reserve on an otherwise stateless fixture makes
        // charge lifetime observable without claiming any native allocation.
        admission.additional_headroom =
            eredu_core::MemoryHeadroomDeclarations::new([("host".into(), reserve.unwrap())]);
        InferenceRequest::from(pool.reserve(&execution, &admission).unwrap())
    };
    let first = request(Some(20));
    let second = request(Some(30));
    let mut installed = InferenceRetention::new();
    let fresh = installed.revision().clone();
    installed.retain(&first);
    installed.validate_revision(&fresh).unwrap();
    installed.admit(&first);
    assert!(installed.validate_revision(&fresh).is_err());
    let admitted = installed.revision().clone();
    let saved = installed.clone();
    installed.admit(&first);
    installed.validate_revision(&admitted).unwrap();
    installed.commit_span(1);
    assert!(installed.validate_revision(&admitted).is_err());
    let committed = installed.revision().clone();
    installed.admit(&first);
    installed.validate_revision(&committed).unwrap();
    assert_eq!(installed.admission().unwrap().position(), 1);
    installed.admit(&second);
    assert!(installed.validate_revision(&committed).is_err());
    let replacement = installed.revision().clone();
    assert_eq!(
        installed.admission().unwrap().position(),
        saved.admission().unwrap().position()
    );
    installed.restore_admission(&saved);
    for previous in [&fresh, &admitted, &committed, &replacement] {
        assert!(installed.validate_revision(previous).is_err());
    }
    installed
        .admission()
        .unwrap()
        .request()
        .validate_same_request(&first)
        .unwrap();
    let restored = installed.revision().clone();
    installed.clone_from(&saved);
    assert!(installed.validate_revision(&restored).is_err());
    assert!(installed.validate_revision(saved.revision()).is_err());
    let restored_again = installed.revision().clone();
    installed.restore_admission(&InferenceRetention::new());
    assert!(installed.validate_revision(&restored_again).is_err());
    assert_eq!(installed.admission().unwrap().position(), 0);
    assert_eq!(installed.requests().len(), 2);
    assert_eq!(pool.payload_used_bytes().unwrap(), 50);
    let host = pool.topology().host_domain();
    let full = first
        .memory_reservation()
        .requirements()
        .get(host)
        .unwrap()
        .total()
        .unwrap()
        + second
            .memory_reservation()
            .requirements()
            .get(host)
            .unwrap()
            .total()
            .unwrap();
    assert_eq!(pool.payload_peak_bytes().unwrap(), full);
    drop((first, second, saved));
    assert_eq!(pool.payload_used_bytes().unwrap(), 50);
    drop(installed);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn restoring_an_empty_equal_branch_also_replaces_its_revision() {
    let mut installed = InferenceRetention::new();
    let saved = installed.clone();
    let revision = installed.revision().clone();
    installed.restore_admission(&saved);
    assert!(installed.admission().is_none());
    assert!(installed.validate_revision(&revision).is_err());
    assert!(installed.validate_revision(saved.revision()).is_err());
    assert_eq!(installed.logical_metadata_bytes(), Some(0));
}

#[test]
fn complete_state_exchange_invalidates_both_sides_without_retiring_their_owners() {
    type EmptyState = DeviceState<WorkspaceBackend, ()>;
    let pool = crate::working_memory::memory_fixture::host_ledger(65536, 0).unwrap();
    let first_owner = pool.acquire_unquoted().unwrap();
    let second_owner = pool.acquire_unquoted().unwrap();
    let mut installed = EmptyState::stateless();
    let mut displaced = EmptyState::stateless();
    installed
        .inference_retention_mut()
        .retain_unquoted(&first_owner);
    displaced
        .inference_retention_mut()
        .retain_unquoted(&second_owner);
    let first = installed.inference_retention().revision().clone();
    let second = displaced.inference_retention().revision().clone();
    drop((first_owner, second_owner));
    exchange_inference_state(&mut installed, &mut displaced);
    for state in [&installed, &displaced] {
        assert!(
            state
                .inference_retention()
                .validate_revision(&first)
                .is_err()
        );
        assert!(
            state
                .inference_retention()
                .validate_revision(&second)
                .is_err()
        );
    }
    let swapped_first = installed.inference_retention().revision().clone();
    let swapped_second = displaced.inference_retention().revision().clone();
    exchange_inference_state(&mut installed, &mut displaced);
    for state in [&installed, &displaced] {
        for previous in [&first, &second, &swapped_first, &swapped_second] {
            assert!(
                state
                    .inference_retention()
                    .validate_revision(previous)
                    .is_err()
            );
        }
        assert!(state.as_ref().is_empty());
    }
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop(installed);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(displaced);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn unquoted_retention_deduplicates_lease_aliases_without_admitting_work() {
    let pool = crate::working_memory::memory_fixture::host_ledger(
        100 + crate::working_memory::MemoryLedger::unquoted_owner_control_bytes().unwrap(),
        7,
    )
    .unwrap();
    let owner = pool.acquire_unquoted().unwrap();
    let mut retention = InferenceRetention::new();
    retention.retain_unquoted(&owner);
    retention.retain_unquoted(&owner.clone());
    assert_eq!(retention.requests().len(), 0);
    assert!(retention.admission().is_none());
    assert_eq!(
        retention.logical_metadata_bytes(),
        Some(std::mem::size_of::<WorkingMemoryUnquotedLease>() as u64),
    );
    let descendant = retention.clone();
    retention.extend_from(&descendant);
    drop((owner, retention));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool.payload_used_bytes().unwrap(), 7);
    assert!(matches!(
        pool.reserve(
            &InferenceExecutionIdentity::default(),
            &zero_admission(&pool)
        ),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(descendant);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.payload_used_bytes().unwrap(), 7);
}

#[test]
fn restoration_keeps_distinct_unquoted_owners_even_from_the_same_pool() {
    let pool = crate::working_memory::memory_fixture::host_ledger(65536, 0).unwrap();
    let previous = pool.acquire_unquoted().unwrap();
    let current = pool.acquire_unquoted().unwrap();
    let mut snapshot = InferenceRetention::new();
    snapshot.retain_unquoted(&previous);
    let mut installed = InferenceRetention::new();
    installed.retain_unquoted(&current);
    installed.clone_from(&snapshot);
    installed.restore_admission(&snapshot);
    installed.clone_from(&InferenceRetention::new());
    assert_eq!(
        installed.logical_metadata_bytes(),
        Some(2 * std::mem::size_of::<WorkingMemoryUnquotedLease>() as u64),
    );
    drop((previous, current, snapshot));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(installed);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn stateless_device_state_clone_and_import_keep_unquoted_ownership() {
    type EmptyState = DeviceState<WorkspaceBackend, ()>;
    let pool = zero_request_ledger();
    let owner = pool.acquire_unquoted().unwrap();
    let mut original = EmptyState::stateless();
    original.inference_retention_mut().retain_unquoted(&owner);
    let snapshot = original.clone();
    let mut imported = EmptyState::stateless();
    imported.inherit_inference_retention(&snapshot);
    imported
        .inference_retention()
        .validate_revision(original.inference_retention().revision())
        .unwrap();
    assert!(imported.as_ref().is_empty());
    drop((owner, original, snapshot));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert!(matches!(
        pool.reserve(
            &InferenceExecutionIdentity::default(),
            &zero_admission(&pool)
        ),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(imported);
    let reserved = pool
        .reserve(
            &InferenceExecutionIdentity::default(),
            &zero_admission(&pool),
        )
        .unwrap();
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(reserved);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[derive(Debug)]
struct HostLayer {
    pool: super::super::MemoryLedger,
    drops: Rc<Cell<usize>>,
    reject_clone: bool,
}

impl Clone for HostLayer {
    fn clone(&self) -> Self {
        assert!(!self.reject_clone, "fixture state copy failure");
        Self {
            pool: self.pool.clone(),
            drops: Rc::clone(&self.drops),
            reject_clone: false,
        }
    }
}

impl Drop for HostLayer {
    fn drop(&mut self) {
        assert!(self.pool.unquoted_owner_count().unwrap() > 0);
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn failed_host_state_restore_keeps_both_owners_until_payload_retirement() {
    let pool = crate::working_memory::memory_fixture::host_ledger(65536, 0).unwrap();
    let first = pool.acquire_unquoted().unwrap();
    let second = pool.acquire_unquoted().unwrap();
    let drops = Rc::new(Cell::new(0));
    let make = |reject_clone| {
        DeviceState::<WorkspaceBackend, HostLayer>::create(
            StateLayout::new(LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap())
                .unwrap(),
            |_, _| {
                Ok::<_, std::convert::Infallible>(HostLayer {
                    pool: pool.clone(),
                    drops: Rc::clone(&drops),
                    reject_clone,
                })
            },
        )
        .unwrap()
    };
    let mut installed = make(false);
    installed.inference_retention_mut().retain_unquoted(&first);
    let mut snapshot = make(true);
    snapshot.inference_retention_mut().retain_unquoted(&second);
    let installed_revision = installed.inference_retention().revision().clone();
    let saved_revision = snapshot.inference_retention().revision().clone();
    drop((first, second));
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        installed.clone_from(&snapshot);
    }));
    assert!(failed.is_err());
    assert!(
        installed
            .inference_retention()
            .validate_revision(&installed_revision)
            .is_err()
    );
    assert!(
        installed
            .inference_retention()
            .validate_revision(&saved_revision)
            .is_err()
    );
    drop(snapshot);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    assert_eq!(drops.get(), 1);
    drop(installed);
    assert_eq!(drops.get(), 2);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

mod empty;

mod checkpoint;
