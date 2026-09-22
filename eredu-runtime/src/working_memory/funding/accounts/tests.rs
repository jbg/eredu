// Host account phases retain their exact charged controls through retirement.
// These neutral fixtures construct the account worker without native execution.
use super::*;
fn account_used(pool: &MemoryLedger) -> Result<u64, WorkingMemoryError> {
    let snapshot = pool.snapshot()?;
    Ok(pool
        .payload_used_bytes()?
        .checked_add(snapshot.domains[0].reservation_control_bytes)
        .unwrap())
}

fn accepted(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    bytes: u64,
    floor: u64,
    capacity: u64,
) -> Result<PendingAccount, WorkingMemoryError> {
    let limits = crate::working_memory::memory_fixture::resolved_host_limits(pool, capacity);
    let mut usage = pool
        .0
        .usage
        .lock()
        .map_err(|_| WorkingMemoryError::Poisoned)?;
    let commit =
        PreparedAccountCommit::prepare(pool, execution, &usage, bytes, Some(&limits), &[])?;
    PendingAccount::accept(
        pool,
        execution,
        &mut usage,
        commit,
        bytes,
        Some(limits.clone()),
        floor,
    )
}

#[test]
fn original_pending_publishes_before_partial_error_and_issued_ids_do_not_recur() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1000, 7).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let pending = accepted(&pool, &execution, 150, 32, 700).unwrap();
    let id = pool
        .0
        .usage
        .lock()
        .unwrap()
        .pending_original
        .as_ref()
        .unwrap()
        .id();
    assert_eq!(account_used(&pool).unwrap(), 157);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 700);
    assert!(matches!(
        accepted(&pool, &execution, 1, 0, 1000),
        Err(WorkingMemoryError::AccountConstructionBusy)
    ));
    assert!(pool.acquire_unquoted().is_err());
    let ticket = pending.publish();
    ticket.status().unwrap();
    assert_eq!(ticket.id(), id);
    assert!(pool.0.usage.lock().unwrap().pending_original.is_none());
    // The actual second diagnostic reserve fails after retaining a real prefix.
    struct Partial {
        prefix: Vec<u64>,
        cause: std::collections::TryReserveError,
        ticket: AccountTicket,
    }
    let mut prefix = Vec::new();
    prefix.try_reserve_exact(3).unwrap();
    prefix.extend_from_slice(&[13, 21, 34]);
    let cause = prefix.try_reserve_exact(usize::MAX).unwrap_err();
    let partial = Partial {
        prefix,
        cause,
        ticket,
    };
    let other = accepted(&pool, &execution, 80, 16, 650).unwrap().publish();
    other.status().unwrap();
    assert!(other.id() > id);
    assert_eq!(partial.prefix, [13, 21, 34]);
    assert!(std::error::Error::source(&partial.cause).is_none());
    assert_eq!(partial.ticket.id(), id);
    assert_eq!(account_used(&pool).unwrap(), 237);
    drop(partial);
    assert_eq!(account_used(&pool).unwrap(), 87);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 650);
    drop(other);
    assert_eq!(account_used(&pool).unwrap(), 7);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 1000);
    assert!(accepted(&pool, &execution, 994, 1, 1000).is_err());
    let next = accepted(&pool, &execution, 40, 16, 1000).unwrap().publish();
    assert!(next.id() > id);
    drop(next);
}

#[test]
fn pending_unwind_returns_only_actual_unpublished_charge_and_keeps_issuance() {
    let pool = crate::working_memory::memory_fixture::host_ledger(500, 11).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let before = pool.0.usage.lock().unwrap().next_funding;
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _pending = accepted(&pool, &execution, 200, 64, 350).unwrap();
        assert_eq!(account_used(&pool).unwrap(), 211);
        panic!("before fixed node construction");
    }));
    assert!(failed.is_err());
    assert_eq!(account_used(&pool).unwrap(), 11);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 500);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, before + 1);
    assert_eq!(pool.payload_peak_bytes().unwrap(), 211);
    drop(accepted(&pool, &execution, 50, 16, 500).unwrap().publish());
    assert_eq!(account_used(&pool).unwrap(), 11);
}

#[test]
fn concurrent_terminal_nodes_keep_exact_ceiling_and_floor_through_deallocation() {
    let pool = crate::working_memory::memory_fixture::host_ledger(900, 7).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let first = accepted(&pool, &execution, 120, 32, 500).unwrap().publish();
    let second = accepted(&pool, &execution, 120, 48, 300).unwrap().publish();
    let (entered, arrival) = std::sync::mpsc::sync_channel(1);
    let (release, wait) = std::sync::mpsc::sync_channel::<()>(1);
    let worker = std::thread::spawn(move || {
        AFTER_NODE_RETIRE.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |pool| {
                // This would deadlock if the node or hook were destroyed under Usage.
                let observation = (account_used(&pool), pool.payload_effective_capacity());
                let _ = entered.send(observation);
                let _ = wait.recv_timeout(std::time::Duration::from_secs(10));
            }))
        });
        drop(first);
    });
    let observed = arrival.recv_timeout(std::time::Duration::from_secs(10));
    // Always release and join before checking assertions, including a failed hook.
    drop(second);
    let held = (account_used(&pool), pool.payload_effective_capacity());
    let third = accepted(&pool, &execution, 50, 16, 700).map(PendingAccount::publish);
    let held_with_third = (account_used(&pool), pool.payload_effective_capacity());
    drop(release);
    let joined = worker.join();
    assert!(joined.is_ok());
    assert_eq!(observed.unwrap(), (Ok(159), Ok(300))); // baseline + first floor + live second
    assert_eq!(held, (Ok(87), Ok(300))); // both exact floors; second is queued
    assert_eq!(held_with_third, (Ok(137), Ok(300)));
    let third = third.unwrap();
    assert_eq!(account_used(&pool).unwrap(), 57);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 700);
    drop(third);
    assert_eq!(account_used(&pool).unwrap(), 7);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 900);
}

#[test]
fn poisoned_original_publication_clears_pending_slot_and_retains_conservative_account() {
    let pool = crate::working_memory::memory_fixture::host_ledger(500, 3).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let pending = accepted(&pool, &execution, 100, 32, 250).unwrap();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _usage = pool.0.usage.lock().unwrap();
        panic!("poison between accepted Q and node publication");
    }));
    assert!(failed.is_err());
    let ticket = pending.publish();
    assert_eq!(ticket.status(), Err(WorkingMemoryError::Poisoned));
    drop(ticket);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert!(usage.pending_original.is_none());
    assert_eq!(usage.reserved, 100);
    assert_eq!(usage.reservations, 1);
    let account = usage.funding.values().next().unwrap();
    assert!(account.quarantined);
    assert!(!account.metadata_live);
    assert_eq!(account.control_floor, 32);
}

#[test]
fn prepared_copy_final_owners_drain_host_account_without_an_external_pool() {
    use crate::working_memory::{AdmittedWorkspaceCopy, WorkspaceCopyAccountLayout};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    struct HostProbe {
        pool: std::sync::Weak<crate::working_memory::Pool>,
        copy: Arc<AtomicU64>,
        drops: Arc<AtomicUsize>,
        bytes: u64,
    }
    impl Drop for HostProbe {
        fn drop(&mut self) {
            // The H ticket still keeps the pool alive here. This acquisition
            // also proves that the final token is retired outside Usage.
            let pool = self.pool.upgrade().expect("host account still owns pool");
            let usage = pool.usage.lock().unwrap();
            assert!(
                usage
                    .funding
                    .live_identity(self.copy.load(Ordering::SeqCst))
                    .is_none()
            );
            assert_eq!(usage.reserved, self.bytes);
            assert_eq!(self.drops.fetch_add(1, Ordering::SeqCst), 0);
        }
    }
    struct HostOwner {
        _probe: HostProbe,
        _ticket: AccountTicket,
    }

    // Numeric account controls only, like the other fixtures in this module.
    // No native work is submitted; the untouched scope can be certified.
    for last in 0..4 {
        let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 7).unwrap();
        let pool_weak = Arc::downgrade(&pool.0);
        let layout = WorkspaceCopyAccountLayout::workspace().unwrap();
        let host_bytes = u64::try_from(layout.requested_bytes()).unwrap();
        let source_execution = InferenceExecutionIdentity::default();
        let ticket = accepted(&pool, &source_execution, host_bytes, host_bytes, 1 << 20)
            .unwrap()
            .publish();
        let drops = Arc::new(AtomicUsize::new(0));
        let copy_id = Arc::new(AtomicU64::new(u64::MAX));
        let host = eredu_core::HostPreparationAuthority::retain(HostOwner {
            _probe: HostProbe {
                pool: pool_weak.clone(),
                copy: copy_id.clone(),
                drops: drops.clone(),
                bytes: host_bytes,
            },
            _ticket: ticket,
        });
        let execution = layout.execution(&host);
        let identity = Arc::downgrade(&execution.0);
        let extra_identity = execution.clone();
        let source = pool
            .register_host_storage(std::iter::empty::<(u32, u64)>())
            .unwrap();
        let pin = RegisteredStoragePin::new(
            pool.pin_registered_storage(std::iter::empty::<(u32, u64)>())
                .unwrap(),
        );
        let (run, scope) = pool
            .open_workspace_copy_account(
                &source,
                pin,
                &execution,
                &crate::working_memory::memory_fixture::host_requirements(&pool, 64),
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 20),
            )
            .unwrap();
        copy_id.store(run.id, Ordering::SeqCst);
        let (custody, scope) = AdmittedWorkspaceCopy::from_account(
            execution,
            crate::working_memory::memory_fixture::host_requirements(&pool, 64),
            run,
            scope,
        )
        .into_parts();
        let retained = (last == 3).then(|| custody.retention());
        assert_eq!(account_used(&pool).unwrap(), 7 + host_bytes + 64);
        drop((source, source_execution, host, pool));
        match last {
            0 => {
                drop((extra_identity, custody));
                assert!(identity.upgrade().is_none());
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                assert!(pool_weak.upgrade().is_some());
                scope.certify().unwrap();
            }
            1 => {
                scope.certify().unwrap();
                drop(custody);
                assert_eq!(Arc::weak_count(&extra_identity.0), 1);
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                drop(extra_identity);
            }
            2 => {
                scope.certify().unwrap();
                drop(extra_identity);
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                drop(custody);
            }
            _ => {
                scope.certify().unwrap();
                drop((extra_identity, custody));
                assert!(identity.upgrade().is_some());
                assert_eq!(drops.load(Ordering::SeqCst), 0);
            }
        }
        drop(retained);
        // No external drainer, poll, pool handle or ordinary global cleanup.
        assert!(identity.upgrade().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(pool_weak.upgrade().is_none());
    }
}

#[test]
fn finite_copy_publication_checks_exact_account_and_retires_last_output_host_custody() {
    use crate::working_memory::{
        AdmittedWorkspaceCopy, WorkspaceCopyAccountLayout, WorkspaceCopyPublicationPlan,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct HostProbe {
        pool: std::sync::Weak<crate::working_memory::Pool>,
        drops: Arc<AtomicUsize>,
        bytes: u64,
    }
    impl Drop for HostProbe {
        fn drop(&mut self) {
            let pool = self
                .pool
                .upgrade()
                .expect("H ticket still retains the pool");
            let usage = pool.usage.lock().unwrap();
            assert_eq!(usage.reserved, self.bytes);
            assert_eq!(usage.registered - usage.registry_metadata, 0);
            assert!(usage.storage.is_empty());
            assert_eq!(self.drops.fetch_add(1, Ordering::SeqCst), 0);
        }
    }
    struct HostOwner {
        _probe: HostProbe,
        _ticket: AccountTicket,
    }
    // This is the neutral physical-capacity contract, without a native worker.
    // The two untouched scopes may be certified after the publication checks.
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let pool_weak = Arc::downgrade(&pool.0);
    let account = WorkspaceCopyAccountLayout::workspace().unwrap();
    let publication = || WorkspaceCopyPublicationPlan::<u32>::new(1, 0).unwrap();
    let host_bytes =
        u64::try_from(account.requested_bytes() * 2 + publication().requested_bytes() * 3).unwrap();
    let source_execution = InferenceExecutionIdentity::default();
    let ticket = accepted(&pool, &source_execution, host_bytes, host_bytes, 1 << 20)
        .unwrap()
        .publish();
    let drops = Arc::new(AtomicUsize::new(0));
    let host = eredu_core::HostPreparationAuthority::retain(HostOwner {
        _probe: HostProbe {
            pool: pool_weak.clone(),
            drops: drops.clone(),
            bytes: host_bytes,
        },
        _ticket: ticket,
    });
    let source = pool
        .register_host_storage(std::iter::empty::<(u32, u64)>())
        .unwrap();
    let open = || {
        let execution = WorkspaceCopyAccountLayout::workspace()
            .unwrap()
            .execution(&host);
        let pin = RegisteredStoragePin::new(
            pool.pin_registered_storage(std::iter::empty::<(u32, u64)>())
                .unwrap(),
        );
        let (run, scope) = pool
            .open_workspace_copy_account(
                &source,
                pin,
                &execution,
                &crate::working_memory::memory_fixture::host_requirements(&pool, 64),
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 20),
            )
            .unwrap();
        AdmittedWorkspaceCopy::from_account(
            execution,
            crate::working_memory::memory_fixture::host_requirements(&pool, 64),
            run,
            scope,
        )
        .into_parts()
    };
    let (copy, scope) = open();
    let (foreign, foreign_scope) = open();
    let before = account_used(&pool).unwrap();
    assert!(matches!(
        publication().prepare(&copy, &foreign_scope),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(account_used(&pool).unwrap(), before);
    let mut rejected = publication().prepare(&copy, &scope).unwrap();
    rejected.push(17, 24, pool.host_placement_handle()).unwrap();
    assert_eq!(
        rejected.publish(&foreign_scope),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert!(rejected.input(0).is_none());
    assert_eq!(account_used(&pool).unwrap(), before);
    assert!(pool.pin_registered_storage([(17u32, 24)]).is_err());
    let mut accepted = publication().prepare(&copy, &scope).unwrap();
    accepted.push(19, 32, pool.host_placement_handle()).unwrap();
    accepted.publish(&scope).unwrap();
    let output = accepted.take_input(0).unwrap();
    assert_eq!(output.bytes(), Some(32));
    {
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.registered - usage.registry_metadata, 32);
    }
    assert_eq!(account_used(&pool).unwrap(), before); // B moves; no second charge
    scope.certify().unwrap();
    foreign_scope.certify().unwrap();
    drop((
        accepted,
        rejected,
        copy,
        foreign,
        source,
        source_execution,
        host,
        pool,
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(pool_weak.upgrade().is_some());
    // The registration and canonical namespace, rather than the copied owner,
    // now keep H alive. Final release needs no external pool poll or drain.
    drop(output);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(pool_weak.upgrade().is_none());
}

#[test]
fn original_ticket_quarantine_preserves_unfunded_phase_and_full_charge() {
    for poison in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1000, 7).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let ticket = accepted(&pool, &execution, 150, 32, 700).unwrap().publish();
        let id = ticket.id();
        ticket.status().unwrap();
        let issuance = pool.0.usage.lock().unwrap().next_funding;
        if poison {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _usage = pool.0.usage.lock().unwrap();
                    panic!("original source publication failure");
                }))
                .is_err()
            );
        }
        // An original ticket remains outside the native Funded lookup. Both
        // ordinary and poisoned cleanup must fence it without a second panic.
        ticket.quarantine();
        ticket.quarantine();
        {
            let usage = pool.0.usage.lock().unwrap_or_else(|e| e.into_inner());
            assert!(usage.funding.get(&id).is_none());
            let node = usage.funding.nodes().find(|node| node.id == id).unwrap();
            assert_eq!(node.phase, Phase::Unfunded);
            assert!(node.state.metadata_live && node.state.quarantined);
            assert_eq!(node.state.control_floor, 32);
            assert_eq!(usage.reserved, 150);
            assert_eq!(usage.next_funding, issuance);
        }
        assert_eq!(
            ticket.status(),
            Err(if poison {
                WorkingMemoryError::Poisoned
            } else {
                WorkingMemoryError::ExecutionFenced
            })
        );
        drop(ticket);
        let usage = pool.0.usage.lock().unwrap_or_else(|e| e.into_inner());
        let state = usage.funding.get(&id).unwrap();
        assert!(state.quarantined && !state.metadata_live);
        assert_eq!(usage.reserved, 150);
        assert_eq!(usage.reservations, 1);
        assert_eq!(
            usage.funding.capacity(pool.topology().host_domain()),
            Ok(
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, 700)
                    .get(pool.topology().host_domain())
                    .unwrap()
            )
        );
    }
}

#[test]
fn original_source_capacity_keeps_unfunded_phase_and_rejects_foreign_or_retired_accounts() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1000, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let ticket = accepted(&pool, &execution, 150, 32, 700).unwrap().publish();
    let id = ticket.id();
    {
        let usage = pool.0.usage.lock().unwrap();
        ticket.validate_in(&usage).unwrap();
        assert!(usage.funding.get(&id).is_none());
        assert_eq!(
            usage.funding.accepted_source_capacity(id, &execution),
            Ok(Some(
                &crate::working_memory::memory_fixture::resolved_host_limits(&pool, 700)
            ))
        );
        assert_eq!(
            usage
                .funding
                .accepted_source_capacity(id, &InferenceExecutionIdentity::default()),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert!(usage.funding.get(&id).is_none());
    }
    assert_eq!(account_used(&pool).unwrap(), 150);
    drop(ticket);
    let usage = pool.0.usage.lock().unwrap();
    assert_eq!(
        usage.funding.accepted_source_capacity(id, &execution),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    drop(usage);
    assert_eq!(account_used(&pool).unwrap(), 0);
}
