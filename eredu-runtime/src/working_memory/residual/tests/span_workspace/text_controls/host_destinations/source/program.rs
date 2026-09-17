use super::*;
use crate::working_memory::{
    HostSourceConstructionProgram, HostSourcePeakSelection, OriginalHostSourceCustody,
    OriginalHostSourcePeakCapacity,
};

// The neutral oracle retains distinct physical owner identities and the actual
// accepted account; it does not stand in for native allocation qualification.
struct Peak {
    selection: HostSourcePeakSelection,
    custody: OriginalHostSourceCustody,
}
impl OriginalHostSourcePeakCapacity for Peak {
    fn selection(&self) -> HostSourcePeakSelection { self.selection }
    fn owner_identity(&self) -> usize { self as *const Self as usize }
    fn backing_bytes(&self) -> u64 { self.selection.backing_bytes() }
    fn source_custody(&self) -> OriginalHostSourceCustody { self.custody.clone() }
}

#[test]
fn admitted_source_program_preserves_independent_peaks_paged_partitions_and_spending() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let storage = pool.register_storage([(1u32, 64)]).unwrap();
    let a_peak = HostSourcePeakSelection::new(64).unwrap();
    let b_peak = HostSourcePeakSelection::new(96).unwrap();
    let a = HostSourceConstructionFacts::new(12, 2, 1).unwrap().with_peak_backing(a_peak).unwrap();
    let b = HostSourceConstructionFacts::new(20, 1, 1).unwrap().with_peak_backing(b_peak).unwrap();
    let paged = HostSourceConstructionFacts::new(24, 3, 2).unwrap();
    let program = HostSourceConstructionProgram::from_components(vec![a, b, paged]).unwrap();
    let Some(quote) = source_request_with_facts(&pool, program.facts()) else { return; };
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded { .. }))));
    let (_, reservation, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &reservation).unwrap();
    let held = span.protected_host_bytes();
    let controls = span.control_guard();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut root = host.take_source_constructions().unwrap();
    // Aggregate scalars never grant ordinary split or debit authority.
    assert!(root.split(12, 2).is_err());
    assert!(!root.try_debit(0).unwrap_err().retains_receipt());
    assert!(root.matches_facts(program.facts()));
    let mut components = root.partition_program(&program).unwrap();
    assert!(components.matches(&program));
    assert!(matches!(components.take(3), Err(WorkingMemoryError::AlreadyStarted)));
    let mut a_bank = components.take(0).unwrap();
    let mut b_bank = components.take(1).unwrap();
    let mut paged_bank = components.take(2).unwrap();
    assert!(a_bank.matches_facts(a) && b_bank.matches_facts(b) && paged_bank.matches_facts(paged));
    assert!(a_bank.belongs_to(&controls) && b_bank.belongs_to(&controls) && paged_bank.belongs_to(&controls));
    let a_owner = Box::new(Peak { selection: a_peak, custody: controls.clone().into() });
    let b_owner = Box::new(Peak { selection: b_peak, custody: controls.clone().into() });
    assert!(matches!(a_bank.bind_peak_capacity(&*b_owner), Err(WorkingMemoryError::IdentityMismatch)));
    assert!(a_bank.split(12, 2).is_err(), "a peak must bind its actual owner first");
    a_bank.bind_peak_capacity(&*a_owner).unwrap();
    b_bank.bind_peak_capacity(&*b_owner).unwrap();
    assert!(matches!(a_bank.bind_peak_capacity(&*a_owner), Err(WorkingMemoryError::IdentityMismatch)));
    let mut a_child = a_bank.split(12, 2).unwrap();
    let mut b_child = b_bank.split(20, 1).unwrap();
    let a_error = a_child.try_debit(13).unwrap_err();
    assert!(a_error.retains_receipt());
    assert_eq!((a_child.remaining_bytes(), a_child.remaining_attempts()), (12, 1));
    let a_receipt = a_child.try_debit(12).unwrap();
    let b_receipt = b_child.try_debit(20).unwrap();
    assert_eq!((a_child.remaining_bytes(), a_child.remaining_attempts()), (0, 0));
    drop((a_child, b_child));
    assert!(matches!(components.take(0), Err(WorkingMemoryError::AlreadyStarted)));
    assert!(a_bank.split(0, 0).is_err(), "retirement cannot restore a parent's partition");
    // This is an original paged root, retaining its two finite child grants.
    assert_eq!(paged_bank.remaining_partitions(), 2);
    let mut first = paged_bank.split(8, 1).unwrap();
    let mut second = paged_bank.split(16, 2).unwrap();
    assert!(first.split(0, 0).is_err(), "ordinary children remain terminal");
    let first_receipt = first.try_debit(8).unwrap();
    let paged_error = second.try_debit(17).unwrap_err();
    assert!(paged_error.retains_receipt());
    let second_receipt = second.try_debit(16).unwrap();
    drop((first, second));
    assert_eq!((paged_bank.remaining_bytes(), paged_bank.remaining_attempts(), paged_bank.remaining_partitions()), (0, 0, 0));
    assert!(paged_bank.split(0, 0).is_err());
    for index in 0..3 { assert!(matches!(components.take(index), Err(WorkingMemoryError::AlreadyStarted))); }
    drop((quote, span, host, reservation, run, storage, components, a_bank, b_bank,
        paged_bank, a_owner, b_owner, controls));
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop((a_receipt, b_receipt, first_receipt, second_receipt, a_error));
    assert_eq!(pool.used_bytes().unwrap(), held, "failed constructor keeps original account custody");
    drop(paged_error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_program_rejects_equal_scalar_and_foreign_program_roots_with_original_custody() {
    for foreign_program in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let storage = pool.register_storage([(1u32, 64)]).unwrap();
        let a = HostSourceConstructionFacts::new(12, 1, 1).unwrap();
        let b = HostSourceConstructionFacts::new(20, 1, 1).unwrap();
        let selected = HostSourceConstructionProgram::from_components(vec![a, b]).unwrap();
        let foreign = HostSourceConstructionProgram::from_components(vec![a, b]).unwrap();
        let admitted = if foreign_program { foreign.facts() }
            else { HostSourceConstructionFacts::new(32, 2, 0).unwrap() };
        assert_eq!(admitted.capacity_bytes(), selected.facts().capacity_bytes());
        assert_eq!(admitted.maximum_attempts(), selected.facts().maximum_attempts());
        assert_eq!(admitted.maximum_partitions(), selected.facts().maximum_partitions());
        if foreign_program { assert_eq!(admitted.protected_bytes(), selected.facts().protected_bytes()); }
        let Some(quote) = source_request_with_facts(&pool, admitted) else { return; };
        let (reservation, run, accepted) = accept(&pool, quote);
        let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &reservation).unwrap();
        let held = span.protected_host_bytes();
        let mut host = span.take_host_destinations().unwrap().unwrap();
        let source = host.take_source_constructions().unwrap();
        let error = source.partition_program(&selected).unwrap_err();
        assert!(matches!(std::error::Error::source(&error).and_then(|cause|cause.downcast_ref::<WorkingMemoryError>()),
            Some(WorkingMemoryError::IdentityMismatch)));
        drop((span, host, reservation, run, storage));
        assert_eq!(pool.used_bytes().unwrap(), held, "refused partition retains its actual original bank");
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn source_program_checks_account_health_before_extracting_an_unspent_component() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let storage = pool.register_storage([(1u32, 64)]).unwrap();
    let facts = HostSourceConstructionFacts::new(12, 1, 1).unwrap();
    let program = HostSourceConstructionProgram::from_components(vec![facts, facts]).unwrap();
    let Some(quote) = source_request_with_facts(&pool, program.facts()) else { return; };
    let (reservation, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &reservation).unwrap();
    let held = span.protected_host_bytes();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut components = host.take_source_constructions().unwrap().partition_program(&program).unwrap();
    let mut first = components.take(0).unwrap();
    let receipt = first.try_debit(12).unwrap();
    drop(run);
    assert!(matches!(components.take(1), Err(WorkingMemoryError::ExecutionFenced)));
    assert!(matches!(components.take(1), Err(WorkingMemoryError::ExecutionFenced)));
    assert_eq!((first.remaining_bytes(), first.remaining_attempts()), (0, 0));
    drop((span, host, reservation, storage, first, receipt));
    assert_eq!(pool.used_bytes().unwrap(), held, "unissued component remains in the original program owner");
    drop(components);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
