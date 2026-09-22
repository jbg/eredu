//! Domain-aware comparison for CPU and physically unified native fixtures.
pub(crate) fn host_budget_numbers(
    error: &eredu_runtime::working_memory::WorkingMemoryError,
) -> Option<(u64, u64)> {
    let eredu_runtime::working_memory::WorkingMemoryError::Domain(
        eredu_core::MemoryDomainError::BudgetExceeded {
            domain,
            requested_bytes,
            limit_bytes,
            existing_bytes,
        },
    ) = error
    else {
        return None;
    };
    assert_eq!(*domain, crate::memory_topology().unwrap().host_domain());
    Some((
        *requested_bytes,
        limit_bytes
            .checked_sub(*existing_bytes)
            .expect("existing charge fits prior ceiling"),
    ))
}
