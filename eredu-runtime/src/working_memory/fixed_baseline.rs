//! Fixed allocations owned by one explicitly initialized physical-domain ledger.
use super::{MemoryLedger, Pool, WorkingMemoryError, qualified_storage};

impl MemoryLedger {
    /// Managed heap extent retained by one explicitly initialized empty ledger.
    /// Includes topology, dense transaction storage, shared accounting identity,
    /// and qualified mutex and condition-variable backing. Cloned ledger handles
    /// allocate none of these.
    /// Dynamic storage registrations/accounts and thread initialization are not
    /// included. An unsupported compiler/target returns `UnknownBound`.
    pub fn fixed_owner_bytes(
        topology: &eredu_core::MemoryTopology,
        limits: &eredu_core::MemoryLimits,
        baseline: &eredu_core::DomainMemoryRequirements,
    ) -> Result<u64, WorkingMemoryError> {
        limits.validate(topology)?;
        baseline.validate(topology)?;
        let per_domain = std::mem::size_of::<super::domains::FixedDomain>()
            .checked_add(std::mem::size_of::<super::domains::DomainUsage>())
            .and_then(|n| n.checked_add(std::mem::size_of::<eredu_core::MemoryDomainId>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<eredu_core::DomainMemoryCharge>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<eredu_core::MemoryLimit>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        let arrays = topology
            .len()
            .checked_mul(per_domain)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        [
            qualified_storage::shared_bytes::<Pool>()?,
            qualified_storage::shared_bytes::<()>()?,
            qualified_storage::shared_layout_bytes(
                eredu_core::SharedStorageAccountingId::shared_payload_layout(),
            )?,
            qualified_storage::shared_bytes::<eredu_core::MemoryTopology>()?,
            qualified_storage::shared_bytes::<eredu_core::MemoryPlacement>()?,
            pal_mutex_bytes().ok_or(WorkingMemoryError::UnknownBound)?,
            pal_condvar_bytes().ok_or(WorkingMemoryError::UnknownBound)?,
            topology.backing_bytes()?,
            limits.backing_bytes()?,
            baseline.backing_bytes()?,
            arrays,
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or(WorkingMemoryError::Overflow)
    }
}

pub(super) fn pal_mutex_bytes() -> Option<u64> {
    // Existing qualification pins the actual Darwin libstd Mutex selector,
    // OnceBox producer and PAL UnsafeCell<pthread_mutex_t> backing. Use the
    // target host type, not an invented private std layout or observed malloc.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        u64::try_from(std::mem::size_of::<libc::pthread_mutex_t>()).ok()
    }
    // The audited Rust 1.98 Linux and Windows selectors use inline futex
    // storage; kernel-private waiting resources are not Eredu allocations.
    #[cfg(all(
        target_arch = "x86_64",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        Some(0)
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(
            target_arch = "x86_64",
            any(target_os = "linux", target_os = "windows")
        )
    )))]
    {
        None
    }
}

pub(super) fn pal_condvar_bytes() -> Option<u64> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        u64::try_from(std::mem::size_of::<libc::pthread_cond_t>()).ok()
    }
    #[cfg(all(
        target_arch = "x86_64",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        Some(0)
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(
            target_arch = "x86_64",
            any(target_os = "linux", target_os = "windows")
        )
    )))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::*;
    use std::sync::Arc;

    #[test]
    fn initialized_owner_metadata_is_charged_once_across_aliases() {
        let topology = Arc::new(
            MemoryTopology::new(vec![MemoryDomainDescription {
                name: "host".into(),
                locations: vec![MemoryLocation::Host],
            }])
            .unwrap(),
        );
        let limits = MemoryLimits::unlimited(&topology);
        let mut baseline = DomainMemoryRequirements::zero(&topology);
        baseline
            .add_allocation(
                7,
                &MemoryPlacement::fixed(&topology, topology.host_domain()).unwrap(),
            )
            .unwrap();
        let bytes = MemoryLedger::fixed_owner_bytes(&topology, &limits, &baseline).unwrap();
        assert!(bytes > 0);
        let pool = MemoryLedger::new(Arc::clone(&topology), limits, baseline.clone()).unwrap();
        let alias = pool.clone();
        assert!(pool.same_ledger(&alias));
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            bytes + 7
        );
        drop(pool);
        assert_eq!(
            alias.snapshot().unwrap().domains[0].current_charge_bytes,
            bytes + 7
        );
        let insufficient = MemoryLimits::resolve(
            &topology,
            [(topology.host_domain(), MemoryLimit::Finite(bytes + 6))],
        )
        .unwrap();
        assert!(matches!(
            MemoryLedger::new(topology, insufficient, baseline),
            Err(WorkingMemoryError::Domain(
                MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
    }
}
