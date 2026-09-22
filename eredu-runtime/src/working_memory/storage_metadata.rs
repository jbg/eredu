//! Host preparation for storage registration, without execution permission.
use super::*;
use eredu_core::{
    HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority,
};
use std::sync::atomic::{AtomicU64, Ordering};

/// Host preparation authenticated to one ledger. Clones retain the same paid
/// controls; this owner cannot create an execution or native allocation scope.
#[derive(Debug, Clone)]
pub struct StorageMetadataFunding {
    pool: MemoryLedger,
    funding: HostMetadataFunding,
}
impl StorageMetadataFunding {
    pub(super) fn pool(&self) -> &MemoryLedger {
        &self.pool
    }

    pub(super) fn from_construction(pool: &MemoryLedger, funding: &HostMetadataFunding) -> Self {
        Self {
            pool: pool.clone(),
            funding: funding.clone(),
        }
    }

    pub(super) fn from_account(
        pool: &MemoryLedger,
        id: u64,
    ) -> Result<Self, HostMetadataFundingError> {
        let funding = HostMetadataFunding::new(StorageMetadataAccount {
            pool: pool.clone(),
            bytes: AtomicU64::new(0),
            origin: Some(id),
            constructor_fixed: None,
        })?;
        Ok(Self {
            pool: pool.clone(),
            funding,
        })
    }

    // The factory creates this role once, before handing out its native scope.
    // Its moved token authenticates the fixed owner through final destruction;
    // a different allocation from this account cannot replace that identity.
    pub(super) fn from_shared_constructor_account(
        pool: &MemoryLedger,
        id: u64,
    ) -> Result<Self, HostMetadataFundingError> {
        let funding = HostMetadataFunding::new(StorageMetadataAccount {
            pool: pool.clone(),
            bytes: AtomicU64::new(0),
            origin: Some(id),
            constructor_fixed: Some(ConstructorFixedMetadata { account: id }),
        })?;
        Ok(Self {
            pool: pool.clone(),
            funding,
        })
    }
    /// Exact host owner allowance, including its retained funding shell and
    /// named constructor transports. No payload allocation or execution occurs.
    pub fn host_owner_bytes(bytes: usize) -> Result<usize, HostMetadataFundingError> {
        let frames = [
            std::mem::size_of::<&Self>(),
            std::mem::size_of::<usize>() * 2,
            std::mem::size_of::<Result<HostPreparationAuthority, HostMetadataFundingError>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        frames.into_iter().try_fold(
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                .and_then(|n| n.checked_add(std::mem::size_of_val(&frames)))
                .and_then(|n| n.checked_add(bytes))
                .ok_or(HostMetadataFundingError::Overflow)?,
            |total, part| {
                total
                    .checked_add(part)
                    .ok_or(HostMetadataFundingError::Overflow)
            },
        )
    }

    /// Pays before constructing a host-only owner and retains that payment
    /// through its last alias. This grants no invocation or native permission.
    pub fn prepare_host_owner(
        &self,
        bytes: usize,
    ) -> Result<HostPreparationAuthority, HostMetadataFundingError> {
        let bytes = Self::host_owner_bytes(bytes)?;
        self.funding.reserve_metadata(bytes)?;
        Ok(HostPreparationAuthority::retain(self.funding.clone()))
    }

    pub(crate) fn validate_ledger(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.pool.same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Borrows the same checked host metadata payer for a producer's finite
    /// collections and constructor records. Charges consume the originating
    /// scope's assigned host allowance when this owner came from a scope;
    /// this loan grants no native allocation or execution authority.
    pub fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
}

fn wrapper_bytes() -> Result<u64, HostMetadataFundingError> {
    u64::try_from(
        std::mem::size_of::<StorageMetadataFunding>()
            .checked_add(std::mem::size_of::<
                Result<StorageMetadataFunding, HostMetadataFundingError>,
            >())
            .ok_or(HostMetadataFundingError::Overflow)?,
    )
    .map_err(|_| HostMetadataFundingError::Overflow)
}

/// Registered metadata has no invocation, work scope or unquoted execution
/// authority. Its cumulative charge retires after the closed funding shells.
#[derive(Debug)]
struct StorageMetadataAccount {
    pool: MemoryLedger,
    bytes: AtomicU64,
    origin: Option<u64>,
    constructor_fixed: Option<ConstructorFixedMetadata>,
}

#[derive(Debug)]
struct ConstructorFixedMetadata {
    account: u64,
}

impl HostMetadataAccount for StorageMetadataAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let bytes = u64::try_from(bytes).map_err(|_| HostMetadataFundingError::Overflow)?;
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| HostMetadataFundingError::Poisoned)?;
        let previous = self.bytes.load(Ordering::Relaxed);
        let bytes = if previous == 0 {
            bytes
                .checked_add(wrapper_bytes()?)
                .ok_or(HostMetadataFundingError::Overflow)?
        } else {
            bytes
        };
        let total = previous
            .checked_add(bytes)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let slot = self.pool.0.host_slot;
        let current = &usage.domains[slot];
        let used = self.pool.0.domains[slot]
            .existing
            .checked_add(current.registered)
            .and_then(|n| n.checked_add(current.reserved))
            .ok_or(HostMetadataFundingError::Overflow)?;
        let limit = self
            .pool
            .0
            .domain_capacity(&usage, self.pool.topology().host_domain(), None)
            .map_err(failure)?;
        let charge = limit.check(
            self.pool.topology().host_domain(),
            used,
            if self.origin.is_some() { 0 } else { bytes },
        )?;
        let registered = current
            .registered
            .checked_add(bytes)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let metadata = current
            .registry_metadata
            .checked_add(bytes)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let assigned = if let Some(id) = self.origin {
            let state = usage
                .funding
                .get(&id)
                .ok_or(HostMetadataFundingError::Unavailable)?;
            if previous == 0 {
                if let Some(fixed) = &self.constructor_fixed {
                    if fixed.account != id {
                        return Err(HostMetadataFundingError::Unavailable);
                    }
                    state.validate_shared_constructor_fixed().map_err(failure)?;
                }
            }
            let (available, allocations) = state
                .storage_metadata_balance(bytes, previous == 0)
                .map_err(failure)?;
            if bytes > available {
                return Err(HostMetadataFundingError::DomainAllowance {
                    domain: self.pool.topology().host_domain(),
                    required: bytes,
                    available,
                });
            }
            Some((
                id,
                state.domains[slot]
                    .remaining
                    .checked_sub(bytes)
                    .ok_or(HostMetadataFundingError::Overflow)?,
                current
                    .reserved
                    .checked_sub(bytes)
                    .ok_or(HostMetadataFundingError::Overflow)?,
                allocations,
            ))
        } else {
            None
        };
        // No allocation, callback or destruction occurs in this commit.
        self.bytes.store(total, Ordering::Relaxed);
        usage.domains[slot].registered = registered;
        usage.domains[slot].registry_metadata = metadata;
        if let Some((id, remaining, reserved, allocations)) = assigned {
            let state = usage
                .funding
                .get_mut(&id)
                .expect("validated funding origin");
            state.domains[slot].remaining = remaining;
            state.allocations = allocations;
            if previous == 0 && self.constructor_fixed.is_some() {
                state.install_shared_constructor_fixed();
            }
            usage.domains[slot].reserved = reserved;
        } else {
            usage.domains[slot].peak = usage.domains[slot].peak.max(charge);
        }
        Ok(())
    }
}

pub(super) fn failure(error: WorkingMemoryError) -> HostMetadataFundingError {
    match error {
        WorkingMemoryError::Domain(cause) => HostMetadataFundingError::Domain(cause),
        WorkingMemoryError::DomainAllowanceExceeded {
            domain,
            required_bytes,
            available_bytes,
        } => HostMetadataFundingError::DomainAllowance {
            domain,
            required: required_bytes,
            available: available_bytes,
        },
        WorkingMemoryError::Overflow => HostMetadataFundingError::Overflow,
        WorkingMemoryError::Poisoned => HostMetadataFundingError::Poisoned,
        _ => HostMetadataFundingError::Unavailable,
    }
}

impl Drop for StorageMetadataAccount {
    fn drop(&mut self) {
        // Unwinding does not prove that all paid producer payloads retired.
        let bytes = *self.bytes.get_mut();
        if bytes == 0 {
            return;
        }
        let mut usage = funding::lock_for_retirement(&self.pool);
        if std::thread::panicking() {
            if let Some(id) = self.origin {
                usage
                    .funding
                    .get_mut(&id)
                    .expect("retained metadata origin")
                    .quarantined = true;
            }
            return;
        }
        let slot = usage.host_slot;
        usage.domains[slot].registry_metadata = usage.domains[slot]
            .registry_metadata
            .checked_sub(bytes)
            .expect("retained storage metadata charge");
        if let Some(id) = self.origin {
            if let Some(fixed) = &self.constructor_fixed {
                assert_eq!(fixed.account, id);
                usage
                    .funding
                    .get_mut(&id)
                    .expect("retained constructor metadata origin")
                    .retire_shared_constructor_fixed();
            }
            funding::retire_allocation(&mut usage, id, bytes);
        } else {
            usage.domains[slot].registered = usage.domains[slot]
                .registered
                .checked_sub(bytes)
                .expect("retained storage metadata charge");
        }
        drop(usage);
        funding::drain_accounts(&self.pool);
    }
}

impl MemoryLedger {
    /// Starts host preparation for registration of existing storage. Each
    /// constructor calls `reserve_metadata` before allocating and retains this
    /// owner through its metadata. Existing unquoted loading owners may use
    /// this account; it grants no execution authority or allocation exemption.
    /// Physical host limits include every other live ledger charge.
    pub fn prepare_storage_metadata(
        &self,
    ) -> Result<StorageMetadataFunding, HostMetadataFundingError> {
        let funding = HostMetadataFunding::new(StorageMetadataAccount {
            pool: self.clone(),
            bytes: AtomicU64::new(0),
            origin: None,
            constructor_fixed: None,
        })?;
        Ok(StorageMetadataFunding {
            pool: self.clone(),
            funding,
        })
    }

    /// Host constructor quotation for the closed storage metadata account.
    pub fn storage_metadata_control_bytes() -> Result<u64, WorkingMemoryError> {
        HostMetadataFunding::constructor_bytes::<StorageMetadataAccount>()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .and_then(|bytes| bytes.checked_add(wrapper_bytes().ok()?))
            .ok_or(WorkingMemoryError::Overflow)
    }
}

impl WorkingMemoryFundingScope {
    /// Pays registration metadata from this account's unprotected fixed host
    /// allowance. Conversion preserves the complete live physical charge.
    /// The returned owner grants no native scope or additional allocation cap.
    pub fn prepare_storage_metadata(
        &self,
    ) -> Result<StorageMetadataFunding, HostMetadataFundingError> {
        self.validate_native_purpose().map_err(failure)?;
        self.prepare_storage_metadata_account()
    }
    fn prepare_storage_metadata_account(
        &self,
    ) -> Result<StorageMetadataFunding, HostMetadataFundingError> {
        StorageMetadataFunding::from_account(self.pool(), self.id)
    }
}

impl funding::WorkingMemoryDecoderHostScope {
    pub(super) fn prepare_storage_metadata(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<StorageMetadataFunding, HostMetadataFundingError> {
        self.storage_metadata_scope(execution)
            .map_err(failure)?
            .prepare_storage_metadata_account()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::working_memory::memory_fixture::host_ledger;

    #[test]
    fn preparation_precedes_allocation_and_preserves_unquoted_exclusion() {
        let controls = MemoryLedger::storage_metadata_control_bytes().unwrap();
        let unquoted_controls = MemoryLedger::unquoted_owner_control_bytes().unwrap();
        let pool = host_ledger(controls + 16 + unquoted_controls, 0).unwrap();
        let baseline = pool.snapshot().unwrap();
        let unquoted = pool.acquire_unquoted().unwrap();
        let funding = pool.prepare_storage_metadata().unwrap();
        funding.funding().reserve_metadata(16).unwrap();
        let charged = pool.snapshot().unwrap();
        assert_eq!(
            charged.domains[0].current_charge_bytes - baseline.domains[0].current_charge_bytes,
            controls + 16 + unquoted_controls
        );
        assert!(matches!(
            funding.funding().reserve_metadata(1),
            Err(HostMetadataFundingError::Domain(
                MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(pool.snapshot().unwrap(), charged);
        assert_eq!(charged.reservations, 0);
        let alias = funding.clone();
        drop(funding);
        assert_eq!(pool.snapshot().unwrap(), charged);
        assert!(matches!(
            pool.reserve(
                &InferenceExecutionIdentity::default(),
                &memory_fixture::host_admission(&pool, 0)
            ),
            Err(WorkingMemoryError::UnknownBound)
        ));
        drop(alias);
        drop(unquoted);
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            baseline.domains[0].current_charge_bytes
        );
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }

    #[test]
    fn assigned_preparation_converts_allowance_and_retires_after_every_alias() {
        let pool = host_ledger(u64::MAX, 0).unwrap();
        let initial = pool.snapshot().unwrap();
        let bytes = MemoryLedger::storage_metadata_control_bytes().unwrap() + 16;
        let admission = memory_fixture::host_admission(&pool, bytes);
        let (reservation, run) = pool
            .reserve(&InferenceExecutionIdentity::default(), &admission)
            .unwrap()
            .into_funding()
            .unwrap();
        let scope = run.scope().unwrap();
        let before = pool.snapshot().unwrap();
        let metadata = scope.prepare_storage_metadata().unwrap();
        metadata.funding().reserve_metadata(16).unwrap();
        let after = pool.snapshot().unwrap();
        assert_eq!(
            before.domains[0].current_charge_bytes,
            after.domains[0].current_charge_bytes
        );
        assert_eq!(
            before.domains[0].historical_peak_bytes,
            after.domains[0].historical_peak_bytes
        );
        assert_eq!(after.domains[0].registry_metadata_bytes, bytes);
        assert!(matches!(
            metadata.funding().reserve_metadata(1),
            Err(HostMetadataFundingError::DomainAllowance {
                required: 1,
                available: 0,
                ..
            })
        ));
        assert_eq!(pool.snapshot().unwrap(), after);
        let alias = metadata.clone();
        scope.certify().unwrap();
        drop(run);
        drop(reservation);
        drop(metadata);
        assert_eq!(
            pool.snapshot().unwrap().domains[0].registry_metadata_bytes,
            bytes
        );
        drop(alias);
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            initial.domains[0].current_charge_bytes
        );
        assert_eq!(pool.snapshot().unwrap().funding_accounts, 0);
    }

    #[test]
    fn host_owner_exact_allowance_survives_all_aliases_without_execution_authority() {
        let payload = 37usize;
        let owner = StorageMetadataFunding::host_owner_bytes(payload).unwrap() as u64;
        let constructor = MemoryLedger::storage_metadata_control_bytes().unwrap();
        let pool = host_ledger(constructor + owner, 0).unwrap();
        let baseline = pool.snapshot().unwrap().domains[0].current_charge_bytes;
        let funding = pool.prepare_storage_metadata().unwrap();
        let custody = funding.prepare_host_owner(payload).unwrap();
        let charged = pool.snapshot().unwrap();
        assert_eq!(
            charged.domains[0].current_charge_bytes - baseline,
            constructor + owner
        );
        assert_eq!((charged.reservations, charged.funding_accounts), (0, 0));
        let alias = custody.clone();
        drop((funding, custody));
        assert_eq!(pool.snapshot().unwrap(), charged);
        drop(alias);
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            baseline
        );
    }

    #[test]
    fn host_owner_one_byte_short_and_overflow_refuse_before_constructing_custody() {
        let payload = 37usize;
        let owner = StorageMetadataFunding::host_owner_bytes(payload).unwrap() as u64;
        let constructor = MemoryLedger::storage_metadata_control_bytes().unwrap();
        let pool = host_ledger(constructor + owner - 1, 0).unwrap();
        let funding = pool.prepare_storage_metadata().unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            funding.prepare_host_owner(payload),
            Err(HostMetadataFundingError::Domain(
                MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
        assert!(matches!(
            funding.prepare_host_owner(usize::MAX),
            Err(HostMetadataFundingError::Overflow)
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
    }
}
