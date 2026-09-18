//! Exact borrowed source of the already selected shared addressable cache.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::parameter_operations::PreparedBankParameterMember;
use std::{mem::{size_of, size_of_val}, sync::{MutexGuard, TryLockError}};

#[derive(Debug, thiserror::Error)]
pub(crate) enum AddressableSourceCause {
    #[error("addressable source cache is busy")]
    Busy,
    #[error("addressable source cache is poisoned")]
    Poisoned,
    #[error("addressable source has no completed selected member declarations")]
    Missing,
    #[error("addressable source metadata accounting overflowed")]
    Overflow,
    #[error("addressable source metadata: {0}")]
    Funding(#[source] HostMetadataFundingError),
}

/// A failed source query keeps its original metadata reservation alive.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(crate) struct AddressableSourceFailure {
    #[source]
    cause: AddressableSourceCause,
    source: SharedAddressableParameterBank,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for AddressableSourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddressableSourceFailure").field("cause", &self.cause)
            .field("scope", &self.source.scope).finish_non_exhaustive()
    }
}

/// A lexical view of one real pool and its optional bank restriction.
///
/// The enclosing try-lock prevents parameter replacement during the census.
/// This is descriptive metadata, not an acquisition, completion or copy grant.
pub(crate) struct AddressableBankSourceLoan<'a> {
    bank: &'a AddressableParameterBank,
    owner: &'a SharedAddressableParameterBank,
}
impl AddressableBankSourceLoan<'_> {
    /// Direct ordered traversal; no cloned member table or repeated rescans.
    pub(crate) fn members(&self) -> impl Iterator<Item = &PreparedBankParameterMember> {
        self.bank.parameter_members.iter().filter(|member|
            self.owner.scope.is_none_or(|scope| scope == member.key.bank()))
    }
    /// Exact unique member population and physical maximum for one selected unit.
    pub(crate) fn unit_population(&self, bank: usize, unit: usize) -> Option<(usize, u64)> {
        if self.owner.scope.is_some_and(|scope| scope != bank) { return None; }
        self.bank.effective_member_bytes.iter()
            .filter(|(key, _)| key.bank() == bank && key.unit() == unit)
            .try_fold((0usize, 0u64), |(count, maximum), (_, &bytes)|
                Some((count.checked_add(1)?, maximum.max(bytes))))
            .filter(|(count, _)| *count != 0)
    }
    /// Actual owner-local ordering and physical byte geometry, without a copied map.
    pub(crate) fn unit_members(&self, bank: usize, unit: usize)
        -> impl Iterator<Item = (ParameterBankKey, u64)> + '_ {
        self.bank.effective_member_bytes.iter().filter(move |(key, _)|
            key.bank() == bank && key.unit() == unit
                && self.owner.scope.is_none_or(|scope| scope == bank))
            .map(|(key, bytes)| (*key, *bytes))
    }
    /// Revision of the actual atomically published replacement source table.
    pub(crate) fn parameter_revision(&self) -> u64 { self.bank.parameter_revision }
    /// Exact fixed mutable telemetry table owned by this same loaded pool.
    pub(crate) fn statistics_storage_bytes(&self)->Option<usize> {
        self.bank.statistics.try_lock().ok()?.storage_bytes()
    }
    /// Actual post-replacement physical byte geometry for this selected scope.
    pub(crate) fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        if self.owner.scope.is_some_and(|scope| scope != key.bank()) { return None; }
        self.bank.effective_member_bytes.get(&key).copied()
    }
    /// The shared physical cache remains one owner across scoped bank handles.
    pub(crate) fn same_pool(&self, actual: &SharedAddressableParameterBank) -> bool {
        Arc::ptr_eq(&self.owner.inner, &actual.inner)
    }
    /// Includes logical bank restriction as well as physical pool identity.
    pub(crate) fn same_source(&self, actual: &SharedAddressableParameterBank) -> bool {
        self.same_pool(actual) && self.owner.scope == actual.scope
    }
    /// Borrows the actual manager for its existing source-only inspectors.
    /// Native acquisition still requires its own separately admitted operation.
    pub(crate) fn manager(&self) -> &ResidencyManager { &self.bank.manager }
    /// Exact compact-bank scratch policy held by the same physical cache.
    pub(crate) fn scratch_bytes(&self) -> u64 { self.bank.scratch_limit }
    /// Actual replacement and its original member-row ordinal, if present.
    /// Only a member borrowed from this scope may name a replacement source.
    pub(crate) fn replacement(&self, member: &PreparedBankParameterMember) -> Option<(&MlxTensor, usize)> {
        if !self.members().any(|actual| std::ptr::eq(actual, member)) { return None; }
        let value = self.bank.parameter_replacements.get(&member.parameter)?;
        let row = self.bank.parameter_members.iter().filter(|candidate|
            candidate.parameter == member.parameter && candidate.key < member.key).count();
        Some((value, row))
    }
}
impl SharedAddressableParameterBank {
    /// Runs a counted metadata census under one nonblocking immutable pool loan.
    /// The callback cannot return borrowed members or keep the lock alive.
    pub(crate) fn with_workspace_source<T, E, F>(&self, funding: &HostMetadataFunding,
        inspect: F) -> Result<Result<T, E>, AddressableSourceFailure>
    where F: for<'loan> FnOnce(AddressableBankSourceLoan<'loan>) -> Result<T, E> {
        let failure = |cause| AddressableSourceFailure { cause, source: self.clone(), funding: funding.clone() };
        let frames = [size_of::<Self>(), size_of::<AddressableBankSourceLoan<'_>>(),
            size_of::<MutexGuard<'_, AddressableParameterBank>>(),
            size_of::<TryLockError<MutexGuard<'_, AddressableParameterBank>>>(),
            size_of::<F>(), size_of::<Result<T, E>>(),
            size_of::<Result<Result<T, E>, AddressableSourceFailure>>(),
            size_of::<AddressableSourceFailure>(), size_of::<AddressableSourceCause>(),
            size_of::<HostMetadataFunding>(), size_of::<Option<usize>>()];
        let bytes = frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| failure(AddressableSourceCause::Overflow))?;
        funding.reserve_metadata(bytes)
            .map_err(|cause| failure(AddressableSourceCause::Funding(cause)))?;
        let bank = self.inner.try_lock().map_err(|cause| failure(match cause {
            TryLockError::WouldBlock => AddressableSourceCause::Busy,
            TryLockError::Poisoned(_) => AddressableSourceCause::Poisoned,
        }))?;
        let source = AddressableBankSourceLoan { bank: &bank, owner: self };
        if source.members().next().is_none() {
            return Err(failure(AddressableSourceCause::Missing));
        }
        Ok(inspect(source))
    }
}
