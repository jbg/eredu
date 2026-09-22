//! Host destinations funded by the enclosing ordinary physical allocation scope.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::expert::{AddressableChunkCensus, AddressableChunkPlan};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};
mod compact;
pub(crate) use compact::compact_transport_control_bytes;

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("ordinary bank source identity or parameter revision differs")]
    Identity,
    #[error("ordinary bank source population or layout overflowed")]
    Overflow,
    #[error("ordinary bank host destination funding: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("ordinary bank host destination allocation: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("ordinary bank source loan: {0}")]
    Source(#[source] AddressableSourceFailure),
    #[error("ordinary compact member rows: {0}")]
    Compact(#[source] super::super::parameters::CompactRowsError),
    #[error("ordinary compact consumer: {0}")]
    Consumer(#[source] Error),
}

#[derive(thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _funding: HostMetadataFunding,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryBankHostFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
fn failure(cause: Cause, funding: &HostMetadataFunding) -> AddressableParameterBankError {
    AddressableParameterBankError::HostMetadata(eredu_nn::Error::backend_retained_source(Failure {
        cause,
        _funding: funding.clone(),
    }))
}

/// An authenticated source and its originating request's host payer.
///
/// This descriptor grants no native execution, completion or allocation scope.
/// The ordinary invocation owns those authorities independently of this payer.
#[derive(Clone)]
pub(crate) struct OrdinaryBankHostSource {
    binding: IndexedBankSource,
    revision: u64,
    census: AddressableChunkCensus,
    funding: HostMetadataFunding,
}
impl OrdinaryBankHostSource {
    fn failure(&self, cause: Cause) -> Error {
        failure(cause, &self.funding).into()
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<AddressableChunkPlan>(),
            size_of::<Option<(usize, u64)>>(),
            eredu_nn::Error::retained_source_construction_bytes::<Failure>()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// The selected immutable chunk policy is checked against this actual bank.
    pub(crate) fn new(
        binding: &IndexedBankSource,
        census: AddressableChunkCensus,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let fail = |cause| Error::from(failure(cause, funding));
        funding
            .reserve_metadata(Self::control_bytes().ok_or_else(|| fail(Cause::Overflow))?)
            .map_err(|cause| fail(Cause::Funding(cause)))?;
        let revision = binding
            .with_workspace_source(funding, |source| {
                let (members, maximum) = source
                    .unit_population(census.bank(), census.unit())
                    .ok_or_else(|| fail(Cause::Identity))?;
                let plan = AddressableChunkPlan::new(
                    census.total_rows(),
                    census.routes(),
                    members,
                    census.access(),
                    Some(maximum),
                    binding.prefill_compact_bank_target_bytes(),
                )
                .map_err(|_| fail(Cause::Identity))?;
                if plan != census.plan() || !source.same_source(binding.storage()) {
                    return Err(fail(Cause::Identity));
                }
                Ok(source.parameter_revision())
            })
            .map_err(|cause| fail(Cause::Source(cause)))??;
        Ok(Self {
            binding: binding.clone(),
            revision,
            census,
            funding: funding.clone(),
        })
    }

    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(crate) fn census(&self) -> AddressableChunkCensus {
        self.census
    }
    pub(crate) fn same_binding(&self, binding: &IndexedBankSource) -> bool {
        self.binding.same_binding(binding)
    }
    /// Finite host destinations of the demand/snapshot worker. The manager's
    /// controller, source materialization and transfer owners remain separate.
    pub(crate) fn acquisition_metadata_bytes(&self) -> Result<usize, Error> {
        self.binding
            .with_workspace_source(&self.funding, |source| {
                if source.parameter_revision() != self.revision {
                    return Err(self.failure(Cause::Identity));
                }
                let members = self.census.maximum_members();
                let maximum_id = source
                    .unit_members(self.census.bank(), self.census.unit())
                    .map(|(key, _)| key.unit_id_length())
                    .max()
                    .ok_or_else(|| self.failure(Cause::Identity))?;
                let values = [
                    acquisition_control_bytes(),
                    vector_bytes::<(ParameterBankKey, u64)>(members),
                    vector_bytes::<ParameterBankKey>(members),
                    vector_bytes::<u64>(members),
                    vector_bytes::<(OffloadUnitId, u64)>(members).and_then(|n| n.checked_mul(2)),
                    vector_bytes::<(&OffloadUnitId, Option<u64>, Option<u64>)>(
                        source.resident_unit_population(),
                    )
                    .and_then(|n| n.checked_mul(2)),
                    vector_bytes::<u8>(maximum_id)
                        .and_then(|n| n.checked_mul(members))
                        .and_then(|n| n.checked_mul(2)),
                ];
                values
                    .into_iter()
                    .try_fold(0usize, |bytes, value| bytes.checked_add(value?))
                    .ok_or_else(|| self.failure(Cause::Overflow))
            })
            .map_err(|cause| self.failure(Cause::Source(cause)))?
    }
    pub(crate) fn validate_bank(&self, bank: &SharedAddressableParameterBank) -> Result<(), Error> {
        self.binding
            .with_workspace_source(&self.funding, |source| {
                if !source.same_source(bank) || source.parameter_revision() != self.revision {
                    return Err(Error::from(failure(Cause::Identity, &self.funding)));
                }
                Ok(())
            })
            .map_err(|cause| Error::from(failure(Cause::Source(cause), &self.funding)))?
    }

    pub(crate) fn acquire(
        &self,
        bank: &SharedAddressableParameterBank,
        entries: &[(ParameterBankKey, u64)],
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, Error> {
        self.validate_bank(bank)?;
        let expected_pass = match self.census.access() {
            ParameterBankAccess::Bulk => BankAccessClass::Bulk,
            ParameterBankAccess::Incremental => BankAccessClass::Incremental,
            _ => return Err(failure(Cause::Identity, &self.funding).into()),
        };
        let selections = entries
            .iter()
            .try_fold(0u64, |sum, (_, count)| sum.checked_add(*count))
            .ok_or_else(|| failure(Cause::Overflow, &self.funding))?;
        if entries.len() > self.census.maximum_members()
            || entries.iter().any(|(key, _)| {
                key.bank() != self.census.bank() || key.unit() != self.census.unit()
            })
            || pass != expected_pass
            || self.census.values().and_then(|n| u64::try_from(n).ok()) != Some(selections)
        {
            return Err(failure(Cause::Identity, &self.funding).into());
        }
        // Parameter publication is serialized with the same native cache.
        let bank = bank
            .inner
            .lock()
            .map_err(|_| failure(Cause::Identity, &self.funding))?;
        if bank.parameter_revision != self.revision {
            return Err(failure(Cause::Identity, &self.funding).into());
        }
        let mut acquisition =
            bank.acquire_entry_demand_with_host(entries, pass, stream, Some(&self.funding))?;
        acquisition.ordinary = Some(self.clone());
        Ok(acquisition)
    }
}

/// Exact empty destination; its producer cannot grow past the selected count.
pub(super) fn vector<T>(
    count: usize,
    funding: Option<&HostMetadataFunding>,
) -> Result<Vec<T>, AddressableParameterBankError> {
    let bytes = vector_bytes::<T>(count).ok_or(AddressableParameterBankError::ByteOverflow)?;
    if let Some(funding) = funding {
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| failure(Cause::Funding(cause), funding))?;
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|cause| match funding {
            Some(funding) => failure(Cause::Allocation(cause), funding),
            None => AddressableParameterBankError::HostAllocation(cause),
        })?;
    Ok(values)
}

pub(super) fn vector_bytes<T>(count: usize) -> Option<usize> {
    let frames = [
        Layout::array::<T>(count).ok()?.size(),
        size_of::<Vec<T>>(),
        size_of::<Result<Vec<T>, AddressableParameterBankError>>(),
        size_of::<std::collections::TryReserveError>(),
        size_of::<(usize, &HostMetadataFunding)>(),
        eredu_nn::Error::retained_source_construction_bytes::<Failure>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

pub(super) fn acquisition_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<AcquiredParameterGroups>(),
        size_of::<ResidentTransfer>(),
        size_of::<Result<AcquiredParameterGroups, AddressableParameterBankError>>(),
        size_of::<[ResidentSnapshot<'_>; 2]>(),
        size_of::<[u64; 12]>(),
        size_of::<(
            Instant,
            Duration,
            Option<usize>,
            &AddressableParameterBank,
            &Stream,
        )>(),
        size_of::<std::sync::MutexGuard<'_, ParameterBankStatisticsTable>>(),
        size_of::<std::sync::MutexGuard<'_, AddressableParameterBank>>(),
        size_of::<(ParameterBankKey, u64, OffloadUnitId)>(),
        size_of::<Result<(), ResidencyError>>(),
        size_of::<Result<ResidentTransfer, ResidencyError>>(),
        size_of::<(
            &[(ParameterBankKey, u64)],
            BankAccessClass,
            &HostMetadataFunding,
        )>(),
        size_of::<[Vec<(OffloadUnitId, u64)>; 2]>(),
        size_of::<Vec<(ParameterBankKey, u64)>>(),
        size_of::<Vec<ParameterBankKey>>(),
        size_of::<Vec<u64>>(),
        eredu_nn::Error::retained_source_construction_bytes::<Failure>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

pub(super) fn fund_acquisition(
    funding: Option<&HostMetadataFunding>,
) -> Result<(), AddressableParameterBankError> {
    if let Some(funding) = funding {
        funding
            .reserve_metadata(
                acquisition_control_bytes().ok_or(AddressableParameterBankError::ByteOverflow)?,
            )
            .map_err(|cause| failure(Cause::Funding(cause), funding))?;
    }
    Ok(())
}

pub(super) fn unit_id(
    key: ParameterBankKey,
    funding: Option<&HostMetadataFunding>,
) -> Result<OffloadUnitId, AddressableParameterBankError> {
    let len = key.unit_id_length();
    let mut bytes = vector::<u8>(len, funding)?;
    bytes.resize(len, 0);
    key.write_unit_id(&mut bytes)
        .ok_or(AddressableParameterBankError::ByteOverflow)?;
    let text = String::from_utf8(bytes).expect("canonical parameter-bank identity is ASCII");
    Ok(OffloadUnitId::new(text).expect("canonical parameter-bank identity is nonempty"))
}

pub(super) fn clone_unit_id(
    id: &OffloadUnitId,
    funding: Option<&HostMetadataFunding>,
) -> Result<OffloadUnitId, AddressableParameterBankError> {
    let mut bytes = vector::<u8>(id.as_str().len(), funding)?;
    bytes.extend_from_slice(id.as_str().as_bytes());
    let text = String::from_utf8(bytes).expect("source identifier is UTF-8");
    Ok(OffloadUnitId::new(text).expect("source identifier is nonempty"))
}
