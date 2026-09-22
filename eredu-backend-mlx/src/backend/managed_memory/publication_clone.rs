//! Independently paid native handles for prepared parameter publication.
use super::*;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{
    Array, PreparedArrayClone, PreparedArrayCloneCause, PreparedSubmissionGraphQuota,
    SubmissionGraphQuotaCause,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum CloneCause {
    #[error(transparent)]
    Arena(SubmissionGraphQuotaCause),
    #[error(transparent)]
    Handle(PreparedArrayCloneCause),
}
#[derive(Debug, thiserror::Error)]
#[error("prepared parameter handle failed: {cause}")]
struct CloneFailure {
    #[source]
    cause: CloneCause,
    _host: HostPreparationAuthority,
}

/// The native handle and its error paths retain their own paid metadata arena.
/// Neither preparation nor fill attaches metadata custody to the source backing.
pub(crate) struct PreparedPublicationClone {
    slot: PreparedArrayClone,
    host: HostPreparationAuthority,
}
impl PreparedPublicationClone {
    pub(crate) fn prepare(pool: &MemoryLedger) -> Result<Self, Error> {
        let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
        let capacity = PreparedArrayClone::arena_capacity().ok_or_else(overflow)?;
        let layout = PreparedSubmissionGraphQuota::<HostPreparationAuthority>::layout(capacity)
            .map_err(|_| overflow())?;
        let frames = [
            layout.total_bytes().ok_or_else(overflow)?,
            PreparedArrayClone::arena_control_bytes().ok_or_else(overflow)?,
            size_of::<Self>(),
            size_of::<CloneCause>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Array, Error>>(),
            size_of::<(&MemoryLedger, &Array, usize)>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<CloneFailure>()
                .ok_or_else(overflow)?,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(overflow)?;
        let funding = pool
            .prepare_storage_metadata()
            .map_err(|cause| Error::PrefillControl(metadata_memory_error(cause)))?;
        let host = funding
            .prepare_host_owner(bytes)
            .map_err(|cause| Error::PrefillControl(metadata_memory_error(cause)))?;
        let prepared = PreparedSubmissionGraphQuota::try_new(capacity, host.clone())
            .map_err(|failure| Self::failure(CloneCause::Arena(failure.cause()), &host))?;
        let arena = prepared
            .try_allocate()
            .map_err(|failure| Self::failure(CloneCause::Arena(failure.cause()), &host))?;
        let slot = PreparedArrayClone::try_prepare_in(&arena)
            .map_err(|cause| Self::failure(CloneCause::Handle(cause), &host))?;
        Ok(Self { slot, host })
    }
    pub(crate) fn fill(mut self, source: &Array) -> Result<Array, Error> {
        self.slot
            .fill_for_inspection(source)
            .map_err(|cause| Self::failure(CloneCause::Handle(cause), &self.host))
    }
    fn failure(cause: CloneCause, host: &HostPreparationAuthority) -> Error {
        Error::AllocationAttachment(eredu_core::BackendFailure::from_error(CloneFailure {
            cause,
            _host: host.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn charge(pool: &MemoryLedger) -> u64 {
        pool.snapshot()
            .unwrap()
            .domains
            .into_iter()
            .find(|row| row.domain == pool.topology().host_domain())
            .unwrap()
            .current_charge_bytes
    }
    #[test]
    fn discarded_publication_handles_refund_with_source_and_alias_still_live() {
        let source = Array::from_slice(&[3f32, -7., 11.], &[3]);
        let source_alias = source.clone();
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let before = charge(&pool);
        let unused = PreparedPublicationClone::prepare(&pool).unwrap();
        assert!(charge(&pool) > before);
        drop(unused);
        safemlx::reclaim_allocation_owners();
        assert_eq!(charge(&pool), before);
        let first = PreparedPublicationClone::prepare(&pool)
            .unwrap()
            .fill(&source)
            .unwrap();
        let one_handle = charge(&pool);
        assert!(one_handle > before);
        let second = PreparedPublicationClone::prepare(&pool)
            .unwrap()
            .fill(&source)
            .unwrap();
        assert!(charge(&pool) > one_handle);
        drop(second);
        safemlx::reclaim_allocation_owners();
        assert_eq!(charge(&pool), one_handle);
        drop(first);
        safemlx::reclaim_allocation_owners();
        assert_eq!(charge(&pool), before);
        assert_eq!(
            source_alias.evaluated().unwrap().as_slice::<f32>(),
            &[3., -7., 11.]
        );
    }
}
