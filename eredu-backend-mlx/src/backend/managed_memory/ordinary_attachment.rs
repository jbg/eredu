//! Explicit host preparation for ordinary native allocation attachments.
use super::*;
use eredu_core::HostPreparationAuthority;
use safemlx::{PreparedAllocationOwner, PreparedAllocationOwnerCause};
use std::{
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
#[error("native allocation attachment failed: {cause}")]
struct AttachmentFailure {
    #[source]
    cause: PreparedAllocationOwnerCause,
    _host: HostPreparationAuthority,
}

struct PaidOwner<T> {
    // Payload metadata retires before its separately paid constructor authority.
    _owner: T,
    _host: HostPreparationAuthority,
}

/// Preparation pays only the attachment and caller-declared host metadata.
/// Existing exclusion/reservation owners retain their original meanings.
pub(crate) struct OrdinaryArrayAttachment<T: Send + 'static> {
    host: HostPreparationAuthority,
    marker: PhantomData<fn() -> T>,
}
impl<T: Send + 'static> OrdinaryArrayAttachment<T> {
    pub(crate) fn prepare(
        pool: &MemoryLedger,
        additional_host_bytes: usize,
    ) -> Result<Self, Error> {
        let layout = PreparedAllocationOwner::<PaidOwner<T>>::layout();
        let parts = [
            layout.allocation_bytes().ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?,
            layout.preparation_control_bytes(),
            layout.prepared_bytes(),
            layout.preparation_failure_bytes(),
            layout.attachment_failure_bytes(),
            layout.original_attachment_control_bytes(),
            size_of::<Self>(),
            size_of::<PaidOwner<T>>(),
            size_of::<PreparedAllocationOwnerCause>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<AttachmentFailure>().ok_or(
                Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow),
            )?,
            size_of::<Result<Self, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<PreparedAllocationOwner<PaidOwner<T>>, Error>>(),
            size_of::<(&MemoryLedger, &safemlx::Array, T, usize)>(),
        ];
        let initial = size_of_val(&parts)
            .checked_add(additional_host_bytes)
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        let bytes = parts
            .into_iter()
            .try_fold(initial, usize::checked_add)
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        let funding = pool
            .prepare_storage_metadata()
            .map_err(|cause| Error::PrefillControl(metadata_memory_error(cause)))?;
        let host = funding
            .prepare_host_owner(bytes)
            .map_err(|cause| Error::PrefillControl(metadata_memory_error(cause)))?;
        Ok(Self {
            host,
            marker: PhantomData,
        })
    }
    pub(crate) fn attach(self, array: &safemlx::Array, owner: T) -> Result<(), Error> {
        let error_host = self.host.clone();
        let prepared = PreparedAllocationOwner::try_new(PaidOwner {
            _owner: owner,
            _host: self.host,
        })
        .map_err(|failure| {
            let (cause, _owner) = failure.into_parts();
            Error::AllocationAttachment(eredu_core::BackendFailure::from_error(AttachmentFailure {
                cause,
                _host: error_host.clone(),
            }))
        })?;
        prepared.try_attach(array).map_err(|failure| {
            let (cause, _prepared) = failure.into_parts();
            Error::AllocationAttachment(eredu_core::BackendFailure::from_error(AttachmentFailure {
                cause,
                _host: error_host,
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn charge(pool: &MemoryLedger) -> u64 {
        pool.snapshot()
            .unwrap()
            .domains
            .into_iter()
            .find(|row| row.domain == pool.topology().host_domain())
            .unwrap()
            .current_charge_bytes
    }
    fn reclaim() {
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
    }

    #[test]
    fn prepaid_attachment_survives_alias_and_cache_until_actual_backing_retirement() {
        let source = safemlx::Array::from_slice(&[2.0f32, -3.0, 5.0, 7.0], &[4]);
        source.evaluated().unwrap();
        let alias = source.clone();
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let baseline = charge(&pool);
        let retired = Arc::new(AtomicUsize::new(0));
        let prepared = OrdinaryArrayAttachment::prepare(&pool, 0).unwrap();
        let funded = charge(&pool);
        assert!(funded > baseline);
        prepared.attach(&source, Retired(retired.clone())).unwrap();
        drop(source);
        reclaim();
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert_eq!(charge(&pool), funded);
        drop(alias);
        reclaim();
        assert_eq!(retired.load(Ordering::SeqCst), 1);
        assert_eq!(charge(&pool), baseline);
    }

    #[test]
    fn rejected_attachment_retains_paid_error_storage_until_error_retirement() {
        let empty = safemlx::Array::from_slice(&[] as &[f32], &[0]);
        empty.evaluated().unwrap();
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let baseline = charge(&pool);
        let retired = Arc::new(AtomicUsize::new(0));
        let prepared = OrdinaryArrayAttachment::prepare(&pool, 0).unwrap();
        let funded = charge(&pool);
        let error = prepared
            .attach(&empty, Retired(retired.clone()))
            .unwrap_err();
        assert!(matches!(&error, Error::AllocationAttachment(_)));
        let mut cause: &dyn std::error::Error = &error;
        while !cause.is::<PreparedAllocationOwnerCause>() {
            cause = cause
                .source()
                .expect("the typed native refusal remains in the source chain");
        }
        assert_eq!(retired.load(Ordering::SeqCst), 1);
        assert_eq!(charge(&pool), funded);
        drop(error);
        assert_eq!(charge(&pool), baseline);
    }
}
