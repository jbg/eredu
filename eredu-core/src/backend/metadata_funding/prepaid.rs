//! Finite partition of an already admitted retained host owner.
use super::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use crate::HostPreparationAuthority;
use std::{mem::{size_of, size_of_val}, sync::atomic::{AtomicUsize, Ordering}};

#[derive(Debug)]
struct Account {
    spent: AtomicUsize,
    limit: usize,
    // The counter and funding shells retire before the accepted outer owner.
    _host: HostPreparationAuthority,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let mut prior = self.spent.load(Ordering::Acquire);
        loop {
            let next = prior.checked_add(bytes).ok_or(HostMetadataFundingError::Overflow)?;
            if next > self.limit {
                return Err(HostMetadataFundingError::Capacity {
                    required: u64::try_from(next).map_err(|_| HostMetadataFundingError::Overflow)?,
                    available: u64::try_from(self.limit).map_err(|_| HostMetadataFundingError::Overflow)?,
                });
            }
            match self.spent.compare_exchange_weak(prior, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(()),
                Err(actual) => prior = actual,
            }
        }
    }
}
fn forwarding_bytes() -> Option<usize> {
    let parts = [size_of::<Account>(), size_of::<HostPreparationAuthority>(),
        size_of::<HostMetadataFunding>(), size_of::<HostMetadataFundingError>(),
        size_of::<Result<HostMetadataFunding, HostMetadataFundingError>>(),
        size_of::<(usize, usize, usize)>(), size_of::<Result<usize, usize>>(),
        size_of::<(usize, HostPreparationAuthority)>(),
        HostMetadataFunding::reservation_control_bytes()];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
impl HostMetadataFunding {
    /// Complete constructor/counter transports for a finite prepaid partition,
    /// excluding its payload. The caller must include this in actual admission.
    pub fn prepaid_control_bytes() -> Option<usize> {
        Self::constructor_bytes::<Account>()?.checked_add(forwarding_bytes()?)
    }
    /// Partitions the caller's already admitted `limit`, including constructor
    /// controls. This performs no reservation on the outer owner and grants no
    /// execution authority. The caller must supply its real prepaid amount and
    /// retain this funding with every produced value and partial failure.
    pub fn from_prepaid(limit: usize, host: HostPreparationAuthority)
        -> Result<Self, HostMetadataFundingError> {
        if host.is_unmanaged() { return Err(HostMetadataFundingError::Unavailable); }
        let funding = Self::new(Account { spent: AtomicUsize::new(0), limit, _host: host })?;
        funding.reserve_metadata(forwarding_bytes().ok_or(HostMetadataFundingError::Overflow)?)?;
        Ok(funding)
    }
}
