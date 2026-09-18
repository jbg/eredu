//! Fixed allocation transport into the original external literal workers.
use crate::util::allocation::{Allocation, AllocationError};
pub(super) struct Funding<'a>(pub(super) &'a dyn Allocation);
#[cfg(all(feature = "std", feature = "perf-literal-substring"))]
impl memchr::allocation::Allocation for Funding<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), memchr::allocation::AllocationError> {
        self.0.reserve(bytes).map_err(|error| match error {
            AllocationError::Refused => memchr::allocation::AllocationError::Refused,
            AllocationError::SizeOverflow => memchr::allocation::AllocationError::SizeOverflow,
            AllocationError::HostAllocation => memchr::allocation::AllocationError::HostAllocation,
        })
    }
}
#[cfg(all(feature = "std", feature = "perf-literal-substring"))]
pub(super) fn memchr(error: memchr::allocation::AllocationError) -> AllocationError {
    match error {
        memchr::allocation::AllocationError::Refused => AllocationError::Refused,
        memchr::allocation::AllocationError::SizeOverflow => AllocationError::SizeOverflow,
        memchr::allocation::AllocationError::HostAllocation => AllocationError::HostAllocation,
    }
}
#[cfg(feature = "perf-literal-multisubstring")]
impl aho_corasick::allocation::Allocation for Funding<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), aho_corasick::allocation::AllocationError> {
        self.0.reserve(bytes).map_err(|error| match error {
            AllocationError::Refused => aho_corasick::allocation::AllocationError::Refused,
            AllocationError::SizeOverflow => {
                aho_corasick::allocation::AllocationError::SizeOverflow
            }
            AllocationError::HostAllocation => {
                aho_corasick::allocation::AllocationError::HostAllocation
            }
        })
    }
}
#[cfg(feature = "perf-literal-multisubstring")]
pub(super) fn aho(error: aho_corasick::allocation::AllocationError) -> AllocationError {
    match error {
        aho_corasick::allocation::AllocationError::Refused => AllocationError::Refused,
        aho_corasick::allocation::AllocationError::SizeOverflow => AllocationError::SizeOverflow,
        aho_corasick::allocation::AllocationError::HostAllocation => {
            AllocationError::HostAllocation
        }
    }
}
