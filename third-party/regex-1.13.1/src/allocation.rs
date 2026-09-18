//! Borrowed storage admission for the original regex construction and search workers.
//!
//! The caller retains its source and invocation funding until their outputs and
//! errors retire. Ordinary entry points use the same workers with `Unenforced`.
use alloc::{string::String, sync::Arc};
use core::{alloc::Layout, sync::atomic::AtomicUsize};
pub use regex_automata::util::allocation::{Allocation, AllocationError, Allocator, Unenforced};

pub(crate) fn strings(
    values: &[String],
    policy: &dyn Allocation,
) -> Result<Arc<[String]>, AllocationError> {
    let allocator = Allocator::new(policy);
    let layout = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::array::<String>(values.len()).map_err(|_| AllocationError::SizeOverflow)?)
        .map_err(|_| AllocationError::SizeOverflow)?
        .0
        .pad_to_align();
    allocator.reserve(layout.size())?;
    for value in values {
        allocator.reserve(value.len())?;
    }
    // The original Arc slice producer clones each String into exact-length
    // backing after the shared shell. Every concrete destination is prepaid.
    Ok(Arc::from(values))
}
