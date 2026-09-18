//! Prospective storage for original owned-needle construction.
//!
//! The caller retains its actual funding authority through finder retirement.
#![forbid(unsafe_code)]
use core::fmt;

/// Fixed failure before an owned-needle destination is constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    /// The caller declined the actual destination.
    Refused,
    /// The requested byte length cannot be represented.
    SizeOverflow,
    /// The host allocator refused the admitted destination.
    HostAllocation,
}
impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "needle allocation refused",
            Self::SizeOverflow => "needle allocation size overflow",
            Self::HostAllocation => "needle host allocation failed",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for AllocationError {}

/// Admission before the original needle allocation or replacement.
pub trait Allocation {
    /// Admit the complete destination byte length before allocation.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
}
/// Explicit ordinary policy for the same owned-needle worker.
pub struct Unenforced;
impl Allocation for Unenforced {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        Ok(())
    }
}

pub(crate) fn copy(
    bytes: &[u8],
    funding: &dyn Allocation,
) -> Result<alloc::boxed::Box<[u8]>, AllocationError> {
    if bytes.len() > isize::MAX as usize {
        return Err(AllocationError::SizeOverflow);
    }
    if !bytes.is_empty() {
        funding.reserve(bytes.len())?;
    }
    let mut destination = alloc::vec::Vec::new();
    destination
        .try_reserve_exact(bytes.len())
        .map_err(|_| AllocationError::HostAllocation)?;
    destination.extend_from_slice(bytes);
    if destination.len() != destination.capacity() && !bytes.is_empty() {
        funding.reserve(bytes.len())?;
    }
    Ok(destination.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    struct Gate {
        calls: Cell<usize>,
        refuse: bool,
    }
    impl Allocation for Gate {
        fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
            assert_eq!(bytes, 6);
            self.calls.set(self.calls.get() + 1);
            if self.refuse {
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn original_finders_refuse_before_copy_and_owned_moves_do_not_allocate() {
        let gate = Gate { calls: Cell::new(0), refuse: true };
        assert!(matches!(
            crate::memmem::Finder::new(b"needle")
                .into_owned_with_allocations(&gate),
            Err(AllocationError::Refused)
        ));
        assert!(matches!(
            crate::memmem::FinderRev::new(b"needle")
                .into_owned_with_allocations(&gate),
            Err(AllocationError::Refused)
        ));
        assert_eq!(2, gate.calls.get());
        let pass = Gate { calls: Cell::new(0), refuse: false };
        let finder = crate::memmem::Finder::new(b"needle")
            .into_owned_with_allocations(&pass)
            .unwrap();
        let reverse = crate::memmem::FinderRev::new(b"needle")
            .into_owned_with_allocations(&pass)
            .unwrap();
        assert_eq!(Some(1), finder.find(b"!needle needle"));
        assert_eq!(Some(8), reverse.rfind(b"!needle needle"));
        assert_eq!(
            Some(0),
            finder.into_owned_with_allocations(&gate).unwrap().find(b"needle")
        );
        assert_eq!(
            Some(0),
            reverse
                .into_owned_with_allocations(&gate)
                .unwrap()
                .rfind(b"needle")
        );
        assert_eq!(2, gate.calls.get());
    }
}
