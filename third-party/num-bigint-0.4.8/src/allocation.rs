//! Prospective storage for the original integer workers.
use alloc::vec::Vec;
use core::{alloc::Layout, fmt};

/// A borrowed account covering the complete lifetime of produced storage.
pub trait Allocation {
    /// Admit the actual next destination before it is allocated.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
    /// Whether an unqualified producer must be rejected before entry.
    fn is_enforced(&self) -> bool {
        true
    }
}
/// Fixed failure; the enclosing owner retains the original account cause.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    /// The source account refused this destination.
    Refused,
    /// The destination layout cannot be represented.
    SizeOverflow,
    /// The host refused the actual allocation.
    HostAllocation,
    /// This original producer has not yet provided an allocation contract.
    Unqualified(&'static str),
}
impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused => f.write_str("integer allocation refused"),
            Self::SizeOverflow => f.write_str("integer allocation overflow"),
            Self::HostAllocation => f.write_str("integer host allocation failed"),
            Self::Unqualified(source) => write!(f, "unqualified integer producer: {source}"),
        }
    }
}
impl core::error::Error for AllocationError {}
/// Explicit ordinary policy for the same workers.
pub struct Unenforced;
impl Allocation for Unenforced {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        Ok(())
    }
    fn is_enforced(&self) -> bool {
        false
    }
}
/// Borrowed prospective producer helpers. No global or retained ambient policy.
#[derive(Clone, Copy)]
pub struct Allocator<'a>(pub &'a dyn Allocation);
impl Allocator<'_> {
    /// Admit actual control storage.
    pub fn controls<T>(&self) -> Result<(), AllocationError> {
        self.0.reserve(core::mem::size_of::<T>())
    }
    /// Allocate exactly one concrete vector backing.
    pub fn vector<T>(&self, capacity: usize) -> Result<Vec<T>, AllocationError> {
        let mut values = Vec::new();
        self.grow(&mut values, capacity)?;
        Ok(values)
    }
    /// Grow a concrete vector once, without a smaller retry.
    pub fn grow<T>(&self, values: &mut Vec<T>, required: usize) -> Result<(), AllocationError> {
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or(AllocationError::SizeOverflow)?;
        let layout = Layout::array::<T>(capacity).map_err(|_| AllocationError::SizeOverflow)?;
        self.0.reserve(layout.size())?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| AllocationError::HostAllocation)?;
        if values.capacity() != capacity {
            return Err(AllocationError::HostAllocation);
        }
        Ok(())
    }
    /// Refuse an as-yet unqualified original producer before it is invoked.
    pub fn qualify(&self, source: &'static str) -> Result<(), AllocationError> {
        if self.0.is_enforced() {
            Err(AllocationError::Unqualified(source))
        } else {
            Ok(())
        }
    }
}
