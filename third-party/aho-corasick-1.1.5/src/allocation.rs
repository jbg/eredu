//! Prospective storage for the original literal automaton producers.
//!
//! Policies borrow the caller's authority, which must outlive every resulting
//! automaton, packed searcher and error. Ordinary calls use [`Unenforced`] on
//! exactly the same workers; policy does not choose a search algorithm.
#![forbid(unsafe_code)]
use alloc::{collections::VecDeque, sync::Arc, vec::Vec};
use core::{alloc::Layout, fmt, mem::size_of, sync::atomic::AtomicUsize};
/// A fixed refusal from one prospective storage producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    /// The caller refused the destination before allocation.
    Refused,
    /// The required capacity or layout cannot be represented.
    SizeOverflow,
    /// The host allocator refused the requested vector/string destination.
    HostAllocation,
}
impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "literal automaton allocation refused",
            Self::SizeOverflow => "literal automaton allocation size overflow",
            Self::HostAllocation => "literal automaton host allocation failed",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for AllocationError {}

/// Admission of the complete capacity of one prospective destination.
pub trait Allocation {
    /// Called before the corresponding allocation or replacement.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
}
/// Ordinary allocation policy for the shared producers.
#[derive(Clone, Copy, Debug)]
pub struct Unenforced;
impl Allocation for Unenforced {
    #[inline]
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        Ok(())
    }
}
/// Borrowed allocation policy and concrete safe storage producers.
#[derive(Clone, Copy)]
pub(crate) struct Allocator<'a> {
    policy: &'a dyn Allocation,
}
impl fmt::Debug for Allocator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Allocator")
    }
}
impl<'a> Allocator<'a> {
    /// Borrow a policy for the duration of construction or search.
    pub fn new(policy: &'a dyn Allocation) -> Self {
        Self { policy }
    }
    /// Borrow this exact policy for a nested producer.
    pub fn policy(self) -> &'a dyn Allocation {
        self.policy
    }
    /// Admit a complete nonempty destination before invoking its producer.
    pub fn reserve(self, bytes: usize) -> Result<(), AllocationError> {
        if bytes == 0 {
            return Ok(());
        }
        self.policy.reserve(bytes)
    }
    fn arc_layout(self, value: Layout) -> Result<(), AllocationError> {
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(value)
            .map_err(|_| AllocationError::SizeOverflow)?;
        self.reserve(layout.pad_to_align().size())
    }
    /// Admit a sized shared owner, including its reference counts and padding.
    pub fn arc<T>(self, value: T) -> Result<Arc<T>, AllocationError> {
        self.arc_layout(Layout::new::<T>())?;
        Ok(Arc::new(value))
    }
    /// Ensure room for additional elements, charging complete geometric replacements.
    pub fn grow<T>(
        self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), AllocationError> {
        let needed = values
            .len()
            .checked_add(additional)
            .ok_or(AllocationError::SizeOverflow)?;
        if needed <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .ok_or(AllocationError::SizeOverflow)?
            .max(needed);
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(AllocationError::SizeOverflow)?;
        self.reserve(bytes)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| AllocationError::HostAllocation)
    }
    /// Admit geometric queue replacement before insertion.
    pub fn queue_push<T>(
        self,
        values: &mut VecDeque<T>,
        value: T,
    ) -> Result<(), AllocationError> {
        if values.len() == values.capacity() {
            let capacity = values
                .capacity()
                .checked_mul(2)
                .ok_or(AllocationError::SizeOverflow)?
                .max(1);
            let layout = Layout::array::<T>(capacity)
                .map_err(|_| AllocationError::SizeOverflow)?;
            self.reserve(layout.size())?;
            values
                .try_reserve_exact(capacity - values.len())
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        values.push_back(value);
        Ok(())
    }
    /// Push after admitting any replacement.
    pub fn push<T>(
        self,
        values: &mut Vec<T>,
        value: T,
    ) -> Result<(), AllocationError> {
        self.grow(values, 1)?;
        values.push(value);
        Ok(())
    }
    /// Resize after admitting any replacement; the value's clone must not allocate.
    pub fn resize_copy<T: Copy>(
        self,
        values: &mut Vec<T>,
        len: usize,
        value: T,
    ) -> Result<(), AllocationError> {
        self.grow(values, len.saturating_sub(values.len()))?;
        values.resize(len, value);
        Ok(())
    }
    /// Copy a slice into its own admitted vector.
    pub fn copy_slice<T: Copy>(
        self,
        values: &[T],
    ) -> Result<Vec<T>, AllocationError> {
        let mut output = Vec::new();
        self.grow(&mut output, values.len())?;
        output.extend_from_slice(values);
        Ok(output)
    }
    /// Admit a possible replacement before shrinking retained vector storage.
    pub fn shrink<T>(
        self,
        values: &mut Vec<T>,
    ) -> Result<(), AllocationError> {
        if values.len() != values.capacity() {
            self.reserve(
                values
                    .len()
                    .checked_mul(size_of::<T>())
                    .ok_or(AllocationError::SizeOverflow)?,
            )?;
            values.shrink_to_fit();
        }
        Ok(())
    }
}

use core::cmp::Ordering;
struct Control {
    root: usize,
    end: usize,
    child: usize,
    selected: usize,
}

pub(crate) fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(
    values: &mut [T],
    mut compare: F,
    allocation: Allocator<'_>,
) -> Result<(), AllocationError> {
    allocation.reserve(size_of::<Control>() + size_of::<F>())?;
    let mut control =
        Control { root: 0, end: values.len(), child: 0, selected: 0 };
    // Build a maximum heap bottom-up, then extract its largest element.
    for root in (0..values.len() / 2).rev() {
        control.root = root;
        sift(values, &mut compare, &mut control);
    }
    while control.end > 1 {
        control.end -= 1;
        values.swap(0, control.end);
        control.root = 0;
        sift(values, &mut compare, &mut control);
    }
    Ok(())
}
fn sift<T, F: FnMut(&T, &T) -> Ordering>(
    values: &mut [T],
    compare: &mut F,
    control: &mut Control,
) {
    while control.root < control.end / 2 {
        control.child = control.root * 2 + 1;
        control.selected = control.child;
        if control.child + 1 < control.end
            && compare(&values[control.child], &values[control.child + 1])
                == Ordering::Less
        {
            control.selected += 1;
        }
        if compare(&values[control.root], &values[control.selected])
            != Ordering::Less
        {
            break;
        }
        values.swap(control.root, control.selected);
        control.root = control.selected;
    }
}

#[cfg(all(feature = "std", feature = "perf-literal"))]
impl memchr::allocation::Allocation for Allocator<'_> {
    fn reserve(
        &self,
        bytes: usize,
    ) -> Result<(), memchr::allocation::AllocationError> {
        Allocator::reserve(*self, bytes).map_err(|error| match error {
            AllocationError::Refused => {
                memchr::allocation::AllocationError::Refused
            }
            AllocationError::SizeOverflow => {
                memchr::allocation::AllocationError::SizeOverflow
            }
            AllocationError::HostAllocation => {
                memchr::allocation::AllocationError::HostAllocation
            }
        })
    }
}
#[cfg(all(feature = "std", feature = "perf-literal"))]
impl From<memchr::allocation::AllocationError> for AllocationError {
    fn from(error: memchr::allocation::AllocationError) -> Self {
        match error {
            memchr::allocation::AllocationError::Refused => Self::Refused,
            memchr::allocation::AllocationError::SizeOverflow => {
                Self::SizeOverflow
            }
            memchr::allocation::AllocationError::HostAllocation => {
                Self::HostAllocation
            }
        }
    }
}

#[cfg(test)]
mod tests;
