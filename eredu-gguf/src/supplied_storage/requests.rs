//! Checked requested storage, not allocation authority or observed capacity.
use std::alloc::Layout;

/// Cumulative supplied-buffer layouts and finite calls, including zero-length
/// requests. Lazy physical slots may not all be reached on an early refusal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StorageRequestBound {
    bytes: usize,
    calls: usize,
}
impl StorageRequestBound {
    /// One actual typed request, including a zero-size allocation attempt.
    pub fn one<T>(count: usize) -> Option<Self> {
        Some(Self {
            bytes: Layout::array::<T>(count).ok()?.size(),
            calls: 1,
        })
    }
    /// One already checked typed layout; descriptive only, never authority.
    pub fn from_layout(layout: Layout) -> Self {
        Self {
            bytes: layout.size(),
            calls: 1,
        }
    }
    /// Checked union of disjoint request populations; no maximum/peak inference.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            bytes: self.bytes.checked_add(other.bytes)?,
            calls: self.calls.checked_add(other.calls)?,
        })
    }
    /// Sum of requested typed layouts. A qualified producer must bound capacity.
    pub fn bytes(self) -> usize {
        self.bytes
    }
    /// Maximum reached calls, including calls whose requested layout is empty.
    pub fn calls(self) -> usize {
        self.calls
    }
    pub(crate) fn add<T>(&mut self, count: usize) -> Option<()> {
        *self = self.checked_add(Self::one::<T>(count)?)?;
        Some(())
    }
    pub(crate) fn descriptor(&mut self, name: usize, rank: usize) -> Option<()> {
        self.add::<u8>(name)?;
        self.add::<u64>(rank)
    }
}
