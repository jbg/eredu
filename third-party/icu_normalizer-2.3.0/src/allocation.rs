//! Prospective storage for the existing normalization workers.
use alloc::{string::String, vec::Vec};
use core::{alloc::Layout, cell::Cell, fmt};
use smallvec::{Array, SmallVec};

/// A borrowed account for reached normalization storage.
pub trait Allocation {
    /// Admit the actual new backing before its allocation.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
    /// Whether a storage failure is returned to the explicit funded caller.
    fn is_enforced(&self) -> bool {
        true
    }
}
/// Fixed storage failure; the enclosing caller retains its original cause.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    /// The account refused the reached allocation.
    Refused,
    /// The destination layout cannot be represented.
    SizeOverflow,
    /// The host allocator refused the actual destination.
    HostAllocation,
}
impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "normalization allocation refused",
            Self::SizeOverflow => "normalization allocation overflow",
            Self::HostAllocation => "normalization host allocation failed",
        })
    }
}
impl core::error::Error for AllocationError {}
/// Explicit ordinary policy for the same workers.
#[derive(Debug)]
pub struct Unenforced;
impl Allocation for Unenforced {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        Ok(())
    }
    fn is_enforced(&self) -> bool {
        false
    }
}
/// A scoped first-failure recorder. It does not own resulting allocations.
pub struct Allocator<'a> {
    source: &'a dyn Allocation,
    enforced: bool,
    failure: Cell<Option<AllocationError>>,
}
impl fmt::Debug for Allocator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Allocator")
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}
impl<'a> Allocator<'a> {
    /// Borrow the account for one operation's complete allocation extent.
    #[inline]
    pub fn new(source: &'a dyn Allocation) -> Self {
        Self {
            source,
            enforced: source.is_enforced(),
            failure: Cell::new(None),
        }
    }
    /// The first reached failure, without formatting or allocating a diagnostic.
    #[inline]
    pub fn failure(&self) -> Option<AllocationError> {
        self.failure.get()
    }
    /// Retain a host or layout failure before any later producer is reached.
    #[inline]
    pub fn fail(&self, error: AllocationError) -> AllocationError {
        if let Some(prior) = self.failure.get() {
            return prior;
        }
        self.failure.set(Some(error));
        if !self.enforced {
            panic!("ordinary normalization allocation failed: {error}");
        }
        error
    }
    /// Admit one actual destination, preserving the first refusal.
    #[inline]
    pub fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        if !self.enforced {
            return Ok(());
        }
        self.source.reserve(bytes).map_err(|error| self.fail(error))
    }
    /// Grow the concrete small-vector backing once before insertion.
    #[inline]
    pub fn grow<A: Array>(
        &self,
        values: &mut SmallVec<A>,
        required: usize,
    ) -> Result<(), AllocationError> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or_else(|| self.fail(AllocationError::SizeOverflow))?;
        let bytes = Layout::array::<A::Item>(capacity)
            .map_err(|_| self.fail(AllocationError::SizeOverflow))?
            .size();
        self.reserve(bytes)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| self.fail(AllocationError::HostAllocation))?;
        if values.capacity() != capacity {
            return Err(self.fail(AllocationError::HostAllocation));
        }
        Ok(())
    }
    /// Publish an item only after its actual backing is admitted.
    #[inline]
    pub fn push<A: Array>(&self, values: &mut SmallVec<A>, value: A::Item) {
        let Some(required) = values.len().checked_add(1) else {
            self.fail(AllocationError::SizeOverflow);
            return;
        };
        if self.grow(values, required).is_ok() {
            values.push(value);
        }
    }
    /// Extend through the same reached element worker, stopping at first failure.
    #[inline]
    pub fn extend<A: Array>(
        &self,
        values: &mut SmallVec<A>,
        input: impl IntoIterator<Item = A::Item>,
    ) {
        for value in input {
            if self.failure().is_some() {
                return;
            }
            self.push(values, value);
        }
    }
    /// Admit a vector with exactly the requested element capacity.
    #[inline]
    pub fn vector<T>(&self, capacity: usize) -> Result<Vec<T>, AllocationError> {
        if let Some(error) = self.failure() {
            return Err(error);
        }
        let mut values = Vec::new();
        let layout =
            Layout::array::<T>(capacity).map_err(|_| self.fail(AllocationError::SizeOverflow))?;
        if layout.size() != 0 {
            self.reserve(layout.size())?;
            values
                .try_reserve_exact(capacity)
                .map_err(|_| self.fail(AllocationError::HostAllocation))?;
            if values.capacity() != capacity {
                return Err(self.fail(AllocationError::HostAllocation));
            }
        }
        Ok(values)
    }
    /// Grow a UTF-8 destination before writing the reached text.
    #[inline]
    pub fn append(&self, output: &mut String, text: &str) -> Result<(), AllocationError> {
        let required = output
            .len()
            .checked_add(text.len())
            .ok_or_else(|| self.fail(AllocationError::SizeOverflow))?;
        if required > output.capacity() {
            let capacity = output
                .capacity()
                .checked_mul(2)
                .map(|n| n.max(required))
                .ok_or_else(|| self.fail(AllocationError::SizeOverflow))?;
            self.reserve(capacity)?;
            output
                .try_reserve_exact(capacity - output.len())
                .map_err(|_| self.fail(AllocationError::HostAllocation))?;
            if output.capacity() != capacity {
                return Err(self.fail(AllocationError::HostAllocation));
            }
        }
        if let Some(error) = self.failure() {
            return Err(error);
        }
        output.push_str(text);
        Ok(())
    }
}
impl Allocation for Allocator<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        self.reserve(bytes)
    }
    fn is_enforced(&self) -> bool {
        self.enforced
    }
}
