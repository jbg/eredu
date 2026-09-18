//! Borrowed admission for the URI workers' actual storage requests.
//!
//! The caller retains its funding owner through all returned values and errors.
//! Ordinary APIs use [`Unenforced`] with the same construction workers.

use alloc::string::String;
use core::{
    fmt,
    ops::{Deref, DerefMut},
};

/// A fixed failure requiring no diagnostic allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    /// The caller refused the prospective request.
    Refused,
    /// A required allocation layout cannot be represented.
    SizeOverflow,
    /// The host could not allocate the declared storage.
    HostAllocation,
}
impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "URI storage request refused",
            Self::SizeOverflow => "URI storage layout overflow",
            Self::HostAllocation => "URI storage allocation failed",
        })
    }
}
#[cfg(feature = "impl-error")]
impl crate::Error for AllocationError {}

/// Reserves the complete reached allocation request before mutation.
pub trait Allocation: Send + Sync {
    /// Admit `bytes` while all earlier allocations remain live.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
    /// Whether opaque external allocation producers require a declared admission contract.
    fn is_enforced(&self) -> bool {
        true
    }
}
/// Ordinary unrestricted policy; it does not establish a memory bound.
#[derive(Clone, Copy, Debug)]
pub struct Unenforced;
impl Allocation for Unenforced {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        Ok(())
    }
    fn is_enforced(&self) -> bool {
        false
    }
}

pub(crate) struct Buffer<'a> {
    value: String,
    allocation: &'a dyn Allocation,
}
impl<'a> Buffer<'a> {
    pub(crate) fn new(
        capacity: usize,
        allocation: &'a dyn Allocation,
    ) -> Result<Self, AllocationError> {
        let mut result = Self {
            value: String::new(),
            allocation,
        };
        result.reserve_to(capacity)?;
        Ok(result)
    }
    fn reserve_to(&mut self, capacity: usize) -> Result<(), AllocationError> {
        if capacity <= self.value.capacity() {
            return Ok(());
        }
        if capacity > isize::MAX as usize {
            return Err(AllocationError::SizeOverflow);
        }
        self.allocation.reserve(capacity)?;
        self.value
            .try_reserve_exact(capacity - self.value.len())
            .map_err(|_| AllocationError::HostAllocation)
    }
    fn grow(&mut self, additional: usize) -> Result<(), AllocationError> {
        let needed = self
            .value
            .len()
            .checked_add(additional)
            .ok_or(AllocationError::SizeOverflow)?;
        if needed <= self.value.capacity() {
            return Ok(());
        }
        let capacity = self
            .value
            .capacity()
            .checked_mul(2)
            .ok_or(AllocationError::SizeOverflow)?
            .max(needed);
        self.reserve_to(capacity)
    }
    pub(crate) fn push_str(&mut self, value: &str) -> Result<(), AllocationError> {
        self.grow(value.len())?;
        self.value.push_str(value);
        Ok(())
    }
    pub(crate) fn push(&mut self, value: char) -> Result<(), AllocationError> {
        self.grow(value.len_utf8())?;
        self.value.push(value);
        Ok(())
    }
    pub(crate) fn insert_str(&mut self, index: usize, value: &str) -> Result<(), AllocationError> {
        self.grow(value.len())?;
        self.value.insert_str(index, value);
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        self.value.clear();
    }
    pub(crate) fn truncate(&mut self, len: usize) {
        self.value.truncate(len);
    }
    pub(crate) fn finish(self) -> String {
        self.value
    }
    pub(crate) fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> Result<(), AllocationError> {
        struct Writer<'a, 'b> {
            buffer: &'a mut Buffer<'b>,
            error: Option<AllocationError>,
        }
        impl fmt::Write for Writer<'_, '_> {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                self.buffer.push_str(value).map_err(|error| {
                    self.error = Some(error);
                    fmt::Error
                })
            }
        }
        let mut writer = Writer {
            buffer: self,
            error: None,
        };
        fmt::write(&mut writer, args)
            .map_err(|_| writer.error.unwrap_or(AllocationError::HostAllocation))
    }
}
impl Deref for Buffer<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        &self.value
    }
}
impl DerefMut for Buffer<'_> {
    fn deref_mut(&mut self) -> &mut str {
        &mut self.value
    }
}

#[cfg(test)]
mod tests;
