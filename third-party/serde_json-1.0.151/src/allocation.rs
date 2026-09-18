//! Explicit prospective allocation for JSON source construction.
//!
//! The caller retains the original account and its concrete refusal. These
//! dependency-local markers do not allocate diagnostics or replace that owner.
use alloc::{string::String, vec::Vec};
use core::{alloc::Layout, fmt};

/// Funding for one reached source allocation.
pub trait Allocation {
    /// Reserve the complete new backing while the prior backing is still live.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
    /// Whether reached producers must have a prospective storage contract.
    /// Only the explicit ordinary policy returns false.
    fn is_enforced(&self) -> bool {
        true
    }
}

/// Fixed storage failure, propagated before the next producer is reached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationError {
    /// The original account refused the reached allocation.
    Refused,
    /// The allocation layout cannot be represented.
    SizeOverflow,
    /// The host allocator refused the requested backing.
    HostAllocation,
}
impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "JSON source allocation refused",
            Self::SizeOverflow => "JSON source allocation extent overflow",
            Self::HostAllocation => "JSON source host allocation failed",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for AllocationError {}

/// Explicit ordinary policy for the same source producers.
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

/// A borrowed source policy. It never adopts ownership of the resulting value.
#[derive(Clone, Copy)]
pub struct Allocator<'a>(&'a dyn Allocation);
impl<'a> Allocator<'a> {
    /// Borrow the account retained by the enclosing source/error owner.
    pub fn new(source: &'a dyn Allocation) -> Self {
        Self(source)
    }
    /// Reserve one concrete allocation before constructing its destination.
    pub fn reserve(self, bytes: usize) -> Result<(), AllocationError> {
        self.0.reserve(bytes)
    }
    /// Grow a vector to the reached extent, preserving it on refusal.
    pub fn grow<T>(self, values: &mut Vec<T>, required: usize) -> Result<(), AllocationError> {
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or(AllocationError::SizeOverflow)?;
        let layout = Layout::array::<T>(capacity).map_err(|_| AllocationError::SizeOverflow)?;
        self.reserve(layout.size())?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| AllocationError::HostAllocation)?;
        if values.capacity() != capacity {
            return Err(AllocationError::HostAllocation);
        }
        Ok(())
    }
    /// Publish a value only after its destination backing is reserved.
    pub fn push<T>(self, values: &mut Vec<T>, value: T) -> Result<(), AllocationError> {
        self.grow(
            values,
            values
                .len()
                .checked_add(1)
                .ok_or(AllocationError::SizeOverflow)?,
        )?;
        values.push(value);
        Ok(())
    }
    /// Copy a UTF-8 string into a prospectively reserved destination.
    pub fn copy_string(self, source: &str) -> Result<String, AllocationError> {
        let mut bytes = Vec::new();
        self.grow(&mut bytes, source.len())?;
        bytes.extend_from_slice(source.as_bytes());
        Ok(String::from_utf8(bytes).expect("copied UTF-8 source"))
    }
    /// Format through the ordinary formatter, funding each reached text growth.
    pub fn format(self, value: impl fmt::Display) -> Result<String, AllocationError> {
        struct Output<'a> {
            allocator: Allocator<'a>,
            text: String,
            failure: Option<AllocationError>,
        }
        impl fmt::Write for Output<'_> {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                if self.failure.is_some() {
                    return Err(fmt::Error);
                }
                match self.allocator.append(&mut self.text, text) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        self.failure = Some(error);
                        Err(fmt::Error)
                    }
                }
            }
        }
        let mut output = Output {
            allocator: self,
            text: String::new(),
            failure: None,
        };
        if fmt::write(&mut output, format_args!("{value}")).is_err() {
            return Err(output.failure.unwrap_or(AllocationError::HostAllocation));
        }
        Ok(output.text)
    }
    /// Grow an existing UTF-8 destination before a reached parser write.
    pub fn grow_string(self, output: &mut String, required: usize) -> Result<(), AllocationError> {
        if required <= output.capacity() {
            return Ok(());
        }
        let capacity = output
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or(AllocationError::SizeOverflow)?;
        self.reserve(capacity)?;
        output
            .try_reserve_exact(capacity - output.len())
            .map_err(|_| AllocationError::HostAllocation)?;
        if output.capacity() != capacity {
            return Err(AllocationError::HostAllocation);
        }
        Ok(())
    }
    /// Append through the same prospectively admitted UTF-8 producer.
    pub fn append(self, output: &mut String, text: &str) -> Result<(), AllocationError> {
        let required = output
            .len()
            .checked_add(text.len())
            .ok_or(AllocationError::SizeOverflow)?;
        self.grow_string(output, required)?;
        output.push_str(text);
        Ok(())
    }
}

/// One invocation's first-failure boundary. Later parser traversal cannot call
/// the original account again after a destination or scanner was refused.
pub struct Session<'a> {
    source: &'a dyn Allocation,
    failure: core::cell::Cell<Option<AllocationError>>,
}
impl<'a> Session<'a> {
    /// Borrow the actual source policy; this control owns no allocation.
    pub fn new(source: &'a dyn Allocation) -> Self {
        Self {
            source,
            failure: core::cell::Cell::new(None),
        }
    }
    /// The first fixed allocation failure from this invocation.
    pub fn failure(&self) -> Option<AllocationError> {
        self.failure.get()
    }
}
impl Allocation for Session<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        self.source
            .reserve(bytes)
            .inspect_err(|&error| self.failure.set(Some(error)))
    }
    fn is_enforced(&self) -> bool {
        self.source.is_enforced()
    }
}
