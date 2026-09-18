//! Prospective allocation for the original automata construction and cache workers.
//!
//! The policy is borrowed. Its owner must retain custody until resulting automata,
//! caches and errors retire. Ordinary entry points use [`Unenforced`] through the
//! same producers. Syntax remains optional; this module does not select an engine.
use alloc::{boxed::Box, rc::Rc, string::String, sync::Arc, vec::Vec};
use core::{
    alloc::Layout,
    cell::Cell,
    fmt,
    hash::{BuildHasher, Hash},
    mem::size_of,
    sync::atomic::AtomicUsize,
};

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
            Self::Refused => "automata allocation refused",
            Self::SizeOverflow => "automata allocation size overflow",
            Self::HostAllocation => "automata host allocation failed",
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
pub struct Allocator<'a> {
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
    /// Use the ordinary policy on the same worker.
    pub const fn unenforced() -> Self {
        Self {
            policy: &Unenforced,
        }
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
    /// Admit one sized box.
    pub fn boxed<T>(self, value: T) -> Result<Box<T>, AllocationError> {
        self.reserve(size_of::<T>())?;
        Ok(Box::new(value))
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
    /// Admit a local shared owner, including its reference counts and padding.
    pub fn rc<T>(self, value: T) -> Result<Rc<T>, AllocationError> {
        let (layout, _) = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<T>())
            .map_err(|_| AllocationError::SizeOverflow)?;
        self.reserve(layout.pad_to_align().size())?;
        Ok(Rc::new(value))
    }
    /// Copy text into an admitted shared string without an intermediate String.
    pub fn arc_str(self, text: &str) -> Result<Arc<str>, AllocationError> {
        self.arc_layout(
            Layout::array::<u8>(text.len()).map_err(|_| AllocationError::SizeOverflow)?,
        )?;
        Ok(Arc::from(text))
    }
    /// Copy plain elements into an admitted shared slice.
    pub fn copy_arc_slice<T: Copy>(self, values: &[T]) -> Result<Arc<[T]>, AllocationError> {
        self.arc_layout(
            Layout::array::<T>(values.len()).map_err(|_| AllocationError::SizeOverflow)?,
        )?;
        Ok(Arc::from(values))
    }
    /// Move a vector into a shared slice, admitting any shrinking destination too.
    pub fn arc_slice<T>(self, values: Vec<T>) -> Result<Arc<[T]>, AllocationError> {
        let values = self.boxed_slice(values)?;
        self.arc_layout(
            Layout::array::<T>(values.len()).map_err(|_| AllocationError::SizeOverflow)?,
        )?;
        Ok(Arc::from(values))
    }
    /// Ensure room for additional elements, charging complete geometric replacements.
    pub fn grow<T>(self, values: &mut Vec<T>, additional: usize) -> Result<(), AllocationError> {
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
    /// Push after admitting any replacement.
    pub fn push<T>(self, values: &mut Vec<T>, value: T) -> Result<(), AllocationError> {
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
    pub fn copy_slice<T: Copy>(self, values: &[T]) -> Result<Vec<T>, AllocationError> {
        let mut output = Vec::new();
        self.grow(&mut output, values.len())?;
        output.extend_from_slice(values);
        Ok(output)
    }
    /// Append plain elements after admitting any replacement.
    pub fn extend_copy<T: Copy>(
        self,
        values: &mut Vec<T>,
        other: &[T],
    ) -> Result<(), AllocationError> {
        self.grow(values, other.len())?;
        values.extend_from_slice(other);
        Ok(())
    }
    /// Convert to a boxed slice after admitting a possible shrinking move.
    pub fn boxed_slice<T>(self, values: Vec<T>) -> Result<Box<[T]>, AllocationError> {
        if values.capacity() != values.len() {
            self.reserve(
                values
                    .len()
                    .checked_mul(size_of::<T>())
                    .ok_or(AllocationError::SizeOverflow)?,
            )?;
        }
        Ok(values.into_boxed_slice())
    }
    /// Admit a possible replacement before shrinking retained vector storage.
    pub fn shrink<T>(self, values: &mut Vec<T>) -> Result<(), AllocationError> {
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
    /// Copy text into admitted string storage.
    pub fn copy_str(self, text: &str) -> Result<String, AllocationError> {
        self.reserve(text.len())?;
        let mut output = String::new();
        output
            .try_reserve_exact(text.len())
            .map_err(|_| AllocationError::HostAllocation)?;
        output.push_str(text);
        Ok(output)
    }
    /// Append text after admitting any complete geometric replacement.
    pub fn push_str(self, output: &mut String, text: &str) -> Result<(), AllocationError> {
        let needed = output
            .len()
            .checked_add(text.len())
            .ok_or(AllocationError::SizeOverflow)?;
        if needed > output.capacity() {
            let capacity = output
                .capacity()
                .checked_mul(2)
                .ok_or(AllocationError::SizeOverflow)?
                .max(needed);
            self.reserve(capacity)?;
            output
                .try_reserve_exact(capacity - output.len())
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        output.push_str(text);
        Ok(())
    }
    /// Append one scalar without a temporary allocation.
    pub fn push_char(self, output: &mut String, value: char) -> Result<(), AllocationError> {
        self.push_str(output, value.encode_utf8(&mut [0; 4]))
    }
    /// Admit the actual hash-table layout before a new-key insertion.
    pub fn prepare_insert<K: Eq + Hash, V, S: BuildHasher>(
        self,
        map: &mut hashbrown::HashMap<K, V, S>,
        key: &K,
    ) -> Result<(), AllocationError> {
        if !map.contains_key(key) {
            if let Some(layout) = map
                .try_reserve_layout(1)
                .map_err(|_| AllocationError::SizeOverflow)?
            {
                self.reserve(layout.size())?;
            }
            map.try_reserve(1)
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        Ok(())
    }
    /// Insert after admitting the actual table layout for a new key.
    pub fn insert<K: Eq + Hash, V, S: BuildHasher>(
        self,
        map: &mut hashbrown::HashMap<K, V, S>,
        key: K,
        value: V,
    ) -> Result<Option<V>, AllocationError> {
        self.prepare_insert(map, &key)?;
        Ok(map.insert(key, value))
    }
    /// Stream a diagnostic into admitted string storage.
    pub fn format(self, arguments: fmt::Arguments<'_>) -> Result<String, AllocationError> {
        struct Writer<'a> {
            allocation: Allocator<'a>,
            output: String,
            failure: Option<AllocationError>,
        }
        impl fmt::Write for Writer<'_> {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                self.allocation
                    .push_str(&mut self.output, text)
                    .map_err(|error| {
                        self.failure = Some(error);
                        fmt::Error
                    })
            }
        }
        let mut writer = Writer {
            allocation: self,
            output: String::new(),
            failure: None,
        };
        fmt::write(&mut writer, arguments)
            .map_err(|_| writer.failure.unwrap_or(AllocationError::SizeOverflow))?;
        Ok(writer.output)
    }
}

#[cfg(feature = "syntax")]
impl From<regex_syntax::allocation::AllocationError> for AllocationError {
    fn from(error: regex_syntax::allocation::AllocationError) -> Self {
        match error {
            regex_syntax::allocation::AllocationError::Refused => Self::Refused,
            regex_syntax::allocation::AllocationError::SizeOverflow => Self::SizeOverflow,
            regex_syntax::allocation::AllocationError::HostAllocation => Self::HostAllocation,
        }
    }
}
#[cfg(feature = "syntax")]
impl From<AllocationError> for regex_syntax::allocation::AllocationError {
    fn from(error: AllocationError) -> Self {
        match error {
            AllocationError::Refused => Self::Refused,
            AllocationError::SizeOverflow => Self::SizeOverflow,
            AllocationError::HostAllocation => Self::HostAllocation,
        }
    }
}
#[cfg(feature = "syntax")]
impl regex_syntax::allocation::Allocation for Allocator<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), regex_syntax::allocation::AllocationError> {
        (*self).reserve(bytes).map_err(Into::into)
    }
}
