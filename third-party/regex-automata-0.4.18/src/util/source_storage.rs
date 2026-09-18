//! Borrowed census of actual retained source allocation groups.
//!
//! A group consists of one shared owner and its exclusively owned backing.
//! Separately shared children are visited separately. The caller owns identity
//! deduplication and any failure; this traversal allocates no census storage.
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{alloc::Layout, fmt, mem, sync::atomic::AtomicUsize};

/// Receives an actual owner identity and its complete retained byte count.
/// Return false to skip a repeated owner or to stop descending after refusal.
pub trait Visitor {
    /// Return true to descend into this owner's separately shared children.
    fn visit(&mut self, identity: *const (), bytes: usize) -> bool;
}
impl<F: FnMut(*const (), usize) -> bool> Visitor for F {
    fn visit(&mut self, identity: *const (), bytes: usize) -> bool {
        self(identity, bytes)
    }
}

/// A source census cannot include mutable invocation storage by implication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// An ordinary persistent pool has been used. Use scoped search storage
    /// when the source must continue to have an immutable retained census.
    WarmedPool,
    /// Actual allocation geometry cannot be represented by the host size type.
    SizeOverflow,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WarmedPool => "regex source has a used persistent search pool",
            Self::SizeOverflow => "regex source storage size overflow",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Exact shared-owner shell, including reference counts and trailing alignment.
pub fn arc_bytes<T: ?Sized>(value: &T) -> Result<usize, Error> {
    Ok(Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::for_value(value))
        .map_err(|_| Error::SizeOverflow)?
        .0
        .pad_to_align()
        .size())
}
/// Visit a shared shell plus its exclusively owned backing. The return value
/// tells the owner whether to descend into separately shared children.
pub fn arc<T: ?Sized>(
    value: &Arc<T>,
    exclusive_bytes: usize,
    visitor: &mut dyn Visitor,
) -> Result<bool, Error> {
    let bytes = arc_bytes(&**value)?
        .checked_add(exclusive_bytes)
        .ok_or(Error::SizeOverflow)?;
    // The data pointer of an empty Arc slice is one-past its allocation.
    // It can equal the start of an adjacent allocation, so use its owning base.
    let (_, offset) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::for_value(&**value))
        .map_err(|_| Error::SizeOverflow)?;
    let identity = Arc::as_ptr(value)
        .cast::<u8>()
        .wrapping_sub(offset)
        .cast::<()>();
    Ok(visitor.visit(identity, bytes))
}
/// Visit an owning box; its inline bytes do not include child allocations.
pub fn boxed<T: ?Sized>(value: &Box<T>, visitor: &mut dyn Visitor) -> bool {
    let bytes = mem::size_of_val(&**value);
    bytes == 0 || visitor.visit((&**value as *const T).cast::<()>(), bytes)
}
/// Visit the actual backing capacity, including spare rows.
pub fn vector<T>(value: &Vec<T>, visitor: &mut dyn Visitor) -> bool {
    let bytes = value.capacity() * mem::size_of::<T>();
    bytes == 0 || visitor.visit(value.as_ptr().cast::<()>(), bytes)
}
/// Visit an owning string's actual byte capacity.
pub fn string(value: &alloc::string::String, visitor: &mut dyn Visitor) -> bool {
    value.capacity() == 0 || visitor.visit(value.as_ptr().cast::<()>(), value.capacity())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_shared_slice_uses_owning_base_not_one_past_data() {
        let source: Arc<str> = Arc::from("");
        let mut identity = core::ptr::null();
        arc(&source, 0, &mut |id, bytes| {
            identity = id;
            assert_eq!(bytes, mem::size_of::<[AtomicUsize; 2]>());
            true
        })
        .unwrap();
        let data = Arc::as_ptr(&source).cast::<u8>();
        assert_eq!(
            identity
                .cast::<u8>()
                .wrapping_add(mem::size_of::<[AtomicUsize; 2]>()),
            data
        );
        assert_ne!(identity, data.cast::<()>());
        arc(&source.clone(), 0, &mut |id, _| {
            assert_eq!(id, identity);
            false
        })
        .unwrap();
    }
}
