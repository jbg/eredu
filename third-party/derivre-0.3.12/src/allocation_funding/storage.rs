//! Prospective allocation through the parser's existing retained account.
use super::{ParserAllocationFailure, ParserAllocationFunding};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    error::Error,
    fmt::{self, Write},
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
enum Cause {
    Syntax(regex_syntax::allocation::AllocationError),
    Overflow,
    Destination,
    Allocation(TryReserveError),
    Table(hashbrown::TryReserveError),
    OrderedTable(indexmap::TryReserveError),
    Funding(ParserAllocationFailure),
}

/// A fixed allocation refusal. The same original account outlives the error;
/// no formatted error, replacement callback or independently refunded account
/// is constructed on this path.
#[derive(Debug)]
pub struct ParserStorageError {
    cause: Cause,
    funding: ParserAllocationFunding,
}
impl fmt::Display for ParserStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Syntax(error) => error.fmt(f),
            Cause::Overflow => f.write_str("parser allocation extent overflow"),
            Cause::Destination => f.write_str("parser allocation destination changed"),
            Cause::Allocation(error) => error.fmt(f),
            Cause::Table(error) => error.fmt(f),
            Cause::OrderedTable(error) => error.fmt(f),
            Cause::Funding(error) => error.fmt(f),
        }
    }
}
impl Error for ParserStorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.cause {
            Cause::Allocation(error) => Some(error),
            Cause::Table(error) => Some(error),
            Cause::OrderedTable(error) => Some(error),
            Cause::Funding(error) => Some(error),
            _ => None,
        }
    }
}

impl ParserAllocationFunding {
    /// Converts the dependency's fixed allocation marker while preserving the
    /// concrete callback refusal retained in this original funding owner.
    pub(crate) fn syntax_allocation_error(&self, error: regex_syntax::allocation::AllocationError) -> ParserStorageError {
        match self.failure() {
            Some(cause) => self.storage_error(Cause::Funding(cause)),
            None => self.storage_error(Cause::Syntax(error)),
        }
    }
    /// Exact reference-count shell and payload before constructing a shared
    /// local parser source. Nested payload storage is funded by its producer.
    pub fn try_rc<T>(&self, value: T) -> Result<std::rc::Rc<T>, ParserStorageError> {
        let layout = Layout::new::<[std::cell::Cell<usize>; 2]>()
            .extend(Layout::new::<T>()).map_err(|_| self.storage_overflow())?.0.pad_to_align();
        let bytes = layout.size().checked_add(size_of::<T>()).and_then(|n| n.checked_add(size_of::<std::rc::Rc<T>>())).ok_or_else(|| self.storage_overflow())?;
        self.reserve(bytes).map_err(|error| self.storage_error(Cause::Funding(error)))?;
        Ok(std::rc::Rc::new(value))
    }
    /// Fixed checked-arithmetic refusal retaining this exact compiler account.
    pub fn storage_overflow(&self) -> ParserStorageError {
        self.storage_error(Cause::Overflow)
    }
    fn storage_error(&self, cause: Cause) -> ParserStorageError {
        ParserStorageError {
            cause,
            funding: self.clone(),
        }
    }

    /// Grow to the reached extent using one explicit geometric allocation.
    /// The complete new allocation is paid while the old allocation remains
    /// live. Accepted reservations are cumulative even if allocation fails.
    pub fn try_grow_vec<T>(
        &self,
        values: &mut Vec<T>,
        required: usize,
    ) -> Result<(), ParserStorageError> {
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        let layout =
            Layout::array::<T>(capacity).map_err(|_| self.storage_error(Cause::Overflow))?;
        let controls = [
            size_of::<(&Self, &mut Vec<T>, usize)>(),
            size_of::<Layout>(),
            size_of::<(usize, usize)>(),
            size_of::<ParserStorageError>(),
            size_of::<Result<(), ParserStorageError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_add(layout.size()))
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.reserve(bytes)
            .map_err(|e| self.storage_error(Cause::Funding(e)))?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|e| self.storage_error(Cause::Allocation(e)))?;
        if values.capacity() != capacity {
            return Err(self.storage_error(Cause::Destination));
        }
        Ok(())
    }

    /// Publish one initialized element only after its reached backing is paid.
    pub fn try_push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), ParserStorageError> {
        let required = values
            .len()
            .checked_add(1)
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.try_grow_vec(values, required)?;
        values.push(value);
        Ok(())
    }

    /// Copies inline elements without invoking a potentially allocating Clone.
    pub fn try_extend_copy<T: Copy>(
        &self,
        values: &mut Vec<T>,
        source: &[T],
    ) -> Result<(), ParserStorageError> {
        let required = values
            .len()
            .checked_add(source.len())
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.try_grow_vec(values, required)?;
        values.extend_from_slice(source);
        Ok(())
    }

    /// Extend UTF-8 through the same checked vector worker. On refusal the
    /// original prefix is restored unchanged, including its existing backing.
    pub fn try_push_str(&self, value: &mut String, suffix: &str) -> Result<(), ParserStorageError> {
        let mut bytes = std::mem::take(value).into_bytes();
        let result = self.try_extend_copy(&mut bytes, suffix.as_bytes());
        *value = String::from_utf8(bytes).expect("UTF-8 prefix and suffix");
        result
    }

    pub fn try_push_char(&self, value: &mut String, ch: char) -> Result<(), ParserStorageError> {
        let mut bytes = [0; 4];
        self.try_push_str(value, ch.encode_utf8(&mut bytes))
    }

    /// Creates the exact UTF-8 payload through the same paid vector worker.
    pub fn try_copy_str(&self, source: &str) -> Result<String, ParserStorageError> {
        let mut bytes = Vec::new();
        self.try_extend_copy(&mut bytes, source.as_bytes())?;
        // The copied source is already UTF-8; conversion moves the same vector.
        Ok(String::from_utf8(bytes).expect("copied UTF-8 source"))
    }

    /// Reserve through the owning map's actual resize/rehash decision. Keys
    /// and values are already initialized; their nested storage is separate.
    pub fn try_reserve_map<K, V, S>(
        &self,
        map: &mut hashbrown::HashMap<K, V, S>,
        additional: usize,
    ) -> Result<(), ParserStorageError>
    where
        K: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        let layout = map
            .try_reserve_layout(additional)
            .map_err(|e| self.storage_error(Cause::Table(e)))?;
        let controls = [
            size_of::<(&Self, &mut hashbrown::HashMap<K, V, S>, usize)>(),
            size_of::<Option<Layout>>(),
            size_of::<ParserStorageError>(),
            size_of::<Result<(), ParserStorageError>>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_add(layout.map_or(0, |layout| layout.size())))
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.reserve(bytes)
            .map_err(|e| self.storage_error(Cause::Funding(e)))?;
        map.try_reserve(additional)
            .map_err(|e| self.storage_error(Cause::Table(e)))?;
        if layout.is_some_and(|layout| layout.size() != map.allocation_size()) {
            return Err(self.storage_error(Cause::Destination));
        }
        Ok(())
    }

    /// Reserve insertion-ordered table and entry backing through its owning
    /// worker. The table may retain its accepted first allocation on refusal.
    pub fn try_reserve_index_map<K, V, S>(
        &self, map: &mut indexmap::IndexMap<K, V, S>, additional: usize,
    ) -> Result<(), ParserStorageError> {
        self.reserve(size_of::<(&Self, &mut indexmap::IndexMap<K, V, S>, usize)>()
            .checked_add(size_of::<ParserStorageError>()).ok_or_else(|| self.storage_overflow())?)
            .map_err(|error| self.storage_error(Cause::Funding(error)))?;
        map.try_reserve_with(additional, |layout| self.reserve(layout.size()))
            .map_err(|error| match error {
                indexmap::TryReserveWithError::Funding(error) => self.storage_error(Cause::Funding(error)),
                indexmap::TryReserveWithError::Allocation(error) => self.storage_error(Cause::OrderedTable(error)),
            })
    }

    /// Insert a reached ordered entry after paying its actual backing. Nested
    /// payloads are retained values constructed by separately funded producers.
    pub fn try_insert_index_map<K, V, S>(
        &self, map: &mut indexmap::IndexMap<K, V, S>, key: K, value: V,
    ) -> Result<Option<V>, ParserStorageError>
    where K: Eq + std::hash::Hash, S: std::hash::BuildHasher {
        if !map.contains_key(&key) { self.try_reserve_index_map(map, 1)?; }
        Ok(map.insert(key, value))
    }

    /// The same prospective producer for insertion-ordered sets.
    pub fn try_insert_index_set<T, S>(
        &self, set: &mut indexmap::IndexSet<T, S>, value: T,
    ) -> Result<bool, ParserStorageError>
    where T: Eq + std::hash::Hash, S: std::hash::BuildHasher {
        if !set.contains(&value) {
            self.reserve(size_of::<(&Self, &mut indexmap::IndexSet<T, S>, &T)>()
                .checked_add(size_of::<ParserStorageError>()).ok_or_else(|| self.storage_overflow())?)
                .map_err(|error| self.storage_error(Cause::Funding(error)))?;
            set.try_reserve_with(1, |layout| self.reserve(layout.size())).map_err(|error| match error {
                indexmap::TryReserveWithError::Funding(error) => self.storage_error(Cause::Funding(error)),
                indexmap::TryReserveWithError::Allocation(error) => self.storage_error(Cause::OrderedTable(error)),
            })?;
        }
        Ok(set.insert(value))
    }

    /// The equivalent owning-table operation when keys live in a separate
    /// source vector. Rehashing borrows those keys and never duplicates them.
    pub fn try_reserve_table<T>(
        &self,
        table: &mut hashbrown::HashTable<T>,
        additional: usize,
        rehash: impl Fn(&T) -> u64,
    ) -> Result<(), ParserStorageError> {
        let layout = table
            .try_reserve_layout(additional)
            .map_err(|e| self.storage_error(Cause::Table(e)))?;
        let controls = [
            size_of::<(&Self, &mut hashbrown::HashTable<T>, usize)>(),
            size_of_val(&rehash),
            size_of::<Option<Layout>>(),
            size_of::<ParserStorageError>(),
            size_of::<Result<(), ParserStorageError>>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_add(layout.map_or(0, |layout| layout.size())))
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.reserve(bytes)
            .map_err(|e| self.storage_error(Cause::Funding(e)))?;
        table
            .try_reserve(additional, rehash)
            .map_err(|e| self.storage_error(Cause::Table(e)))?;
        if layout.is_some_and(|layout| layout.size() != table.allocation_size()) {
            return Err(self.storage_error(Cause::Destination));
        }
        Ok(())
    }

    /// Insert through the same map after reserving its reached allocation.
    pub fn try_insert<K, V, S>(
        &self,
        map: &mut hashbrown::HashMap<K, V, S>,
        key: K,
        value: V,
    ) -> Result<Option<V>, ParserStorageError>
    where
        K: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        if !map.contains_key(&key) {
            self.try_reserve_map(map, 1)?;
        }
        Ok(map.insert(key, value))
    }

    /// A set has one owned key per table slot; duplicates require no growth.
    pub fn try_insert_set<T, S>(
        &self,
        set: &mut hashbrown::HashSet<T, S>,
        value: T,
    ) -> Result<bool, ParserStorageError>
    where
        T: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        if set.contains(&value) {
            return Ok(false);
        }
        let layout = set
            .try_reserve_layout(1)
            .map_err(|e| self.storage_error(Cause::Table(e)))?;
        let controls = [
            size_of::<(&Self, &mut hashbrown::HashSet<T, S>)>(),
            size_of::<T>(),
            size_of::<Option<Layout>>(),
            size_of::<ParserStorageError>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
            size_of::<Result<bool, ParserStorageError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_add(layout.map_or(0, |layout| layout.size())))
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.reserve(bytes)
            .map_err(|e| self.storage_error(Cause::Funding(e)))?;
        set.try_reserve(1)
            .map_err(|e| self.storage_error(Cause::Table(e)))?;
        if layout.is_some_and(|layout| layout.size() != set.allocation_size()) {
            return Err(self.storage_error(Cause::Destination));
        }
        Ok(set.insert(value))
    }

    /// Fund the actual box allocation before moving its initialized value.
    /// The surrounding constructor retains this account with the returned box.
    pub fn try_box<T>(&self, value: T) -> Result<Box<T>, ParserStorageError> {
        let controls = [
            size_of::<(&Self,)>(),
            size_of::<T>(),
            size_of::<Box<T>>(),
            size_of::<ParserStorageError>(),
            size_of::<Result<Box<T>, ParserStorageError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_add(Layout::new::<T>().size()))
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.reserve(bytes)
            .map_err(|e| self.storage_error(Cause::Funding(e)))?;
        Ok(Box::new(value))
    }

    /// Writes formatted output into a prepaid destination. Both passes use
    /// stack-only writers; a changing Display implementation cannot trigger
    /// implicit growth on the second pass.
    pub fn try_format(&self, args: fmt::Arguments<'_>) -> Result<String, ParserStorageError> {
        struct Count(usize);
        impl Write for Count {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                self.0 = self.0.checked_add(value.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        struct Output<'a>(&'a mut Vec<u8>);
        impl Write for Output<'_> {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                if value.len() > self.0.capacity() - self.0.len() {
                    return Err(fmt::Error);
                }
                self.0.extend_from_slice(value.as_bytes());
                Ok(())
            }
        }
        let controls = [
            size_of::<(&Self, fmt::Arguments<'_>)>(), size_of::<Count>(),
            size_of::<Output<'_>>(), size_of::<fmt::Arguments<'_>>(),
            size_of::<fmt::Result>(), size_of::<String>(), size_of::<Vec<u8>>(),
            size_of::<ParserStorageError>(), size_of::<Result<String, ParserStorageError>>(),
        ];
        let control_bytes = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| self.storage_error(Cause::Overflow))?;
        self.reserve(control_bytes).map_err(|error| self.storage_error(Cause::Funding(error)))?;
        let mut count = Count(0);
        count
            .write_fmt(args)
            .map_err(|_| self.storage_error(Cause::Overflow))?;
        let mut bytes = Vec::new();
        self.try_grow_vec(&mut bytes, count.0)?;
        Output(&mut bytes)
            .write_fmt(args)
            .map_err(|_| self.storage_error(Cause::Destination))?;
        if bytes.len() != count.0 {
            return Err(self.storage_error(Cause::Destination));
        }
        Ok(String::from_utf8(bytes).expect("formatted UTF-8 chunks"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    #[derive(Debug)]
    struct Refused;
    impl fmt::Display for Refused {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("refused")
        }
    }
    impl Error for Refused {}

    #[test]
    fn refused_growth_does_not_publish_or_allocate_and_keeps_original_cause() {
        let refused = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let funding = ParserAllocationFunding::prepare({
            let refused = refused.clone();
            let calls = calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                if refused.load(Ordering::SeqCst) {
                    Err(Refused)
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        let mut values = Vec::new();
        funding.try_extend_copy(&mut values, &[1, 2]).unwrap();
        let before = (
            values.as_ptr(),
            values.capacity(),
            calls.load(Ordering::SeqCst),
        );
        refused.store(true, Ordering::SeqCst);
        let error = funding.try_push(&mut values, 3).unwrap_err();
        assert_eq!(values, [1, 2]);
        assert_eq!((values.as_ptr(), values.capacity()), (before.0, before.1));
        assert_eq!(calls.load(Ordering::SeqCst), before.2 + 1);
        let original = error.source().unwrap().source().unwrap();
        assert!(original.is::<Refused>());
        drop(funding);
        assert!(error.source().unwrap().source().unwrap().is::<Refused>());
    }

    #[test]
    fn formatting_controls_refuse_before_the_first_display_call() {
        struct Counted(std::cell::Cell<usize>);
        impl fmt::Display for Counted {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.set(self.0.get() + 1);
                f.write_str("text")
            }
        }
        let refused = Arc::new(AtomicBool::new(false));
        let funding = ParserAllocationFunding::prepare({
            let refused = refused.clone();
            move |_| if refused.load(Ordering::SeqCst) { Err(Refused) } else { Ok(()) }
        }).unwrap();
        refused.store(true, Ordering::SeqCst);
        let display = Counted(std::cell::Cell::new(0));
        let error = funding.try_format(format_args!("{display}")).unwrap_err();
        assert_eq!(display.0.get(), 0);
        assert!(error.source().unwrap().source().unwrap().is::<Refused>());
    }

    #[test]
    fn formatting_cannot_grow_after_the_counted_pass() {
        struct Changes(std::cell::Cell<bool>);
        impl fmt::Display for Changes {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(if self.0.replace(true) { "longer" } else { "x" })
            }
        }
        let funding = ParserAllocationFunding::unenforced();
        assert_eq!(
            funding.try_format(format_args!("value {}", 17)).unwrap(),
            "value 17"
        );
        assert!(matches!(
            funding
                .try_format(format_args!("{}", Changes(false.into())))
                .unwrap_err()
                .cause,
            Cause::Destination
        ));
    }
}
