//! Shared recursive memoization with the exact prospective table request.
use super::workspace::{Error, Workspace};
use ahash::{AHasher, RandomState};
use hashbrown::{HashTable, TryReserveError};
use std::{
    alloc::Layout,
    borrow::Borrow,
    hash::{BuildHasher, Hash},
    mem::{size_of, size_of_val},
};

pub(crate) struct Cache<K, V> {
    rows: HashTable<(K, V)>,
    hasher: RandomState,
}
impl<K: Hash + Eq, V> Cache<K, V> {
    pub(crate) fn new(hasher: RandomState) -> Self {
        Self {
            rows: HashTable::new(),
            hasher,
        }
    }
    pub(crate) fn get<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.rows
            .find(self.hasher.hash_one(key), |(stored, _)| {
                stored.borrow() == key
            })
            .map(|(_, value)| value)
    }
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }
    pub(crate) fn reserve(&mut self, additional: usize, workspace: &mut Workspace<'_>) -> bool {
        let parts = [
            size_of::<Self>(),
            size_of::<usize>(),
            size_of::<&RandomState>(),
            size_of::<AHasher>(),
            size_of::<Result<Option<Layout>, TryReserveError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Error>(),
            size_of::<(&mut Self, usize, &mut Workspace<'_>)>(),
        ];
        if !workspace.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        ) {
            return false;
        }
        if workspace.original() {
            let request = match self.rows.try_reserve_layout(additional) {
                Ok(request) => request,
                Err(error) => return workspace.refuse(Error::Table(error)),
            };
            if let Some(layout) = request {
                if !workspace.reserve(Some(layout.size())) {
                    return false;
                }
            }
            let hasher = &self.hasher;
            if let Err(error) = self
                .rows
                .try_reserve(additional, |(key, _)| hasher.hash_one(key))
            {
                return workspace.refuse(Error::Table(error));
            }
        } else {
            let hasher = &self.hasher;
            self.rows
                .reserve(additional, |(key, _)| hasher.hash_one(key));
        }
        true
    }
    /// Replacement does not grow. On a new key, both ordinary and funded calls
    /// use the same table worker; the funded call first pays its complete request.
    pub(crate) fn insert(&mut self, key: K, value: V, workspace: &mut Workspace<'_>) -> bool {
        if workspace.failed() {
            return false;
        }
        let parts = [
            size_of::<Self>(),
            size_of::<(K, V)>(),
            size_of::<RandomState>(),
            size_of::<AHasher>(),
            size_of::<u64>(),
            size_of::<Option<&mut (K, V)>>(),
            size_of::<Result<Option<Layout>, TryReserveError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Error>(),
            size_of::<hashbrown::hash_table::OccupiedEntry<'_, (K, V)>>(),
            size_of::<(&mut Self, &mut Workspace<'_>, &K, &RandomState)>(),
        ];
        if !workspace.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        ) {
            return false;
        }
        let hash = self.hasher.hash_one(&key);
        if let Some((_, stored)) = self.rows.find_mut(hash, |(stored, _)| stored == &key) {
            *stored = value;
            return true;
        }
        if !self.reserve(1, workspace) {
            return false;
        }
        let hasher = &self.hasher;
        self.rows
            .insert_unique(hash, (key, value), |(key, _)| hasher.hash_one(key));
        true
    }
}
