use derivre::{ParserAllocationFunding, ParserStorageError, RandomState};
use std::hash::BuildHasher;
use std::{hash::Hash, num::NonZeroU32};

#[derive(Debug)]
pub struct HashId<T> {
    id: NonZeroU32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Clone for HashId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for HashId<T> {}

impl<T> PartialEq for HashId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for HashId<T> {}

impl<T> PartialOrd for HashId<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for HashId<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<T> Hash for HashId<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug)]
pub struct HashCons<T> {
    by_t: hashbrown::HashTable<NonZeroU32>,
    all_t: Vec<T>,
    hasher: RandomState,
    funding: ParserAllocationFunding,
}

impl<T: Eq + Hash> Default for HashCons<T> {
    fn default() -> Self {
        Self::new(ParserAllocationFunding::unenforced())
    }
}

impl<T: Eq + Hash> HashCons<T> {
    pub fn new(funding: ParserAllocationFunding) -> Self {
        Self {
            by_t: hashbrown::HashTable::new(),
            all_t: Vec::new(),
            hasher: RandomState::default(),
            funding,
        }
    }

    pub fn insert(&mut self, t: T) -> Result<HashId<T>, ParserStorageError> {
        let hash = self.hasher.hash_one(&t);
        if let Some(&id) = self
            .by_t
            .find(hash, |id| self.all_t[id.get() as usize - 1] == t)
        {
            return Ok(HashId {
                id,
                _marker: std::marker::PhantomData,
            });
        }
        let entries = self
            .all_t
            .len()
            .checked_add(1)
            .and_then(|n| u32::try_from(n).ok())
            .and_then(NonZeroU32::new)
            .ok_or_else(|| self.funding.storage_overflow())?;
        self.funding
            .try_grow_vec(&mut self.all_t, entries.get() as usize)?;
        self.funding.try_reserve_table(&mut self.by_t, 1, |id| {
            self.hasher.hash_one(&self.all_t[id.get() as usize - 1])
        })?;
        // Both destinations are reserved before either logical entry publishes.
        // The payload lives exactly once; table slots contain only its stable ID.
        self.all_t.push(t);
        self.by_t.insert_unique(hash, entries, |id| {
            self.hasher.hash_one(&self.all_t[id.get() as usize - 1])
        });
        Ok(HashId {
            id: entries,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn get(&self, id: HashId<T>) -> &T {
        &self.all_t[id.id.get() as usize - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct Key(&'static str);
    #[derive(Debug)]
    struct Refused;
    impl std::fmt::Display for Refused {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("refused")
        }
    }
    impl std::error::Error for Refused {}

    #[test]
    fn nonclone_payload_is_stored_once_and_refused_growth_preserves_ids() {
        let refused = Arc::new(AtomicBool::new(false));
        let funding = ParserAllocationFunding::prepare({
            let refused = refused.clone();
            move |_| {
                if refused.load(Ordering::SeqCst) {
                    Err(Refused)
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        let mut table = HashCons::new(funding);
        let first = table.insert(Key("first")).unwrap();
        refused.store(true, Ordering::SeqCst);
        assert_eq!(table.insert(Key("first")).unwrap(), first);
        let error = table.insert(Key("second")).unwrap_err();
        assert_eq!(table.get(first), &Key("first"));
        assert_eq!((table.all_t.len(), table.by_t.len()), (1, 1));
        drop(table);
        let cause = std::error::Error::source(&error).unwrap();
        assert!(cause.source().unwrap().is::<Refused>());
    }
}
