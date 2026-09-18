//! One canonical cache catalog with ordinary or prepaid storage.
use std::{collections::BTreeMap, mem::size_of, ops::Index};

use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding},
};

/// Two borrowed access paths to the same counted metadata helpers. This adds
/// no account or grant; prepared destinations retain the supplied owner.
#[derive(Clone, Copy)]
pub(super) enum MetadataSource<'a> {
    Context(&'a WorkspaceContext),
    Funding(&'a HostMetadataFunding),
}
impl MetadataSource<'_> {
    pub(super) fn charge(self, bytes: usize) -> Result<(), Error> {
        match self {
            Self::Context(context) => context.charge_metadata(bytes).map_err(Into::into),
            Self::Funding(funding) => funding.reserve_metadata(bytes)
                .map_err(WorkspaceMetadataError::from).map_err(Into::into),
        }
    }
    pub(super) fn vector<T>(self, count: usize) -> Result<Vec<T>, Error> {
        match self {
            Self::Context(context) => context.metadata_vec(count),
            Self::Funding(funding) => funding.metadata_vec(count),
        }
    }
    pub(super) fn funding(self) -> Option<HostMetadataFunding> {
        match self {
            Self::Context(context) => context.metadata_funding(),
            Self::Funding(funding) => Some(funding.clone()),
        }
    }
}

/// A checked cache destination cannot accept this mutation. This descriptor
/// carries no source, execution, transfer or completion authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CacheTableCapacityError {
    /// The same canonical table has not received a prepared destination.
    #[error("cache catalog storage was not prepared")]
    Unprepared,
    /// The explicit limit is exhausted, irrespective of allocator overcapacity.
    #[error("prepared cache catalog capacity is exhausted")]
    Exhausted,
}

/// Empty, paid storage for one exact maximum catalog population. The storage
/// retires before its host account. Payload owners must retain their own custody.
#[derive(Debug)]
pub struct PreparedCacheTable<K, V> {
    entries: Vec<(K, V)>,
    maximum: usize,
    _funding: Option<HostMetadataFunding>,
}
impl<K, V> PreparedCacheTable<K, V> {
    /// Explicit maximum population, independent of allocator overcapacity.
    pub const fn maximum(&self) -> usize {
        self.maximum
    }

    /// Complete destination backing and its constructor/installation controls.
    pub fn control_bytes(maximum: usize) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<MetadataSource<'_>>(),
            size_of::<CacheRecordTable<K, V>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<CacheRecordTable<K, V>, Self>>(),
            size_of::<Option<HostMetadataFunding>>(),
            size_of::<(K, V)>(),
            size_of::<std::collections::btree_map::IntoIter<K, V>>(),
            size_of::<std::vec::IntoIter<(K, V)>>(),
            size_of::<CacheTableCapacityError>(),
            size_of::<Result<Option<V>, (CacheTableCapacityError, K, V)>>(),
            size_of::<Result<usize, usize>>(),
        ];
        frames.into_iter().try_fold(
            std::mem::size_of_val(&frames)
                .checked_add(WorkspaceContext::metadata_vec_bytes::<(K, V)>(maximum)?)?,
            usize::checked_add,
        )
    }

    /// Reserves all controls before allocating the final empty inventory.
    /// Preparation is metadata only; it grants no permission to use its values.
    pub fn prepare(maximum: usize, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::prepare_from(maximum, MetadataSource::Context(context))
    }
    /// Same final destination worker against an already retained Host account.
    pub fn prepare_with_funding(maximum: usize, funding: &HostMetadataFunding) -> Result<Self, Error> {
        Self::prepare_from(maximum, MetadataSource::Funding(funding))
    }
    pub(super) fn prepare_from(maximum: usize, source: MetadataSource<'_>) -> Result<Self, Error> {
        let backing = WorkspaceContext::metadata_vec_bytes::<(K, V)>(maximum)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let controls = Self::control_bytes(maximum)
            .and_then(|all| all.checked_sub(backing))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        source.charge(controls)?;
        let entries = source.vector(maximum)?;
        Ok(Self {
            entries,
            maximum,
            _funding: source.funding(),
        })
    }
}

/// Authoritative ordered cache records. Preparing storage moves the same
/// records, including lease/source identity; it does not create a second catalog.
/// Ordinary insertion preserves ordinary growing behavior. Original callers
/// must use the checked operations after separately authenticating their source.
#[derive(Debug)]
pub struct CacheRecordTable<K, V>(Storage<K, V>);
#[derive(Debug)]
enum Storage<K, V> {
    Growing(BTreeMap<K, V>, Option<HostMetadataFunding>),
    Prepared(PreparedCacheTable<K, V>),
}
impl<K, V> Default for CacheRecordTable<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K, V> CacheRecordTable<K, V> {
    /// Creates the existing ordinary growing catalog.
    pub const fn new() -> Self {
        Self(Storage::Growing(BTreeMap::new(), None))
    }
    /// Moves an ordinary ordered map into the shared catalog representation.
    /// No record or allocation is copied by this compatibility constructor.
    pub fn from_ordered_map(entries: BTreeMap<K, V>) -> Self {
        Self(Storage::Growing(entries, None))
    }

    /// Fixed frames for one checked insertion/removal or capacity validation.
    /// An enclosing operation prices each attempted call; this is not credit.
    pub fn mutation_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<&mut Self>(),
            size_of::<K>(),
            size_of::<V>(),
            size_of::<Option<V>>(),
            size_of::<Result<Option<V>, (CacheTableCapacityError, K, V)>>(),
            size_of::<CacheTableCapacityError>(),
            size_of::<Result<(), CacheTableCapacityError>>(),
            size_of::<Result<usize, usize>>(),
            size_of::<Option<usize>>(),
            size_of::<(&K, &K)>(),
            size_of::<std::cmp::Ordering>(),
            size_of::<(usize, usize)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Current canonical population.
    pub fn len(&self) -> usize {
        match &self.0 {
            Storage::Growing(v, _) => v.len(),
            Storage::Prepared(v) => v.entries.len(),
        }
    }
    /// Whether no records are present.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Checks a final population against the explicit prepared capacity.
    pub fn validate_prepared_population(
        &self,
        population: usize,
    ) -> Result<(), CacheTableCapacityError> {
        match &self.0 {
            Storage::Growing(_, _) => Err(CacheTableCapacityError::Unprepared),
            Storage::Prepared(v) if population <= v.maximum => Ok(()),
            Storage::Prepared(_) => Err(CacheTableCapacityError::Exhausted),
        }
    }
    /// Installs an empty prepared destination without copying any key or value.
    /// Refusal returns its unchanged paid storage and leaves the table untouched.
    /// Success returns the now-empty prior storage, so its backing and account
    /// can retire after the caller releases any manager/source lock.
    pub fn install(
        &mut self,
        mut destination: PreparedCacheTable<K, V>,
    ) -> Result<Self, PreparedCacheTable<K, V>> {
        if self.len() > destination.maximum {
            return Err(destination);
        }
        let mut prior = std::mem::replace(&mut self.0, Storage::Growing(BTreeMap::new(), None));
        match &mut prior {
            Storage::Growing(map, _) => destination.entries.extend(std::mem::take(map)),
            Storage::Prepared(old) => destination.entries.append(&mut old.entries),
        }
        self.0 = Storage::Prepared(destination);
        Ok(Self(prior))
    }
    /// Removes records while preserving an installed destination and its custody.
    pub fn clear(&mut self) {
        match &mut self.0 {
            Storage::Growing(v, _) => v.clear(),
            Storage::Prepared(v) => v.entries.clear(),
        }
    }
    /// Borrowed identity-ordered traversal shared by both physical stores.
    pub fn iter(&self) -> CacheRecordTableIter<'_, K, V> {
        match &self.0 {
            Storage::Growing(v, _) => CacheRecordTableIter::Growing(v.iter()),
            Storage::Prepared(v) => CacheRecordTableIter::Prepared(v.entries.iter()),
        }
    }
    /// Identity-ordered keys without allocation.
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator {
        self.iter().map(|(k, _)| k)
    }
    /// Identity-ordered records without allocation.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator {
        self.iter().map(|(_, v)| v)
    }
    /// Mutable records without changing their identities or physical slots.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        enum Values<'a, K, V> {
            Growing(std::collections::btree_map::ValuesMut<'a, K, V>),
            Prepared(std::slice::IterMut<'a, (K, V)>),
        }
        impl<'a, K, V> Iterator for Values<'a, K, V> {
            type Item = &'a mut V;
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::Growing(v) => v.next(),
                    Self::Prepared(v) => v.next().map(|(_, v)| v),
                }
            }
        }
        match &mut self.0 {
            Storage::Growing(v, _) => Values::Growing(v.values_mut()),
            Storage::Prepared(v) => Values::Prepared(v.entries.iter_mut()),
        }
    }
}
impl<K: Ord, V> CacheRecordTable<K, V> {
    /// Looks up the same exact key in either canonical storage form.
    pub fn get(&self, key: &K) -> Option<&V> {
        match &self.0 {
            Storage::Growing(v, _) => v.get(key),
            Storage::Prepared(v) => v
                .entries
                .binary_search_by(|(k, _)| k.cmp(key))
                .ok()
                .map(|i| &v.entries[i].1),
        }
    }
    /// Mutably looks up the same exact key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match &mut self.0 {
            Storage::Growing(v, _) => v.get_mut(key),
            Storage::Prepared(v) => v
                .entries
                .binary_search_by(|(k, _)| k.cmp(key))
                .ok()
                .map(|i| &mut v.entries[i].1),
        }
    }
    /// Whether this identity is currently present.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
    /// Ordinary insertion, including ordinary growth after a prepared period.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Storage::Prepared(v) = &mut self.0 {
            match v.entries.binary_search_by(|(k, _)| k.cmp(&key)) {
                Ok(i) => return Some(std::mem::replace(&mut v.entries[i].1, value)),
                Err(i) if v.entries.len() < v.maximum => {
                    v.entries.insert(i, (key, value));
                    return None;
                }
                Err(_) => {}
            }
        }
        self.grow_ordinary();
        match &mut self.0 {
            Storage::Growing(v, _) => v.insert(key, value),
            Storage::Prepared(_) => unreachable!(),
        }
    }
    fn grow_ordinary(&mut self) {
        if let Storage::Prepared(previous) = &mut self.0 {
            let mut map = BTreeMap::new();
            map.extend(previous.entries.drain(..));
            let funding = previous._funding.take();
            self.0 = Storage::Growing(map, funding);
        }
    }

    /// Inserts without allocating. Refusal retains the actual unaccepted value
    /// for the caller to retire outside any source/manager loan.
    pub fn insert_prepared(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (CacheTableCapacityError, K, V)> {
        let additional = usize::from(!self.contains_key(&key));
        let population = self.len().checked_add(additional);
        let check = population
            .ok_or(CacheTableCapacityError::Exhausted)
            .and_then(|n| self.validate_prepared_population(n));
        match check {
            Ok(()) => Ok(self.insert(key, value)),
            Err(cause) => Err((cause, key, value)),
        }
    }
    /// Removes the canonical value without shrinking an installed inventory.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match &mut self.0 {
            Storage::Growing(v, _) => v.remove(key),
            Storage::Prepared(v) => v
                .entries
                .binary_search_by(|(k, _)| k.cmp(key))
                .ok()
                .map(|i| v.entries.remove(i).1),
        }
    }
}
impl<K: Ord, V> Index<&K> for CacheRecordTable<K, V> {
    type Output = V;
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("cache catalog key is present")
    }
}

/// Allocation-free traversal of the one canonical ordered inventory.
pub enum CacheRecordTableIter<'a, K, V> {
    /// Ordinary map storage.
    Growing(std::collections::btree_map::Iter<'a, K, V>),
    /// Exact prepared contiguous storage.
    Prepared(std::slice::Iter<'a, (K, V)>),
}
impl<'a, K, V> Iterator for CacheRecordTableIter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Growing(v) => v.next(),
            Self::Prepared(v) => v.next().map(|(k, v)| (k, v)),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Growing(v) => v.size_hint(),
            Self::Prepared(v) => v.size_hint(),
        }
    }
}
impl<K, V> DoubleEndedIterator for CacheRecordTableIter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Growing(v) => v.next_back(),
            Self::Prepared(v) => v.next_back().map(|(k, v)| (k, v)),
        }
    }
}
impl<K, V> ExactSizeIterator for CacheRecordTableIter<'_, K, V> {}
impl<K, V> std::iter::FusedIterator for CacheRecordTableIter<'_, K, V> {}

impl<K: Ord, V: Default> CacheRecordTable<K, V> {
    /// Returns an existing value, or inserts its ordinary default. Original
    /// callers first validate the exact final population against paid storage.
    /// This worker preserves ordinary growing behavior outside that contract.
    pub fn get_or_default(&mut self, key: K) -> &mut V {
        let grow = match &self.0 {
            Storage::Prepared(entries) => entries.entries.len() == entries.maximum
                && entries.entries.binary_search_by(|(candidate, _)| candidate.cmp(&key)).is_err(),
            Storage::Growing(_, _) => false,
        };
        if grow { self.grow_ordinary(); }
        match &mut self.0 {
            Storage::Growing(entries, _) => entries.entry(key).or_default(),
            Storage::Prepared(entries) => {
                let index = match entries.entries.binary_search_by(|(candidate, _)| candidate.cmp(&key)) {
                    Ok(index) => index,
                    Err(index) => { entries.entries.insert(index, (key, V::default())); index }
                };
                &mut entries.entries[index].1
            }
        }
    }
}

impl<K, V> Clone for CacheRecordTableIter<'_, K, V> {
    fn clone(&self) -> Self {
        match self {
            Self::Growing(values) => Self::Growing(values.clone()),
            Self::Prepared(values) => Self::Prepared(values.clone()),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
