//! Final sorted manager rows. Keys never change after constructor validation.
use std::{borrow::Borrow, ops::Index};

#[cfg_attr(test, derive(Clone))]
pub(super) struct Rows<K, V>(Vec<(K, V)>);
impl<K: Ord, V> FromIterator<(K, V)> for Rows<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut rows: Vec<_> = iter.into_iter().collect();
        rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        assert!(
            rows.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "validated unique manager rows"
        );
        Self(rows)
    }
}
impl<K: Ord, V> Rows<K, V> {
    /// Consume final rows whose unique order was validated by the owning source.
    /// No allocation, key clone, sorting scratch or payload conversion occurs.
    pub(super) fn from_sorted(rows: Vec<(K, V)>) -> Self {
        assert!(rows.windows(2).all(|pair| pair[0].0 < pair[1].0));
        Self(rows)
    }
    fn position<Q: Ord + ?Sized>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q>,
    {
        self.0
            .binary_search_by(|(stored, _)| stored.borrow().cmp(key))
    }
    pub(super) fn get<Q: Ord + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.position(key).ok().map(|position| &self.0[position].1)
    }
    pub(super) fn get_mut<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.position(key)
            .ok()
            .map(|position| &mut self.0[position].1)
    }
    pub(super) fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.0.iter())
    }
    pub(super) fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.0.iter().map(|(_, value)| value)
    }
    pub(super) fn iter(&self) -> std::slice::Iter<'_, (K, V)> {
        self.0.iter()
    }
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    pub(super) fn requested_layout(count: usize) -> Option<std::alloc::Layout> {
        std::alloc::Layout::array::<(K, V)>(count).ok()
    }
    #[cfg(test)]
    pub(super) fn remove(&mut self, key: &K) -> Option<V> {
        self.position(key)
            .ok()
            .map(|position| self.0.remove(position).1)
    }
    pub(super) fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.position(&key) {
            Ok(position) => Some(std::mem::replace(&mut self.0[position].1, value)),
            Err(position) => {
                self.0.insert(position, (key, value));
                None
            }
        }
    }
}
impl<'a, K, V> IntoIterator for &'a Rows<K, V> {
    type Item = &'a (K, V);
    type IntoIter = std::slice::Iter<'a, (K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<K: Ord, Q: Ord + ?Sized, V> Index<&Q> for Rows<K, V>
where
    K: Borrow<Q>,
{
    type Output = V;
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("validated manager row")
    }
}

/// Canonical-owner pins have a fixed key domain: one bit per unit and tier.
/// Ordinary and prepared acquisition share this same bounded update worker.
pub(super) struct AliasPins {
    rows: Vec<((super::OffloadUnitId, super::MemoryTier), bool)>,
}
impl AliasPins {
    pub(super) fn new<'a>(ids: impl ExactSizeIterator<Item = &'a super::OffloadUnitId>) -> Self {
        let mut rows =
            Vec::with_capacity(ids.len().checked_mul(3).expect("validated pin population"));
        for id in ids {
            for tier in [
                super::MemoryTier::Disk,
                super::MemoryTier::Host,
                super::MemoryTier::Device,
            ] {
                rows.push(((id.clone(), tier), false));
            }
        }
        rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Self { rows }
    }
    pub(super) fn position(
        &self,
        id: &super::OffloadUnitId,
        tier: super::MemoryTier,
    ) -> Option<usize> {
        self.rows
            .binary_search_by(|((stored_id, stored_tier), _)| {
                stored_id.cmp(id).then(stored_tier.cmp(&tier))
            })
            .ok()
    }
    pub(super) fn contains(&self, id: &super::OffloadUnitId, tier: super::MemoryTier) -> bool {
        self.position(id, tier)
            .is_some_and(|index| self.rows[index].1)
    }
    pub(super) fn id(&self, index: usize) -> &super::OffloadUnitId {
        &self.rows[index].0 .0
    }
    pub(super) fn is_pinned(&self, index: usize) -> bool {
        self.rows[index].1
    }
    pub(super) fn pin(&mut self, index: usize) {
        self.rows[index].1 = true;
    }
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.rows.iter().all(|(_, pinned)| !pinned)
    }
    pub(super) fn requested_layout(units: usize) -> Option<std::alloc::Layout> {
        std::alloc::Layout::array::<((super::OffloadUnitId, super::MemoryTier), bool)>(
            units.checked_mul(3)?,
        )
        .ok()
    }
}

pub(super) struct Keys<'a, K, V>(std::slice::Iter<'a, (K, V)>);
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(key, _)| key)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K, V> ExactSizeIterator for Keys<'_, K, V> {}
