use serde::{ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
use std::{collections::BTreeMap, ops::Index};

/// Immutable, sorted input-identity entries in one exact-sized allocation.
///
/// Lookup and iteration follow ordered-map semantics. Serialization remains a
/// map; physical capacity inspection does not construct a map or clone values.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InputIdentityMap<K, V>(Box<[(K, V)]>);

impl<K, V> InputIdentityMap<K, V> {
    pub(super) fn empty() -> Self {
        Self(Box::new([]))
    }

    pub(super) fn from_map(map: BTreeMap<K, V>) -> Self {
        Self(map.into_iter().collect::<Vec<_>>().into_boxed_slice())
    }

    /// Number of retained entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this map retains any entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Entries in ascending key order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + DoubleEndedIterator {
        self.0.iter().map(|(key, value)| (key, value))
    }

    /// Keys in ascending order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> + DoubleEndedIterator {
        self.0.iter().map(|(key, _)| key)
    }

    /// Values in ascending key order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + DoubleEndedIterator {
        self.0.iter().map(|(_, value)| value)
    }

    pub(super) fn capacity_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<(K, V)>().checked_mul(self.0.len())?).ok()
    }
}

impl<K: Ord, V> InputIdentityMap<K, V> {
    /// Consumes already allocated entries without creating a map or copying
    /// values. Rejection returns the exact allocation to its current owner.
    /// This is a representation constructor, not allocation authority.
    pub fn from_sorted_entries(entries: Box<[(K, V)]>) -> Result<Self, Box<[(K, V)]>> {
        if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            Err(entries)
        } else {
            Ok(Self(entries))
        }
    }
    /// Ordinary normalization from an existing ordered map. Construction can
    /// allocate; closed original workers use already reserved fixed entries.
    pub fn from_ordered_map(map: BTreeMap<K, V>) -> Self {
        Self::from_map(map)
    }
    /// Looks up an existing entry without allocating.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.0
            .binary_search_by(|(candidate, _)| candidate.cmp(key))
            .ok()
            .map(|index| &self.0[index].1)
    }

    /// Whether an entry exists for this key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}

impl<K: Ord, V> Index<&K> for InputIdentityMap<K, V> {
    type Output = V;
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("input identity key is absent")
    }
}

impl<'a, K, V> IntoIterator for &'a InputIdentityMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::Map<std::slice::Iter<'a, (K, V)>, fn(&'a (K, V)) -> (&'a K, &'a V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(|(key, value)| (key, value))
    }
}

impl<K: Serialize, V: Serialize> Serialize for InputIdentityMap<K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, value) in self {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Deserialize<'de>
    for InputIdentityMap<K, V>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Preserve the BTreeMap serde representation and duplicate-key
        // behavior. Validated public constructors and wire decoding separately
        // enforce the modality and uniqueness rules for prepared input.
        BTreeMap::deserialize(deserializer).map(Self::from_map)
    }
}

#[cfg(test)]
mod fixed_entries_tests {
    use super::*;
    #[test]
    fn sorted_map_rejection_returns_same_nonclone_values_and_allocation() {
        struct Value(u8);
        for keys in [[1u8, 1], [2, 1]] {
            let entries: Box<[(u8, Value)]> = Box::new([(keys[0], Value(7)), (keys[1], Value(19))]);
            let pointer = entries.as_ptr();
            let returned = match InputIdentityMap::from_sorted_entries(entries) {
                Err(entries) => entries,
                Ok(_) => panic!("unordered map accepted"),
            };
            assert_eq!(pointer, returned.as_ptr());
            assert_eq!(returned[0].1 .0, 7);
            assert_eq!(returned[1].1 .0, 19);
        }
        let entries: Box<[(u8, Value)]> = Box::new([(1, Value(7)), (2, Value(19))]);
        let pointer = &entries[0].1 as *const Value;
        let map = match InputIdentityMap::from_sorted_entries(entries) {
            Ok(map) => map,
            Err(_) => panic!("ordered map rejected"),
        };
        assert_eq!(map.get(&1).unwrap() as *const Value, pointer);
        assert_eq!(
            map.iter().map(|(k, v)| (*k, v.0)).collect::<Vec<_>>(),
            [(1, 7), (2, 19)]
        );
    }
}
