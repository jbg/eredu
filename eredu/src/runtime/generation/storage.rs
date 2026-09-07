//! Logical owned host data copied by a semantic snapshot. Counts inline values
//! and live owned strings/elements, excluding allocator capacity/node overhead and
//! shared immutable configuration. This is not a physical memory ceiling.
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) trait SnapshotStorage {
    fn heap_bytes(&self) -> Option<u64>;
    fn snapshot_bytes(&self) -> Option<u64> {
        (std::mem::size_of_val(self) as u64).checked_add(self.heap_bytes()?)
    }
}

// Destructuring every field makes adding state require an explicit sizing decision.
macro_rules! snapshot_fields {
    ($ty:ty { $($field:ident),* $(,)? }) => {
        impl $crate::runtime::generation::storage::SnapshotStorage for $ty {
            fn heap_bytes(&self) -> Option<u64> {
                let Self { $($field),* } = self;
                let bytes = 0u64;
                $(let bytes = bytes.checked_add($crate::runtime::generation::storage::SnapshotStorage::heap_bytes($field)?)?;)*
                Some(bytes)
            }
        }
    };
}
pub(crate) use snapshot_fields;

macro_rules! scalar {
    ($($ty:ty),*) => { $(impl SnapshotStorage for $ty { fn heap_bytes(&self) -> Option<u64> { Some(0) } })* };
}
scalar!(bool, char, u8, u32, u64, usize, i32, f32);
impl<T: ?Sized> SnapshotStorage for &'static T {
    fn heap_bytes(&self) -> Option<u64> {
        Some(0)
    }
}
impl SnapshotStorage for String {
    fn heap_bytes(&self) -> Option<u64> {
        u64::try_from(self.len()).ok()
    }
}
impl<T: SnapshotStorage> SnapshotStorage for Vec<T> {
    fn heap_bytes(&self) -> Option<u64> {
        self.iter()
            .try_fold(0u64, |n, value| n.checked_add(value.snapshot_bytes()?))
    }
}
impl<T: SnapshotStorage> SnapshotStorage for Option<T> {
    fn heap_bytes(&self) -> Option<u64> {
        self.as_ref().map_or(Some(0), SnapshotStorage::heap_bytes)
    }
}
impl<A: SnapshotStorage, B: SnapshotStorage> SnapshotStorage for (A, B) {
    fn heap_bytes(&self) -> Option<u64> {
        self.0.heap_bytes()?.checked_add(self.1.heap_bytes()?)
    }
}
impl<T: SnapshotStorage> SnapshotStorage for BTreeSet<T> {
    fn heap_bytes(&self) -> Option<u64> {
        self.iter()
            .try_fold(0u64, |n, v| n.checked_add(v.snapshot_bytes()?))
    }
}
fn map_bytes<'a, K: SnapshotStorage + 'a, V: SnapshotStorage + 'a>(
    mut entries: impl Iterator<Item = (&'a K, &'a V)>,
) -> Option<u64> {
    entries.try_fold(0u64, |n, (k, v)| {
        n.checked_add(std::mem::size_of::<(K, V)>() as u64)?
            .checked_add(k.heap_bytes()?)?
            .checked_add(v.heap_bytes()?)
    })
}
impl<K: SnapshotStorage, V: SnapshotStorage> SnapshotStorage for BTreeMap<K, V> {
    fn heap_bytes(&self) -> Option<u64> {
        map_bytes(self.iter())
    }
}
impl<K: SnapshotStorage, V: SnapshotStorage> SnapshotStorage for HashMap<K, V> {
    fn heap_bytes(&self) -> Option<u64> {
        map_bytes(self.iter())
    }
}
impl SnapshotStorage for serde_json::Map<String, serde_json::Value> {
    fn heap_bytes(&self) -> Option<u64> {
        map_bytes(self.iter())
    }
}
impl SnapshotStorage for serde_json::Value {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Null | Self::Bool(_) | Self::Number(_) => Some(0),
            Self::String(v) => v.heap_bytes(),
            Self::Array(v) => v.heap_bytes(),
            Self::Object(v) => v.heap_bytes(),
        }
    }
}
