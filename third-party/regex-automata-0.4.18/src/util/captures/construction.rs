//! One-pattern anonymous capture metadata for the private source constructor.
//!
//! This module has no account API. Its caller must derive `Plan` from the
//! selected immutable source recipe before allocating any construction storage.

use alloc::{collections::TryReserveError, sync::Arc, vec::Vec};
use core::{alloc::Layout, fmt, mem, sync::atomic::AtomicUsize};

use super::{CaptureNameMap, GroupInfo, GroupInfoInner};
use crate::util::primitives::SmallIndex;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Buffer {
    Ranges,
    Maps,
    Rows,
    Names,
}

impl Buffer {
    const ALL: [Self; 4] = [Self::Ranges, Self::Maps, Self::Rows, Self::Names];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("anonymous capture construction geometry overflow")
    }
}

#[derive(Debug)]
pub(crate) struct Plan {
    groups: usize,
    end: SmallIndex,
    names_bytes: usize,
    required_bytes: usize,
    heap_bytes: usize,
}

impl Plan {
    pub(crate) fn new(groups: usize) -> Result<Self, Overflow> {
        if groups == 0 {
            return Err(Overflow);
        }
        let slots = groups.checked_mul(2).ok_or(Overflow)?;
        let end = SmallIndex::new(slots).map_err(|_| Overflow)?;
        let names_bytes = array::<Option<Arc<str>>>(groups)?;
        let heap_bytes = [
            array::<(SmallIndex, SmallIndex)>(1)?,
            array::<Option<CaptureNameMap>>(1)?,
            array::<Vec<Option<Arc<str>>>>(1)?,
            names_bytes,
            arc_layout::<GroupInfoInner>()?.size(),
        ]
        .iter()
        .try_fold(0usize, |sum, &bytes| {
            sum.checked_add(bytes).ok_or(Overflow)
        })?;
        let required_bytes = [
            heap_bytes,
            mem::size_of::<Plan>(),
            mem::size_of::<Result<Plan, Overflow>>(),
            mem::size_of::<Storage>(),
            mem::size_of::<Failure>(),
            mem::size_of::<Result<GroupInfo, Failure>>(),
            mem::size_of::<Result<(), TryReserveError>>(),
            mem::size_of::<GroupInfoInner>(),
            mem::size_of::<GroupInfo>(),
            mem::size_of::<Buffer>(),
        ]
        .iter()
        .try_fold(0usize, |sum, &bytes| {
            sum.checked_add(bytes).ok_or(Overflow)
        })?;
        Ok(Self { groups, end, names_bytes, required_bytes, heap_bytes })
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.heap_bytes
    }

    pub(crate) fn required_bytes(&self) -> usize {
        self.required_bytes
    }

    pub(crate) fn prepare(
        self,
        failure: Option<Buffer>,
    ) -> Result<GroupInfo, Failure> {
        let mut storage = Storage::new();
        for buffer in Buffer::ALL {
            let expected = match buffer {
                Buffer::Names => self.groups,
                _ => 1,
            };
            let requested =
                if failure == Some(buffer) { usize::MAX } else { expected };
            let result = match buffer {
                Buffer::Ranges => {
                    storage.inner.slot_ranges.try_reserve_exact(requested)
                }
                Buffer::Maps => {
                    storage.inner.name_to_index.try_reserve_exact(requested)
                }
                Buffer::Rows => {
                    storage.inner.index_to_name.try_reserve_exact(requested)
                }
                Buffer::Names => storage.names.try_reserve_exact(requested),
            };
            if let Err(cause) = result {
                return Err(Failure {
                    buffer,
                    cause: Cause::Reserve(cause),
                    storage,
                });
            }
            let actual = match buffer {
                Buffer::Ranges => storage.inner.slot_ranges.capacity(),
                Buffer::Maps => storage.inner.name_to_index.capacity(),
                Buffer::Rows => storage.inner.index_to_name.capacity(),
                Buffer::Names => storage.names.capacity(),
            };
            if actual != expected {
                return Err(Failure {
                    buffer,
                    cause: Cause::Capacity,
                    storage,
                });
            }
        }
        storage.inner.slot_ranges.push((
            SmallIndex::new(2).expect("validated nonempty group range"),
            self.end,
        ));
        // None is the closed anonymous mode. No RandomState, map buckets or
        // named Arc strings are constructed along this path.
        storage.inner.name_to_index.push(None);
        storage.names.resize_with(self.groups, || None);
        storage.inner.memory_extra = self.names_bytes;
        storage.inner.index_to_name.push(mem::take(&mut storage.names));
        // The complete Arc request is included above. Arc::new is the pinned
        // infallible allocation mechanism; allocator OOM is not a reserve error.
        Ok(GroupInfo(Arc::new(storage.inner)))
    }
}

#[derive(Debug)]
struct Storage {
    inner: GroupInfoInner,
    names: Vec<Option<Arc<str>>>,
}

impl Storage {
    fn new() -> Self {
        Self {
            // GroupInfo::default would allocate a temporary Arc. Construct
            // the actual final inner fields instead.
            inner: GroupInfoInner {
                slot_ranges: Vec::new(),
                name_to_index: Vec::new(),
                index_to_name: Vec::new(),
                memory_extra: 0,
            },
            names: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Cause {
    Reserve(TryReserveError),
    Capacity,
}

#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) buffer: Buffer,
    pub(crate) cause: Cause,
    storage: Storage,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve(error) => error.fmt(f),
            Cause::Capacity => {
                f.write_str("anonymous capture allocation capacity mismatch")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve(error) => Some(error),
            Cause::Capacity => None,
        }
    }
}

pub(crate) fn array<T>(count: usize) -> Result<usize, Overflow> {
    Layout::array::<T>(count).map(|layout| layout.size()).map_err(|_| Overflow)
}

/// Mechanism assumption tied to pinned Rust 1.98 alloc::sync::ArcInner:
/// repr(C) { strong: AtomicUsize, weak: AtomicUsize, data: T }.
/// This is a private diagnostic, not a portable promise about opaque Arc ABI.
pub(crate) fn arc_layout<T>() -> Result<Layout, Overflow> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align())
        .map_err(|_| Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::primitives::PatternID;

    #[test]
    fn anonymous_source_metadata_preserves_actual_capture_queries() {
        for groups in [1, 3, 17] {
            let actual = Plan::new(groups).unwrap().prepare(None).unwrap();
            let ordinary =
                GroupInfo::new(
                    [core::iter::repeat(None::<&str>).take(groups)],
                )
                .unwrap();
            assert_eq!(actual.pattern_len(), ordinary.pattern_len());
            assert_eq!(actual.slot_len(), ordinary.slot_len());
            for pid in [PatternID::ZERO, PatternID::new(1).unwrap()] {
                assert_eq!(actual.group_len(pid), ordinary.group_len(pid));
                assert_eq!(
                    actual.to_index(pid, "absent"),
                    ordinary.to_index(pid, "absent")
                );
                for group in 0..groups + 1 {
                    assert_eq!(
                        actual.slots(pid, group),
                        ordinary.slots(pid, group)
                    );
                    assert_eq!(
                        actual.to_name(pid, group),
                        ordinary.to_name(pid, group)
                    );
                }
            }
            assert!(actual.0.name_to_index[0].is_none());
            assert!(ordinary.0.name_to_index[0].is_some());
        }
        let named = GroupInfo::new([[None, Some("name")]]).unwrap();
        assert_eq!(named.to_index(PatternID::ZERO, "name"), Some(1));
    }

    #[test]
    fn each_actual_capture_reserve_failure_keeps_earlier_buffers() {
        for (index, buffer) in Buffer::ALL.into_iter().enumerate() {
            let failure =
                Plan::new(3).unwrap().prepare(Some(buffer)).unwrap_err();
            assert!(matches!(&failure.cause, Cause::Reserve(_)));
            assert_eq!(failure.buffer, buffer);
            let storage = &failure.storage;
            let capacities = [
                storage.inner.slot_ranges.capacity(),
                storage.inner.name_to_index.capacity(),
                storage.inner.index_to_name.capacity(),
                storage.names.capacity(),
            ];
            for (slot, actual) in capacities.into_iter().enumerate() {
                assert_eq!(actual, if slot < index { 1 } else { 0 });
            }
            assert!(storage.inner.slot_ranges.is_empty());
            assert!(storage.names.is_empty());
        }
    }

    #[test]
    fn invalid_capture_geometry_rejects_before_any_storage_exists() {
        assert!(Plan::new(0).is_err());
        assert!(Plan::new(usize::MAX).is_err());
        assert!(Plan::new(SmallIndex::MAX.as_usize()).is_err());
    }
}
