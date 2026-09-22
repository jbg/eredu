//! Outer type directory only; each value remains the one canonical Registry<K>.
use super::{Registry, WorkingMemoryError};
use crate::working_memory::{OriginalHostMetadataCustody, funding::RawSpanHostOwner};
use std::{
    any::{Any, TypeId},
    mem::size_of,
};

type Value = Box<dyn Any + Send>;

#[derive(Default)]
pub(in crate::working_memory) struct Directory {
    head: Option<PreparedNamespace>,
}
struct Node {
    kind: TypeId,
    value: Value,
    next: Directory,
}
// The raw hold is outside the Box: physical Registry/Node deallocation precedes
// its final release. It owns no P partition, source table or native handle.
pub(super) struct PreparedNamespace {
    node: Box<Node>,
    _custody: MetadataCustody,
}
struct MetadataCustody {
    raw: Option<OriginalHostMetadataCustody>,
    _preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl Drop for MetadataCustody {
    fn drop(&mut self) {
        // This field is reached after node/key destruction, including if a
        // provider destructor began unwinding with another raw alias alive.
        if std::thread::panicking() {
            if let Some(raw) = &self.raw {
                raw.quarantine();
            }
        }
    }
}
impl PreparedNamespace {
    pub(super) fn prepare<K: Ord + Send + 'static>(raw: Option<RawSpanHostOwner>) -> Self {
        // Both allocations occur before the caller borrows Usage. The local
        // hold covers partial construction; no provider key exists yet.
        Self::prepare_source::<K>(raw.map(Into::into))
    }
    pub(super) fn prepare_source<K: Ord + Send + 'static>(
        raw: Option<OriginalHostMetadataCustody>,
    ) -> Self {
        let custody = MetadataCustody {
            raw,
            _preparation: None,
        };
        let value: Value = Box::new(Registry::<K>::new());
        let node = Box::new(Node {
            kind: TypeId::of::<K>(),
            value,
            next: Directory::default(),
        });
        Self {
            node,
            _custody: custody,
        }
    }
    pub(super) fn prepare_copy<K: Ord + Send + 'static>(
        host: &eredu_core::HostPreparationAuthority,
    ) -> Self {
        let custody = MetadataCustody {
            raw: None,
            _preparation: Some(host.clone()),
        };
        let value: Value = Box::new(Registry::<K>::new());
        let node = Box::new(Node {
            kind: TypeId::of::<K>(),
            value,
            next: Directory::default(),
        });
        Self {
            node,
            _custody: custody,
        }
    }
    pub(super) fn cold_control_bytes<K: Ord + Send + 'static>() -> Result<u64, WorkingMemoryError> {
        u64::try_from(
            size_of::<Node>()
                .checked_add(size_of::<Registry<K>>())
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .map_err(|_| WorkingMemoryError::Overflow)
    }
    pub(super) fn registry<K: Ord + Send + 'static>(&self) -> &Registry<K> {
        self.node
            .value
            .downcast_ref()
            .expect("prepared typed namespace")
    }
    pub(super) fn requested_control_bytes<K: Ord + Send + 'static>()
    -> Result<u64, WorkingMemoryError> {
        // Actual Box payloads, owning wrapper and simultaneously representable
        // construction/retirement transports. Box adds no managed heap header
        // on the already-qualified Global producer; arbitrary K is still absent.
        let parts = [
            size_of::<Node>(),
            size_of::<Registry<K>>(),
            size_of::<Self>(),
            size_of::<Value>(),
            size_of::<Box<Node>>(),
            size_of::<MetadataCustody>(),
            size_of::<Option<Self>>(),
            size_of::<&mut Directory>(),
            size_of::<TypeId>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of::<[usize; 9]>(), usize::checked_add)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
}
impl Directory {
    pub(super) fn get(&self, kind: &TypeId) -> Option<&Value> {
        let mut entry = self.head.as_ref();
        while let Some(current) = entry {
            if current.node.kind == *kind {
                return Some(&current.node.value);
            }
            entry = current.node.next.head.as_ref();
        }
        None
    }
    pub(super) fn get_mut(&mut self, kind: &TypeId) -> Option<&mut Value> {
        let mut entry = self.head.as_mut();
        while let Some(current) = entry {
            if current.node.kind == *kind {
                return Some(&mut current.node.value);
            }
            entry = current.node.next.head.as_mut();
        }
        None
    }
    pub(in crate::working_memory) fn len(&self) -> usize {
        let mut result = 0;
        let mut entry = self.head.as_ref();
        while let Some(current) = entry {
            result += 1;
            entry = current.node.next.head.as_ref();
        }
        result
    }
    pub(in crate::working_memory) fn is_empty(&self) -> bool {
        self.head.is_none()
    }
    pub(super) fn install(&mut self, mut prepared: PreparedNamespace) {
        // Callers already validated absence under this same Usage loan. This
        // uses only scalar identity and pointer moves; no allocation/callback.
        assert!(self.get(&prepared.node.kind).is_none());
        assert!(prepared.node.next.is_empty());
        prepared.node.next.head = self.head.take();
        self.head = Some(prepared);
    }
    pub(super) fn remove(&mut self, kind: &TypeId) -> Option<PreparedNamespace> {
        let mut cursor = &mut self.head;
        loop {
            if cursor.as_ref()?.node.kind == *kind {
                let mut removed = cursor.take().expect("located namespace");
                *cursor = removed.node.next.head.take();
                return Some(removed);
            }
            cursor = &mut cursor.as_mut().expect("live cursor").node.next.head;
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        // Detach the tail before any erased Registry destructor. The tail is
        // itself a Directory, so unwind cannot turn it into recursive Box drop.
        while let Some(mut current) = self.head.take() {
            self.head = current.node.next.head.take();
            drop(current);
        }
    }
}
