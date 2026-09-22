//! Closed ownership of an actual live layout and all its retained host storage.

use super::{StateLayout, StateSegmentId, StateSegmentSpec};
use crate::host_metadata::{HostMetadataIdentity, MetadataCustody};
use eredu_core::{
    cache::{LayerCachePolicy, StateComponentPolicy, StateTensorDimension, StateTensorPolicy},
    SharedStorageAccountingId, SharedStorageAttachmentError,
};
use std::{fmt, mem::size_of, sync::Arc};

mod workspace;

struct LayoutInner {
    // Field order retires every nested layout allocation before custody.
    payload: StateLayout,
    #[cfg(test)]
    payload_retired: Option<tests::PayloadRetired>,
    custody: MetadataCustody,
}

/// Immutable shared ownership of a complete live state-layout payload.
///
/// Existing aliases observe later per-domain accounting attachments. A borrowed
/// layout cannot be mutated or moved out; cloning the borrowed `StateLayout`
/// creates separate caller-owned storage without copying this owner's custody.
#[derive(Clone)]
pub struct SharedStateLayout(Arc<LayoutInner>);

impl SharedStateLayout {
    /// Transfers an existing layout without cloning or shrinking its storage.
    pub fn new(layout: StateLayout) -> Self {
        Self::from_payload(layout, MetadataCustody::new())
    }

    fn from_payload(layout: StateLayout, custody: MetadataCustody) -> Self {
        Self(Arc::new(LayoutInner {
            payload: layout,
            #[cfg(test)]
            payload_retired: None,
            custody,
        }))
    }

    /// Borrows the exact immutable layout.
    pub fn layout(&self) -> &StateLayout {
        &self.0.payload
    }

    /// Payload-free identity of this shared owner, independent of its contents.
    pub fn identity(&self) -> &HostMetadataIdentity {
        self.0.custody.identity()
    }

    /// Exact managed host payload and retained allocation capacities.
    ///
    /// Includes the layout value, boxed layer schedule, every nested vector
    /// capacity and segment-name capacity. Arc/custody/allocator bookkeeping is
    /// excluded. This performs no allocation; overflow returns `None`.
    pub fn capacity_bytes(&self) -> Option<u64> {
        retained_bytes(self.layout())
    }

    /// Whether both handles share this exact payload owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Attaches one accounting handle per domain, including to earlier aliases.
    ///
    /// An existing domain returns false without calling the provider. Metadata
    /// nodes are allocated after provider acquisition; rejection preserves previously
    /// attached handles. The provider runs under custody and may perform only
    /// closed accounting operations: no owner reentry, other source locks,
    /// native work or user callbacks. Its handle must not retain this payload,
    /// directly or indirectly. Register only payload-free identity/domain keys.
    /// Provider handles retire outside the custody lock, after the layout.
    pub(crate) fn original_attachment_ready(
        &self,
        domain: &SharedStorageAccountingId,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        self.0.custody.original_attachment_ready(domain)
    }

    pub(crate) fn try_attach_owned_prepared<T: eredu_core::SharedStorageRetirement, E>(
        &self,
        owner: &SharedStorageAccountingId,
        acquire: impl FnOnce(
            eredu_core::SharedStorageAttachmentLayout,
        ) -> Result<eredu_core::SharedStorageOwner<T>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.custody.try_attach_owned_prepared(owner, acquire)
    }

    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageAccountingId,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.custody.try_attach(domain, acquire)
    }
}

impl AsRef<StateLayout> for SharedStateLayout {
    fn as_ref(&self) -> &StateLayout {
        self.layout()
    }
}

impl PartialEq for SharedStateLayout {
    fn eq(&self, other: &Self) -> bool {
        self.layout() == other.layout()
    }
}
impl Eq for SharedStateLayout {}

impl fmt::Debug for SharedStateLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SharedStateLayout")
            .field(self.layout())
            .finish()
    }
}

fn bytes<T>(count: usize) -> Option<u64> {
    u64::try_from(size_of::<T>().checked_mul(count)?).ok()
}

fn retained_bytes(layout: &StateLayout) -> Option<u64> {
    let StateLayout {
        layers,
        components,
        segments,
    } = layout;
    // LayerSchedule owns a boxed slice: its allocation extent is exactly len.
    let mut total = bytes::<StateLayout>(1)?
        .checked_add(bytes::<LayerCachePolicy>(layers.len())?)?
        .checked_add(bytes::<Vec<StateComponentPolicy>>(components.capacity())?)?
        .checked_add(bytes::<StateSegmentSpec>(segments.capacity())?)?;
    for layer in layers.iter() {
        let tensors = match layer {
            LayerCachePolicy::NoState
            | LayerCachePolicy::KeyValue {
                attention: _,
                num_key_value_heads: _,
                head_dim: _,
            }
            | LayerCachePolicy::KeyOnly {
                attention: _,
                num_key_heads: _,
                head_dim: _,
            }
            | LayerCachePolicy::CompressedLatentRotary {
                attention: _,
                latent_dim: _,
                rotary_dim: _,
            } => continue,
            LayerCachePolicy::FixedState { tensors }
            | LayerCachePolicy::KeyValueWithFixedState {
                attention: _,
                num_key_value_heads: _,
                head_dim: _,
                tensors,
            }
            | LayerCachePolicy::KeyOnlyWithFixedState {
                attention: _,
                num_key_heads: _,
                head_dim: _,
                tensors,
            } => tensors,
        };
        total = total.checked_add(bytes::<StateTensorPolicy>(tensors.capacity())?)?;
        for tensor in tensors {
            let StateTensorPolicy {
                role: _,
                shape,
                dtype: _,
                residency: _,
                presence: _,
            } = tensor;
            total = total.checked_add(bytes::<StateTensorDimension>(shape.capacity())?)?;
        }
    }
    for group in components {
        total = total.checked_add(bytes::<StateComponentPolicy>(group.capacity())?)?;
        for component in group {
            total =
                total.checked_add(bytes::<StateTensorDimension>(component.shape_capacity())?)?;
        }
    }
    for segment in segments {
        let StateSegmentSpec {
            id: StateSegmentId(name),
            layers: _,
            lifetime: _,
            processed_token_offset: _,
        } = segment;
        total = total.checked_add(u64::try_from(name.capacity()).ok()?)?;
    }
    Some(total)
}

#[cfg(test)]
mod tests;
