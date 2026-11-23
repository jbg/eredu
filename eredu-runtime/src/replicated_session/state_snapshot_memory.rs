//! Portable state snapshot estimates from declared geometry and native storage costs.
use eredu_core::{
    cache::{StateTensorDimension, StateTensorDtype},
    execution_control::SnapshotEstimate,
};

use crate::StateLayout;

/// Native storage costs for retaining an isolated state copy.
pub struct StateSnapshotMemoryCosts {
    /// Per-layer metadata outside the logical layout.
    pub layer_metadata: u64,
    /// Native metadata for one declared tensor slot, even when empty.
    pub slot_metadata: u64,
    /// Per-array metadata independent of rank.
    pub array_metadata: u64,
    /// Native descriptor storage per dimension.
    pub dimension_metadata: u64,
    /// Maximum simultaneous native buffer count per copied tensor.
    pub copy_buffers: u64,
    /// Maximum byte width of the selected native floating representations.
    pub floating_width: u64,
}

impl StateSnapshotMemoryCosts {
    /// Combines native tensor costs with the portable layout accounting allowance.
    pub fn for_tensors(tensors: eredu_core::cache::mechanisms::CacheSnapshotTensorCosts) -> Self {
        Self {
            layer_metadata: 1024,
            slot_metadata: 256,
            array_metadata: tensors.tensor_metadata,
            dimension_metadata: tensors.dimension_metadata,
            copy_buffers: tensors.copy_buffers,
            floating_width: tensors.floating_width,
        }
    }

    /// Native bytes to retain and copy one tensor with the supplied shape rank.
    pub fn tensor_bytes(&self, payload: u64, dimensions: usize) -> Option<u64> {
        eredu_core::cache::mechanisms::CacheSnapshotTensorCosts {
            tensor_metadata: self.array_metadata,
            dimension_metadata: self.dimension_metadata,
            copy_buffers: self.copy_buffers,
            floating_width: self.floating_width,
        }
        .bytes(payload, dimensions)
    }

    /// Accounts for all declared slots and the exact retained native backing.
    pub fn estimate(
        &self,
        layout: &StateLayout,
        state_owner_bytes: u64,
        tensor_bytes: u64,
        auxiliary_bytes: u64,
    ) -> Option<SnapshotEstimate> {
        let mut bytes = state_owner_bytes
            .checked_add(layout.logical_metadata_bytes()?)?
            .checked_add(
                u64::try_from(layout.len())
                    .ok()?
                    .checked_mul(self.layer_metadata)?,
            )?
            .checked_add(tensor_bytes)?
            .checked_add(auxiliary_bytes)?;
        for layer in 0..layout.len() {
            let slots = u64::try_from(layout.components(layer)?.len()).ok()?;
            bytes = bytes.checked_add(slots.checked_mul(self.slot_metadata)?)?;
        }
        Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    /// Reserves the full future payload of the single-sequence layout, including
    /// currently empty fixed state and the largest possible remainder.
    pub fn growth(&self, layout: &StateLayout, prefix: u64, auxiliary_bytes: u64) -> Option<u64> {
        use StateTensorDimension as Dim;
        let mut bytes = auxiliary_bytes;
        for layer in 0..layout.len() {
            for component in layout.components(layer)? {
                let elements = component.shape().iter().try_fold(1u64, |elements, dim| {
                    let count = match dim {
                        Dim::Batch | Dim::Scalar => 1,
                        Dim::Fixed(n) => u64::from(n.get()),
                        Dim::PrefixTokens => prefix,
                        Dim::PrefixTokensDiv(n) => prefix / u64::from(n.get()),
                        Dim::PrefixTokensRem(n) => prefix.min(u64::from(n.get()) - 1),
                    };
                    elements.checked_mul(count)
                })?;
                let width = match component.dtype() {
                    StateTensorDtype::Floating => self.floating_width,
                    StateTensorDtype::Float32
                    | StateTensorDtype::Int32
                    | StateTensorDtype::Uint32 => 4,
                };
                bytes = bytes.checked_add(
                    self.tensor_bytes(elements.checked_mul(width)?, component.shape().len())?,
                )?;
            }
        }
        Some(bytes)
    }
}
