//! Final lazy lease from the same immutable source/selection plan. No I/O.
use super::*;
use crate::store::{MetadataCloneLayout, SelectionCloneLayout};
use std::{alloc::Layout, mem::size_of};

impl GgufConversionPlan {
    pub(crate) fn matches_store(&self, store: &GgufWeightStore) -> bool {
        self.source.same(&store.inner)
    }
    fn acquisition_entry(&self, request: &TensorReadRequest) -> Option<&CatalogEntry> {
        let entry = self.source.catalog.get(&request.key)?;
        (entry.checkpoint == self.identity.checkpoint
            && entry.physical_name == self.identity.physical_name
            && entry.original_name == self.output_name)
            .then_some(entry)
    }
    pub(crate) fn acquisition_diagnostic_key_bytes(
        &self,
        request: &TensorReadRequest,
    ) -> Option<usize> {
        Some(self.acquisition_entry(request)?.metadata.name.len())
    }
    pub(crate) fn acquisition_storage(&self, request: &TensorReadRequest) -> Option<usize> {
        let entry = self.acquisition_entry(request)?;
        let physical = super::super::lease_controls::physical_clone_layout(entry, &self.identity)?;
        [
            Layout::new::<GgufLease>().size(),
            MetadataCloneLayout::of(&entry.metadata)?.payload_bytes()?,
            physical.payload_bytes()?,
            SelectionCloneLayout::of(&request.selection)?
                .elements
                .map_or(0, |l| l.size()),
            Layout::array::<usize>(self.output_shape.len()).ok()?.size(),
            // PreparedGgufLease's independent matching key.
            request.key.len(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    pub(crate) fn acquisition_controls() -> Option<usize> {
        [
            size_of::<GgufLease>(),
            size_of::<Option<GgufLease>>(),
            size_of::<CatalogEntry>(),
            size_of::<GgufLeaseIdentity>(),
            size_of::<&CatalogEntry>(),
            size_of::<BoundedReadProof>(),
            size_of::<Vec<usize>>(),
            size_of::<String>(),
            size_of::<Vec<u64>>(),
            size_of::<Option<GgufPhysicalSelection>>(),
            size_of::<TensorSelection>(),
            size_of::<TensorMetadata>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    pub(crate) fn prepare_acquisition(
        &self,
        store: &GgufWeightStore,
        request: &TensorReadRequest,
    ) -> Option<GgufLease> {
        if !self.source.same(&store.inner) {
            return None;
        }
        let entry = self.acquisition_entry(request)?;
        // The selected wrapper retains the exact logical request. Shape/read
        // planning already ran on this immutable source. Reuse its result;
        // ordinary acquisition remains on the shared original validator.
        Some(GgufLease {
            store: self.source.clone(),
            entry: entry.clone(),
            selection: request.selection.clone(),
            output_shape: self.output_shape.clone(),
            proof: BoundedReadProof {
                physically_bounded: self.read.physically_bounded,
                offset_bytes: self.read.offset,
                length_bytes: self.read.bytes,
                physical_reads: 1,
                physical_read_bytes: self.read.bytes,
            },
            identity: self.identity.clone(),
            selection_is_materialized: self.selection_is_materialized,
        })
    }
}
