//! Fixed borrowed-owner retention. A failed prefix stays in its caller's capsule.
use crate::backend::runtime::residency::storage::RetainedArray;
use super::capacity::OpeningSlots;
use super::*;
use crate::backend::runtime::residency::storage::{
    RetainedStorageInspectionError, RetainedStorageRef,
};
use eredu_checkpoint::store::{SourceStorageOwner, StoreError};
use eredu_runtime::{HostSlotMetadata, SharedStateLayout, StateError};
use safemlx::{
    AllocationInfo, ArrayAllocationInfo, ArrayMetadataError, HostTransferMetadataError,
    ImmutableHostTransferBuffer,
};
use std::{collections::TryReserveError, mem::size_of, sync::Arc};

#[cfg(test)]
thread_local! {
    static FAIL_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
}

#[derive(Debug, thiserror::Error)]
pub(super) enum OpeningError {
    #[error("native opening owner identity differs")]
    Identity,
    #[error("native opening owner count or physical allocation is unknown")]
    Unknown,
    #[error("native opening cache catalog growth/acquisition is not bound")]
    CacheCatalog,
    #[error("native opening owner capacity overflow")]
    Overflow,
    #[error("native opening owner slot capacity exceeded")]
    Capacity,
    #[error("native opening inventory is incomplete")]
    Incomplete,
    #[error("native opening owner capsule has already been used")]
    Used,
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    #[error(transparent)]
    Array(#[from] ArrayMetadataError),
    #[error(transparent)]
    Host(#[from] HostTransferMetadataError),
    #[error(transparent)]
    Inspection(#[from] RetainedStorageInspectionError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error(transparent)]
    StateOwner(Error),
    #[error(transparent)]
    Funding(#[from] eredu_runtime::working_memory::WorkingMemoryError),
}
struct Slot<T, F> {
    owner: T,
    fact: Option<F>,
}
struct Slots<T, F> {
    values: Vec<Slot<T, F>>,
    limit: usize,
}
impl<T, F> Slots<T, F> {
    fn new(limit: usize) -> Self {
        Self {
            values: Vec::new(),
            limit,
        }
    }
    fn reserve(&mut self) -> Result<(), OpeningError> {
        #[cfg(test)]
        if self.limit != 0 {
            if let Some(remaining) = FAIL_AFTER.get() {
                if remaining == 0 {
                    FAIL_AFTER.set(None);
                    return Err(OpeningError::Allocation(
                        Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err(),
                    ));
                }
                FAIL_AFTER.set(Some(remaining - 1));
            }
        }
        self.values.try_reserve_exact(self.limit)?;
        // No reallocating/shrinking conversion is used. This implementation
        // retains the actual Vec capacity; unexpected excess is rejected before
        // visiting any source. Allocator bookkeeping is not payload capacity.
        if self.values.capacity() != self.limit {
            return Err(OpeningError::Capacity);
        }
        Ok(())
    }
    fn available(&self) -> Result<(), OpeningError> {
        if self.values.len() >= self.limit || self.values.capacity() < self.limit {
            Err(OpeningError::Capacity)
        } else {
            Ok(())
        }
    }
    fn install(&mut self, owner: T) {
        // All callers check the empty, already allocated slot BEFORE retaining.
        debug_assert!(self.values.len() < self.limit);
        self.values.push(Slot { owner, fact: None });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Empty,
    Retaining,
    Inspecting,
    Complete,
}
pub(super) struct FixedOpeningOwners {
    arrays: Slots<RetainedArray, ArrayAllocationInfo>,
    hosts: Slots<crate::backend::runtime::residency::manager::RetainedHostBuffer, AllocationInfo>,
    bytes: Slots<Arc<[u8]>, u64>,
    sources: Slots<SourceStorageOwner, u64>,
    layouts: Slots<SharedStateLayout, u64>,
    tables: Slots<HostSlotMetadata, u64>,
    status: Status,
}
/// Borrowed complete facts only. No owning export, publication or grant.
pub(super) enum OpeningEntry<'a> {
    Array(&'a Array, ArrayAllocationInfo),
    Host(
        &'a crate::backend::runtime::residency::manager::RetainedHostBuffer,
        AllocationInfo,
    ),
    Bytes(&'a Arc<[u8]>, u64),
    Source(eredu_checkpoint::store::SourceStorageRef<'a>),
    Layout(&'a SharedStateLayout, u64),
    Table(&'a HostSlotMetadata, u64),
}
impl FixedOpeningOwners {
    /// The linked clone constructor creates one C++ array handle per retained
    /// Array. It shares ArrayDesc/backing; aliases still create distinct handles.
    /// Reserve the entire declared slot ceiling, including initially absent
    /// state fields. Rust slots and native handles have the same capsule lifetime.
    pub(super) fn storage_bytes(n: OpeningSlots) -> Result<u64, OpeningError> {
        let handles = n
            .arrays
            .checked_mul(Array::inspection_clone_handle_bytes())
            .ok_or(OpeningError::Overflow)?;
        Self::buffer_bytes(n)?
            .checked_add(u64::try_from(handles).map_err(|_| OpeningError::Overflow)?)
            .ok_or(OpeningError::Overflow)
    }
    /// Borrow one original category-order owner, after complete fact inspection.
    /// No owner is moved and no physical identity or allocation is synthesized.
    pub(super) fn entry_at(&self, mut index: usize) -> Result<OpeningEntry<'_>, OpeningError> {
        if self.status != Status::Complete {
            return Err(OpeningError::Incomplete);
        }
        if let Some(s) = self.arrays.values.get(index) {
            return Ok(OpeningEntry::Array(
                &s.owner,
                s.fact.expect("complete facts"),
            ));
        }
        index = index
            .checked_sub(self.arrays.values.len())
            .ok_or(OpeningError::Capacity)?;
        if let Some(s) = self.hosts.values.get(index) {
            return Ok(OpeningEntry::Host(
                &s.owner,
                s.fact.expect("complete facts"),
            ));
        }
        index = index
            .checked_sub(self.hosts.values.len())
            .ok_or(OpeningError::Capacity)?;
        if let Some(s) = self.bytes.values.get(index) {
            return Ok(OpeningEntry::Bytes(
                &s.owner,
                s.fact.expect("complete facts"),
            ));
        }
        index = index
            .checked_sub(self.bytes.values.len())
            .ok_or(OpeningError::Capacity)?;
        if let Some(s) = self.sources.values.get(index) {
            return Ok(OpeningEntry::Source(s.owner.as_storage_ref()));
        }
        index = index
            .checked_sub(self.sources.values.len())
            .ok_or(OpeningError::Capacity)?;
        if let Some(s) = self.layouts.values.get(index) {
            return Ok(OpeningEntry::Layout(
                &s.owner,
                s.fact.expect("complete facts"),
            ));
        }
        index = index
            .checked_sub(self.layouts.values.len())
            .ok_or(OpeningError::Capacity)?;
        if let Some(s) = self.tables.values.get(index) {
            return Ok(OpeningEntry::Table(
                &s.owner,
                s.fact.expect("complete facts"),
            ));
        }
        index = index
            .checked_sub(self.tables.values.len())
            .ok_or(OpeningError::Capacity)?;
        let _ = index;
        Err(OpeningError::Capacity)
    }
    pub(super) fn visit<E: From<OpeningError>>(
        &self,
        visitor: &mut dyn FnMut(OpeningEntry<'_>) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.status != Status::Complete {
            return Err(OpeningError::Incomplete.into());
        }
        for s in &self.arrays.values {
            visitor(OpeningEntry::Array(
                &s.owner,
                s.fact.expect("complete facts"),
            ))?;
        }
        for s in &self.hosts.values {
            visitor(OpeningEntry::Host(
                &s.owner,
                s.fact.expect("complete facts"),
            ))?;
        }
        for s in &self.bytes.values {
            visitor(OpeningEntry::Bytes(
                &s.owner,
                s.fact.expect("complete facts"),
            ))?;
        }
        for s in &self.sources.values {
            visitor(OpeningEntry::Source(s.owner.as_storage_ref()))?;
        }
        for s in &self.layouts.values {
            visitor(OpeningEntry::Layout(
                &s.owner,
                s.fact.expect("complete facts"),
            ))?;
        }
        for s in &self.tables.values {
            visitor(OpeningEntry::Table(
                &s.owner,
                s.fact.expect("complete facts"),
            ))?;
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn fail_after_allocations(count: usize) {
        FAIL_AFTER.set(Some(count));
    }
    #[cfg(test)]
    pub(super) fn retained_counts(&self) -> OpeningSlots {
        OpeningSlots {
            arrays: self.arrays.values.len(),
            hosts: self.hosts.values.len(),
            bytes: self.bytes.values.len(),
            sources: self.sources.values.len(),
            layouts: self.layouts.values.len(),
            tables: self.tables.values.len(),
        }
    }
    pub(super) fn buffer_bytes(n: OpeningSlots) -> Result<u64, OpeningError> {
        let terms = [
            (n.arrays, size_of::<Slot<RetainedArray, ArrayAllocationInfo>>()),
            (
                n.hosts,
                size_of::<
                    Slot<
                        crate::backend::runtime::residency::manager::RetainedHostBuffer,
                        AllocationInfo,
                    >,
                >(),
            ),
            (n.bytes, size_of::<Slot<Arc<[u8]>, u64>>()),
            (n.sources, size_of::<Slot<SourceStorageOwner, u64>>()),
            (n.layouts, size_of::<Slot<SharedStateLayout, u64>>()),
            (n.tables, size_of::<Slot<HostSlotMetadata, u64>>()),
        ];
        let total = terms.into_iter().try_fold(0usize, |sum, (n, size)| {
            sum.checked_add(n.checked_mul(size).ok_or(OpeningError::Overflow)?)
                .ok_or(OpeningError::Overflow)
        })?;
        u64::try_from(total.checked_add(RetainedArray::control_bytes().ok_or(OpeningError::Overflow)?).ok_or(OpeningError::Overflow)?).map_err(|_| OpeningError::Overflow)
    }
    pub(super) fn empty(n: OpeningSlots) -> Self {
        Self {
            arrays: Slots::new(n.arrays),
            hosts: Slots::new(n.hosts),
            bytes: Slots::new(n.bytes),
            sources: Slots::new(n.sources),
            layouts: Slots::new(n.layouts),
            tables: Slots::new(n.tables),
            status: Status::Empty,
        }
    }
    pub(super) fn reserve(&mut self) -> Result<(), OpeningError> {
        self.arrays.reserve()?;
        self.hosts.reserve()?;
        self.bytes.reserve()?;
        self.sources.reserve()?;
        self.layouts.reserve()?;
        self.tables.reserve()
    }
    pub(super) fn begin(&mut self) -> Result<(), OpeningError> {
        if self.status != Status::Empty {
            return Err(OpeningError::Used);
        }
        // Any error or unwind keeps this terminal nonempty status; no reset,
        // fallback allocation or owner destruction happens during collection.
        self.status = Status::Retaining;
        Ok(())
    }
    pub(super) fn retain_array(&mut self, value: &Array) -> Result<(), OpeningError> {
        if self.status != Status::Retaining {
            return Err(OpeningError::Used);
        }
        self.arrays.available()?;
        let retained = value.try_clone_for_inspection()?;
        self.arrays.install(retained.into());
        Ok(())
    }
    pub(super) fn retain(&mut self, value: RetainedStorageRef<'_>) -> Result<(), OpeningError> {
        if self.status != Status::Retaining {
            return Err(OpeningError::Used);
        }
        match value {
            RetainedStorageRef::Array(value) => self.retain_array(value),
            RetainedStorageRef::CanonicalArray(cell) => {
                self.arrays.available()?;
                self.arrays.install(RetainedArray::from_canonical(cell));
                Ok(())
            },
            RetainedStorageRef::Host(value) => {
                self.hosts.available()?;
                self.hosts.install(Arc::clone(value).into());
                Ok(())
            }
            RetainedStorageRef::RetainedHost(value) => {
                self.hosts.available()?;
                self.hosts.install(value.clone());
                Ok(())
            }
            RetainedStorageRef::Bytes(value) => {
                self.bytes.available()?;
                self.bytes.install(Arc::clone(value));
                Ok(())
            }
            RetainedStorageRef::Source(value) => {
                self.sources.available()?;
                self.sources.install(value.retain());
                Ok(())
            }
        }
    }
    pub(super) fn retain_layout(&mut self, value: &SharedStateLayout) -> Result<(), OpeningError> {
        if self.status != Status::Retaining {
            return Err(OpeningError::Used);
        }
        self.layouts.available()?;
        self.layouts.install(value.clone());
        Ok(())
    }
    pub(super) fn retain_table(&mut self, value: &HostSlotMetadata) -> Result<(), OpeningError> {
        if self.status != Status::Retaining {
            return Err(OpeningError::Used);
        }
        self.tables.available()?;
        self.tables.install(value.clone());
        Ok(())
    }
    pub(super) fn retain_checkpoint(
        &mut self,
        value: &dyn CheckpointSource,
    ) -> Result<(), OpeningError> {
        let mut first = None;
        let complete = value.visit_source_storage(&mut |v| {
            if first.is_none() {
                first = self.retain(RetainedStorageRef::Source(v)).err();
            }
        });
        // Callback and subsequent source errors are disposed outside all loans.
        if let Some(error) = first {
            return Err(error);
        }
        if !complete? {
            return Err(OpeningError::Incomplete);
        }
        Ok(())
    }
    pub(super) fn inspect_facts(&mut self) -> Result<(), OpeningError> {
        if self.status != Status::Retaining {
            return Err(OpeningError::Used);
        }
        self.status = Status::Inspecting;
        for slot in &mut self.arrays.values {
            slot.fact = Some(
                slot.owner
                    .try_allocation_info()?
                    .ok_or(OpeningError::Unknown)?,
            );
        }
        for slot in &mut self.hosts.values {
            slot.fact = Some(slot.owner.try_allocation_info()?);
        }
        for slot in &mut self.bytes.values {
            slot.fact = Some(u64::try_from(slot.owner.len()).map_err(|_| OpeningError::Overflow)?);
        }
        for slot in &mut self.sources.values {
            slot.fact = Some(slot.owner.bytes());
        }
        for slot in &mut self.layouts.values {
            slot.fact = Some(slot.owner.capacity_bytes().ok_or(OpeningError::Overflow)?);
        }
        for slot in &mut self.tables.values {
            slot.fact = Some(slot.owner.capacity_bytes().ok_or(OpeningError::Overflow)?);
        }
        self.status = Status::Complete;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
