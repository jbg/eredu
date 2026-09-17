//! Physical payload ownership, independent of logical weight-residency tiers.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::store::SourceStorage;
use safemlx::{Array, ImmutableHostTransferBuffer};

use super::manager::{ResidencyError, RetainedHostBuffer};

pub(crate) mod filled_host;

mod borrowed;
pub(crate) use borrowed::{RetainedStorageInspectionError, RetainedStorageRef};
pub(crate) use borrowed::{RetainedStorageVisitFailure, visit_checkpoint_storage};

mod admission;
pub(crate) mod native_storage;
mod original;
pub use admission::RetainedStorageReservation;
pub(crate) use admission::StorageIdentity;
pub(crate) use admission::{
    CopyPublicationLayout, PendingCopyPublication, PendingNativePublication,
    RetainedStoragePublication, retain_copy_publication_failure,
};

/// A retained inventory of native arrays, immutable transfer buffers and exact
/// checkpoint sources. Aliases count once while their physical owners stay alive.
///
/// This is evidence for admission, not a reservation or permission to allocate.
/// It includes explicitly declared immutable layout/input owners and fixed host
/// slot extents, plus explicitly included shared admitted capture plans. Slot
/// tokens retain accounting custody, not table contents. Other bookkeeping,
/// allocator caches and process memory remain excluded. Callers must
/// also include loaded module parameters, decoder state and other retained resources;
/// a residency manager alone does not own all of them. Future materialization
/// requires a separate workspace bound and reservation.
#[derive(Default)]
pub struct RetainedStorage {
    arrays: BTreeMap<safemlx::AllocationIdentity, (u64, Array)>,
    group_buffers: BTreeMap<
        safemlx::distributed::GroupBufferIdentity,
        (u64, safemlx::distributed::RetainedGroupBuffer),
    >,
    hosts: BTreeMap<safemlx::AllocationIdentity, (u64, RetainedHostBuffer)>,
    byte_buffers: BTreeMap<usize, Arc<[u8]>>,
    metadata:
        BTreeMap<eredu_runtime::HostMetadataIdentity, (u64, eredu_runtime::SharedHostMetadata)>,
    slot_metadata:
        BTreeMap<eredu_runtime::HostMetadataIdentity, (u64, eredu_runtime::HostSlotMetadata)>,
    capture_plans:
        BTreeMap<eredu_core::SharedStorageIdentity, (u64, eredu_core::capture::SharedCapturePlan)>,
    sources: SourceStorage,
    unknown_arrays: Vec<Array>,
    unknown_metadata: Vec<eredu_runtime::SharedHostMetadata>,
    unknown_capture_plans: Vec<eredu_core::capture::SharedCapturePlan>,
    unknown_slot_metadata: Vec<eredu_runtime::HostSlotMetadata>,
    incomplete: bool,
    original: Option<original::Inventory>,
    census: Option<original::Census>,
}

impl std::fmt::Debug for RetainedStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedStorage")
            .field("native_allocations", &self.arrays.len())
            .field("communicator_buffers", &self.group_buffers.len())
            .field("host_allocations", &self.hosts.len())
            .field("byte_buffers", &self.byte_buffers.len())
            .field("metadata_owners", &self.metadata.len())
            .field("host_slot_tokens", &self.slot_metadata.len())
            .field("capture_plan_owners", &self.capture_plans.len())
            .field("source_storage", &self.sources)
            .field(
                "unknown_array_shapes",
                &self
                    .unknown_arrays
                    .iter()
                    .map(Array::shape)
                    .collect::<Vec<_>>(),
            )
            .field("incomplete", &self.incomplete)
            .finish()
    }
}

/// Fixed inspection transport; no source formatting or backend erasure occurs.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SnapshotStorageInspectionError {
    #[error(transparent)]
    Boundary(#[from] eredu_runtime::replicated_session::RuntimeInspectionBoundary),
    #[error(transparent)]
    Residency(#[from] ResidencyError),
    #[error("snapshot source census is unavailable")]
    Unavailable,
}

impl RetainedStorage {
    pub(crate) fn is_snapshot_census(&self) -> bool {
        self.census.is_some()
    }

    /// A census has no owning rows. This exact count preserves incomplete facts.
    pub(crate) fn snapshot_census_rows(&self) -> Option<usize> {
        self.census
            .as_ref()
            .filter(|_| !self.incomplete)
            .map(|c| c.rows)
    }

    /// Borrows the exact source owners already captured by this inventory.
    /// Does not inspect native arrays or upgrade an unknown total bound.
    #[cfg(test)]
    pub(crate) fn source_storage(&self) -> &SourceStorage {
        &self.sources
    }

    /// Moves only the native descriptors actually captured by this inventory.
    /// This neither queries/evaluates them nor upgrades unknown/incomplete
    /// coverage. Explicit loading work can settle these roots under its existing
    /// authority, then obtain a fresh cold inventory for physical publication.
    /// No host/source buffer, unloaded unit or future allocation is materialized.
    /// Original/census inventories refuse and return their complete owner;
    /// only ordinary loading inventories can export bare Array handles.
    pub(crate) fn into_retained_arrays(
        self,
    ) -> Result<impl Iterator<Item = Array>, (ResidencyError, Self)> {
        // A prepared clone shell belongs to this inventory's raw custody. Bare
        // extraction would let that shell outlive its charge. Keep the complete
        // input on refusal; census is observational and owns no array rows.
        if self.original.is_some() || self.census.is_some() {
            return Err((
                original::failure(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ),
                self,
            ));
        }
        Ok(self
            .arrays
            .into_values()
            .map(|(_, array)| array)
            .chain(self.unknown_arrays))
    }

    /// Actual metadata roots retained by this inventory, without reconstruction.
    #[cfg(test)]
    pub(crate) fn metadata_sources(
        &self,
    ) -> impl Iterator<Item = &eredu_runtime::SharedHostMetadata> {
        self.metadata_entries().map(|(_, (_, owner))| owner)
    }

    /// Actual fixed-table tokens retained by this inventory; no table payload
    /// or reconstructed capacity is exposed through this test seam.
    #[cfg(test)]
    pub(crate) fn slot_metadata_sources(
        &self,
    ) -> impl Iterator<Item = &eredu_runtime::HostSlotMetadata> {
        self.slot_entries().map(|(_, (_, token))| token)
    }

    /// Already captured native allocation facts for identity-based conformance
    /// assertions. Does not query arrays or upgrade incomplete coverage.
    #[cfg(test)]
    pub(crate) fn array_allocation_facts(&self) -> BTreeMap<safemlx::AllocationIdentity, u64> {
        self.array_entries()
            .map(|(identity, (bytes, _))| (*identity, *bytes))
            .collect()
    }

    /// Retained roots whose backing was unknown when the inventory was taken.
    /// Test code can establish completion separately from cold inspection.
    #[cfg(test)]
    pub(crate) fn unknown_arrays(&self) -> &[Array] {
        &self.unknown_arrays
    }

    /// Collects native backing from a cold traversal of all retained module
    /// numerical values, including nonparameter operator helpers.
    /// Unavailable traversal stays unknown. This contributes module values only;
    /// source stores, residency buffers, banks and state are separate owners.
    pub fn collect_retained_values<E: From<ResidencyError>>(
        inspect: impl FnOnce(&mut dyn FnMut(&crate::MlxTensor)) -> Result<bool, E>,
    ) -> Result<Self, E> {
        let mut storage = Self::default();
        storage.include_retained_values(inspect)?;
        Ok(storage)
    }

    pub(crate) fn include_retained_values<E: From<ResidencyError>>(
        &mut self,
        inspect: impl FnOnce(&mut dyn FnMut(&crate::MlxTensor)) -> Result<bool, E>,
    ) -> Result<(), E> {
        let mut failure = None;
        let complete = inspect(&mut |value| {
            if failure.is_none() {
                failure = self.include_array(value.as_array()).err();
            }
        })?;
        if let Some(error) = failure {
            return Err(error.into());
        }
        if !complete {
            self.mark_incomplete();
        }
        Ok(())
    }

    /// Adds completed certified backing without evaluation, polling or runtime
    /// housekeeping. Foreign runtime contention returns a typed unavailable
    /// error without changing the inventory. Unknown backing remains unknown.
    /// The caller keeps the settled source unchanged while inspecting/retaining.
    #[track_caller]
    pub fn include_array(&mut self, array: &Array) -> Result<(), ResidencyError> {
        if let Some(census) = &mut self.census {
            match array
                .try_allocation_info()
                .map_err(ResidencyError::OriginalArrayInspection)?
            {
                Some(info) if info.bytes() != 0 => census.add()?,
                Some(_) => {}
                None => self.incomplete = true,
            }
            return Ok(());
        }
        if self.original.is_some() {
            let info = array
                .try_allocation_info()
                .map_err(array_inspection_error)?;
            if let Some(info) = info {
                let bytes = checked_bytes(info.bytes())?;
                self.check_backing_capacity(info.identity(), bytes)?;
                if self.array_entry(&info.identity()).is_some() || bytes == 0 {
                    return Ok(());
                }
                self.original.as_ref().unwrap().ensure_room()?;
                let retained = self.original.as_mut().unwrap().clone_array(array)?;
                return self.original.as_mut().unwrap().insert(
                    Some(StorageIdentity::Native(info.identity())),
                    original::Value::Array((bytes, retained)),
                );
            }
            if std::env::var_os("EREDU_HOST_SAVED_SOURCE_DIAGNOSTICS").is_some() {
                eprintln!("NATIVE_PUBLICATION_UNKNOWN_ARRAY at {} shape={:?} dtype={:?}",
                    std::panic::Location::caller(), array.shape(), array.dtype());
            }
            self.original.as_ref().unwrap().ensure_room()?;
            let retained = self.original.as_mut().unwrap().clone_array(array)?;
            return self
                .original
                .as_mut()
                .unwrap()
                .insert(None, original::Value::UnknownArray(retained));
        }
        let allocation = array
            .try_allocation_info()
            .map_err(array_inspection_error)?;
        match allocation {
            Some(info) => {
                let bytes = checked_bytes(info.bytes())?;
                self.check_backing_capacity(info.identity(), bytes)?;
                if let Some((prior, _)) = self.arrays.get(&info.identity()) {
                    require_same_capacity(*prior, bytes)?;
                } else if bytes != 0 {
                    // Clone only a handle that will actually be retained. A
                    // duplicate's temporary handle/drop could run housekeeping.
                    // Both fallible native observations precede mutation.
                    let retained = array
                        .try_clone_for_inspection()
                        .map_err(array_inspection_error)?;
                    self.arrays.insert(info.identity(), (bytes, retained));
                }
            }
            None => {
                let retained = array
                    .try_clone_for_inspection()
                    .map_err(array_inspection_error)?;
                // Even if another alias completed after the snapshot, keep
                // that observation unknown rather than silently upgrading it.
                self.unknown_arrays.push(retained);
            }
        }
        Ok(())
    }

    /// Adds the whole immutable host allocation. Rust transfer buffers have
    /// unique ownership before freezing. Certified array aliases use the same
    /// native owner identity and count once across both storage categories.
    /// Inspection does not run native housekeeping or reclaim unrelated owners;
    /// runtime contention returns a typed error before this inventory changes.
    pub fn include_host(
        &mut self,
        buffer: Arc<ImmutableHostTransferBuffer>,
    ) -> Result<(), ResidencyError> {
        self.include_retained_host(buffer.into())
    }

    pub(crate) fn include_retained_host(
        &mut self,
        buffer: RetainedHostBuffer,
    ) -> Result<(), ResidencyError> {
        let info = buffer.try_allocation_info().map_err(|cause| {
            if self.census.is_some() {
                ResidencyError::OriginalHostInspection(cause)
            } else {
                host_inspection_error(cause)
            }
        })?;
        let identity = info.identity();
        let bytes = checked_bytes(info.bytes())?;
        if let Some(census) = &mut self.census {
            if bytes != 0 {
                census.add()?;
            }
            return Ok(());
        }
        self.check_backing_capacity(identity, bytes)?;
        if let Some(original) = self.original.as_mut() {
            if bytes == 0 {
                return Ok(());
            }
            return original.insert(
                Some(StorageIdentity::Native(identity)),
                original::Value::Host((bytes, buffer)),
            );
        }
        if let Some((prior, _)) = self.hosts.get(&identity) {
            require_same_capacity(*prior, bytes)?;
        } else if bytes != 0 {
            self.hosts.insert(identity, (bytes, buffer));
        }
        Ok(())
    }

    /// Borrow the immutable source captured by actual native setup. Missing
    /// source coverage remains unknown; no query or payload copy runs here.
    pub(crate) fn include_group_buffer(
        &mut self,
        source: Option<&safemlx::distributed::RetainedGroupBuffer>,
    ) -> Result<(), ResidencyError> {
        let Some(source) = source else {
            self.mark_incomplete();
            return Ok(());
        };
        let bytes = checked_bytes(source.bytes())?;
        if bytes == 0 {
            return Ok(());
        }
        if let Some(census) = &mut self.census {
            return census.add();
        }
        let identity = source.identity();
        if let Some((prior, _)) = self
            .group_buffer_entries()
            .find_map(|(key, value)| (*key == identity).then_some(value))
        {
            return require_same_capacity(*prior, bytes);
        }
        if let Some(original) = &mut self.original {
            original.ensure_room()?;
            return original.insert(
                Some(StorageIdentity::GroupBuffer(identity)),
                original::Value::GroupBuffer((bytes, source.clone())),
            );
        }
        self.group_buffers.insert(identity, (bytes, source.clone()));
        Ok(())
    }

    /// Adds source-owned payload capacity, preserving physical source aliases.
    pub fn include_sources(
        &mut self,
        sources: Option<SourceStorage>,
    ) -> Result<(), ResidencyError> {
        if self.original.is_some() || self.census.is_some() {
            // Original visitors retain SourceStorageRef directly. An already
            // allocated ordinary source map cannot become an admitted owner.
            self.incomplete = true;
            return Err(original::failure(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        match sources {
            Some(sources) => self.sources.merge(sources)?,
            None => self.incomplete = true,
        }
        Ok(())
    }

    /// Retains an immutable byte allocation, such as a buffered cache shard.
    /// Arc slices have no spare payload capacity beyond their length.
    pub(crate) fn include_bytes(&mut self, bytes: Arc<[u8]>) {
        if self.census.is_some() {
            self.incomplete |= !bytes.is_empty();
            return;
        }
        if let Some(original) = self.original.as_mut() {
            if !bytes.is_empty() {
                let _ = original.refuse(original::Value::Bytes(bytes));
                self.incomplete = true;
            }
            return;
        }
        if !bytes.is_empty() {
            self.byte_buffers
                .entry(bytes.as_ptr() as usize)
                .or_insert(bytes);
        }
    }

    /// Includes one actual immutable metadata owner. Shared aliases retain one
    /// payload-free identity and one exact capacity; no raw byte estimate enters.
    pub(crate) fn include_metadata(
        &mut self,
        owner: eredu_runtime::SharedHostMetadata,
    ) -> Result<(), ResidencyError> {
        if let Some(census) = &mut self.census {
            match owner.validate_original_attachment(census.pool.shared_storage_domain()) {
                Ok(()) => {
                    if owner.capacity_bytes().is_some() {
                        census.add()?;
                    } else {
                        self.incomplete = true;
                    }
                }
                Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound) => {
                    self.incomplete = true
                }
                Err(cause) => return Err(original::failure(cause)),
            }
            return Ok(());
        }
        if let Some(original) = self.original.as_mut() {
            original
                .custody()
                .validate_metadata(&owner)
                .map_err(|cause| original::failure(cause))?;
            let Some(bytes) = owner.capacity_bytes() else {
                self.incomplete = true;
                return original.insert(None, original::Value::UnknownMetadata(owner));
            };
            let key = StorageIdentity::HostMetadata(owner.identity().registry_key().clone());
            return original.insert(Some(key), original::Value::Metadata((bytes, owner)));
        }
        let Some(bytes) = owner.capacity_bytes() else {
            self.incomplete = true;
            self.unknown_metadata.push(owner);
            return Ok(());
        };
        if let Some((prior, _)) = self.metadata.get(owner.identity()) {
            require_same_capacity(*prior, bytes)?;
        } else {
            self.metadata
                .insert(owner.identity().clone(), (bytes, owner));
        }
        Ok(())
    }

    /// Includes a closed token for one actual fixed host slot allocation.
    /// Only its checked inline extent is counted; nested slot payloads require
    /// their own inventory. Tokens do not retain table contents or grant source
    /// access. Publication checks live attachment before releasing these tokens.
    pub(crate) fn include_slot_metadata(
        &mut self,
        token: eredu_runtime::HostSlotMetadata,
    ) -> Result<(), ResidencyError> {
        if let Some(census) = &mut self.census {
            match token.validate_original_attachment(census.pool.shared_storage_domain()) {
                Ok(()) => {
                    if token.capacity_bytes().is_some() {
                        census.add()?;
                    } else {
                        self.incomplete = true;
                    }
                }
                Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound) => {
                    self.incomplete = true
                }
                Err(cause) => return Err(original::failure(cause)),
            }
            return Ok(());
        }
        if let Some(original) = self.original.as_mut() {
            original
                .custody()
                .validate_slot_metadata(&token)
                .map_err(|cause| original::failure(cause))?;
            let Some(bytes) = token.capacity_bytes() else {
                self.incomplete = true;
                return original.insert(None, original::Value::UnknownSlot(token));
            };
            let key = StorageIdentity::HostMetadata(token.identity().registry_key().clone());
            return original.insert(Some(key), original::Value::Slot((bytes, token)));
        }
        let Some(bytes) = token.capacity_bytes() else {
            self.incomplete = true;
            self.unknown_slot_metadata.push(token);
            return Ok(());
        };
        if let Some((prior, _)) = self.slot_metadata.get(token.identity()) {
            require_same_capacity(*prior, bytes)?;
        } else {
            self.slot_metadata
                .insert(token.identity().clone(), (bytes, token));
        }
        Ok(())
    }

    /// Includes the exact immutable admission owner and all its retained nested
    /// capacities. Aliases share a payload-free storage key; independently
    /// admitted equal plans remain different allocations. No plan reconstruction,
    /// native work, construction bound or capture authority is inferred here.
    pub(crate) fn include_capture_plan(
        &mut self,
        owner: eredu_core::capture::SharedCapturePlan,
    ) -> Result<(), ResidencyError> {
        if self.census.is_some() {
            self.incomplete = true;
            return Ok(());
        }
        if let Some(original) = self.original.as_mut() {
            self.incomplete = true;
            return Err(original.refuse(original::Value::UnknownCapture(owner)));
        }
        let Some(bytes) = owner.capacity_bytes() else {
            self.incomplete = true;
            self.unknown_capture_plans.push(owner);
            return Ok(());
        };
        if let Some((prior, _)) = self.capture_plans.get(owner.storage_identity()) {
            require_same_capacity(*prior, bytes)?;
        } else {
            self.capture_plans
                .insert(owner.storage_identity().clone(), (bytes, owner));
        }
        Ok(())
    }

    /// Combines inventories while keeping identities retained throughout merging.
    pub fn merge(&mut self, other: Self) -> Result<(), ResidencyError> {
        if self.original.is_some()
            || other.original.is_some()
            || self.census.is_some()
            || other.census.is_some()
        {
            self.incomplete = true;
            return Err(original::failure(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        for (identity, (bytes, array)) in other.arrays {
            self.check_backing_capacity(identity, bytes)?;
            if let Some((prior, _)) = self.arrays.get(&identity) {
                require_same_capacity(*prior, bytes)?;
            } else {
                self.arrays.insert(identity, (bytes, array));
            }
        }
        for (identity, (bytes, buffer)) in other.hosts {
            self.check_backing_capacity(identity, bytes)?;
            if let Some((prior, _)) = self.hosts.get(&identity) {
                require_same_capacity(*prior, bytes)?;
            } else {
                self.hosts.insert(identity, (bytes, buffer));
            }
        }
        for (_, (_, source)) in other.group_buffers {
            self.include_group_buffer(Some(&source))?;
        }
        for (_, (_, owner)) in other.metadata {
            self.include_metadata(owner)?;
        }
        for (_, (_, token)) in other.slot_metadata {
            self.include_slot_metadata(token)?;
        }
        for (_, (_, owner)) in other.capture_plans {
            self.include_capture_plan(owner)?;
        }
        self.sources.merge(other.sources)?;
        self.byte_buffers.extend(other.byte_buffers);
        self.unknown_arrays.extend(other.unknown_arrays);
        self.unknown_metadata.extend(other.unknown_metadata);
        self.unknown_capture_plans
            .extend(other.unknown_capture_plans);
        self.unknown_slot_metadata
            .extend(other.unknown_slot_metadata);
        self.incomplete |= other.incomplete;
        Ok(())
    }

    /// Unique retained physical payload capacity. Unknown resources never become
    /// a logical-size estimate. The value cannot establish future allocation cost.
    pub fn byte_bound(&self) -> Result<Option<u64>, ResidencyError> {
        if self.census.is_some() {
            return Ok(None);
        }
        if let Some(original) = &self.original {
            return if self.incomplete {
                Ok(None)
            } else {
                original.byte_bound()
            };
        }
        if self.incomplete || !self.unknown_arrays.is_empty() {
            return Ok(None);
        }
        let buffers =
            self.byte_buffers
                .values()
                .try_fold(self.sources.bytes()?, |total, bytes| {
                    total.checked_add(checked_bytes(bytes.len())?).ok_or(
                        ResidencyError::ArithmeticOverflow {
                            context: "retained byte buffer total",
                        },
                    )
                })?;
        self.arrays
            .values()
            .map(|(bytes, _)| *bytes)
            .chain(self.group_buffers.values().map(|(bytes, _)| *bytes))
            .chain(self.metadata.values().map(|(bytes, _)| *bytes))
            .chain(self.capture_plans.values().map(|(bytes, _)| *bytes))
            .chain(self.slot_metadata.values().map(|(bytes, _)| *bytes))
            .chain(
                self.hosts
                    .iter()
                    .filter(|(identity, _)| !self.arrays.contains_key(identity))
                    .map(|(_, (bytes, _))| *bytes),
            )
            .try_fold(buffers, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or(ResidencyError::ArithmeticOverflow {
                        context: "retained physical storage total",
                    })
            })
            .map(Some)
    }

    /// Resources retained outside this inventory, such as an in-flight transfer,
    /// prevent the owner from supplying a complete snapshot.
    pub(crate) fn mark_incomplete(&mut self) {
        self.incomplete = true;
    }

    fn check_backing_capacity(
        &self,
        identity: safemlx::AllocationIdentity,
        bytes: u64,
    ) -> Result<(), ResidencyError> {
        if let Some((prior, _)) = self.array_entry(&identity) {
            require_same_capacity(*prior, bytes)?;
        }
        if let Some((prior, _)) = self.host_entry(&identity) {
            require_same_capacity(*prior, bytes)?;
        }
        Ok(())
    }
}

fn checked_bytes(bytes: usize) -> Result<u64, ResidencyError> {
    u64::try_from(bytes).map_err(|_| ResidencyError::ArithmeticOverflow {
        context: "retained physical storage capacity",
    })
}

fn require_same_capacity(prior: u64, bytes: u64) -> Result<(), ResidencyError> {
    if prior != bytes {
        return Err(native_error(
            "retained allocation identity",
            safemlx::error::Exception::custom("retained allocation capacity changed"),
        ));
    }
    Ok(())
}

fn array_inspection_error(source: safemlx::ArrayMetadataError) -> ResidencyError {
    native_error(
        "retained array allocation inspection",
        safemlx::error::Exception::from_source(source),
    )
}

fn host_inspection_error(source: safemlx::HostTransferMetadataError) -> ResidencyError {
    native_error(
        "retained host allocation inspection",
        safemlx::error::Exception::from_source(source),
    )
}

fn native_error(operation: &'static str, source: safemlx::error::Exception) -> ResidencyError {
    ResidencyError::Mlx {
        id: eredu_core::residency::OffloadUnitId::new("retained-storage")
            .expect("nonempty internal storage identifier"),
        operation,
        source,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod cold_tests;
