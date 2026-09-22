//! Native inventory ownership alongside neutral shared-memory accounting.

use super::*;
use crate::backend::error::Error;
use crate::backend::ordinary_retirement::OrdinaryRetirement;
use eredu_runtime::working_memory::{
    MemoryLedger, StorageAllocation, UnquotedOriginalSlotSources, WorkingMemoryError,
    WorkingMemoryStorage,
};

mod publication;
mod source_pin;
pub(crate) use publication::{
    retain_copy_publication_failure, CopyPublicationLayout, PendingCopyPublication,
    PendingNativePublication, RetainedStoragePublication,
};
pub(crate) use source_pin::SourcePinPlan;

/// Pin the Arc allocation's address while it is used as an accounting key.
/// Weak ownership does not keep the strong payload alive; for an inline byte
/// slice, however, its allocation remains reserved until the key retires.
#[derive(Clone, Debug)]
pub(crate) struct ByteStorageIdentity(std::sync::Weak<[u8]>);

impl ByteStorageIdentity {
    /// Reuse the canonical weak allocation key without copying bytes or allocating.
    fn from_bytes(bytes: &Arc<[u8]>) -> Self {
        Self(Arc::downgrade(bytes))
    }

    fn address(&self) -> usize {
        self.0.as_ptr() as *const u8 as usize
    }
}

impl PartialEq for ByteStorageIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.address() == other.address()
    }
}
impl Eq for ByteStorageIdentity {}
impl PartialOrd for ByteStorageIdentity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByteStorageIdentity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.address().cmp(&other.address())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum StorageIdentity {
    Native(safemlx::AllocationIdentity),
    NativeControl(safemlx::AllocationIdentity),
    GroupBuffer(safemlx::distributed::GroupBufferIdentity),
    Source(eredu_checkpoint::store::SourceStorageIdentity),
    Bytes(ByteStorageIdentity),
    HostMetadata(eredu_runtime::HostMetadataKey),
    CapturePlan(eredu_core::SharedStorageIdentity),
}

impl StorageIdentity {
    /// Exact existing byte-allocation key; only an Arc weak count is retained.
    pub(crate) fn from_bytes(bytes: &Arc<[u8]>) -> Self {
        Self::Bytes(ByteStorageIdentity::from_bytes(bytes))
    }
}

impl eredu_runtime::working_memory::HostSlotStorageKey for StorageIdentity {
    fn from_host_slot_identity(identity: eredu_runtime::HostMetadataKey) -> Option<Self> {
        Some(Self::HostMetadata(identity))
    }

    fn host_slot_identity(&self) -> Option<&eredu_runtime::HostMetadataKey> {
        match self {
            Self::HostMetadata(identity) => Some(identity),
            Self::Native(_)
            | Self::NativeControl(_)
            | Self::GroupBuffer(_)
            | Self::Source(_)
            | Self::Bytes(_)
            | Self::CapturePlan(_) => None,
        }
    }
}

/// Registered existing storage. Its inventory keeps physical identities valid;
/// dropping the last handle stages safe host cleanup. That cleanup releases
/// the physical roots before releasing their charge, outside native locks.
///
/// Retain this with a session or operation for as long as the covered allocations
/// belong to its managed domain. Fixed host-slot tokens retain accounting only,
/// so callers also retain their actual tables for access to those contents.
/// This does not reserve future materialization,
/// copies or execution workspace. Clones share both storage and accounting.
#[derive(Clone)]
#[must_use = "retain registered storage alongside the session or operation using it"]
pub struct RetainedStorageReservation(std::rc::Rc<OrdinaryRetirement<RegisteredStorage>>);

impl std::fmt::Debug for RetainedStorageReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedStorageReservation")
            .field("inventory", self.inventory())
            .field("bytes", &self.bytes())
            .finish()
    }
}

#[derive(Debug)]
struct RegisteredStorage {
    // Field order matters: native roots retire before their charge can be freed.
    inventory: RetainedStorage,
    charge: WorkingMemoryStorage<StorageIdentity>,
}

impl RetainedStorageReservation {
    /// The immutable certified inventory retained by this registration.
    pub fn inventory(&self) -> &RetainedStorage {
        &self.0.inventory
    }

    /// Full inventory capacity, including aliases shared with other registrations.
    /// Only the pool's used-byte report gives total domain usage.
    pub fn bytes(&self) -> Option<u64> {
        self.0.charge.bytes()
    }
}

// This wrapper belongs only to the explicitly unquoted ordinary population.
// Cause retirement precedes the exact original table witnesses on failure.
#[derive(Debug)]
struct OriginalSourceFailure {
    cause: Error,
    _sources: UnquotedOriginalSlotSources,
}
impl std::fmt::Display for OriginalSourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalSourceFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
fn original_source_failure(cause: Error, sources: UnquotedOriginalSlotSources) -> Error {
    if !sources.has_custody() {
        cause
    } else {
        Error::StorageSource(eredu_core::BackendFailure::from_error(
            OriginalSourceFailure {
                cause,
                _sources: sources,
            },
        ))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct PreparedOriginalSourceFailure {
    #[source]
    cause: Error,
    _sources: eredu_runtime::working_memory::RetainedOriginalStorageSources,
}

impl RetainedStorage {
    /// Complete opening publication population from actual retained owners.
    /// The initial original route excludes arbitrary bytes/capture identities;
    /// source constructors must belong to this same pool. Future native births
    /// are supplied separately by the selected lowering, never a byte cap.
    pub(crate) fn original_publication_rows(
        &self,
        pool: &MemoryLedger,
    ) -> Result<Option<usize>, Error> {
        if let Some(census) = &self.census {
            if census.overflowed {
                return Err(Error::PrefillControl(WorkingMemoryError::Overflow));
            }
            return Ok((census.original && !self.incomplete).then_some(census.rows));
        }
        if self.validate_inventory_completeness().is_err()
            || !self.byte_buffers.is_empty()
            || !self.capture_plans.is_empty()
        {
            return Ok(None);
        }
        for (identity, bytes) in self.source_capacities() {
            match pool.validate_retained_source_inventory(&StorageIdentity::Source(identity), bytes)
            {
                Ok(()) => {}
                Err(WorkingMemoryError::UnknownBound) => return Ok(None),
                Err(cause) => return Err(Error::PrefillControl(cause)),
            }
        }
        let controls = self
            .array_entries()
            .try_fold(0usize, |count, (_, (_, array))| {
                let facts = array
                    .try_allocation_info()
                    .map_err(array_inspection_error)?
                    .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
                count
                    .checked_add(usize::from(facts.host_control_bytes() != 0))
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
            })?;
        let host_controls = self
            .host_entries()
            .try_fold(0usize, |count, (_, (_, host))| {
                let facts = host.try_allocation_info().map_err(host_inspection_error)?;
                count
                    .checked_add(usize::from(facts.host_control_bytes() != 0))
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
            })?;
        // Counting both sides of a captured alias is conservative and uses
        // actual rows. Registry commit still deduplicates physical identities.
        [
            controls,
            host_controls,
            self.array_entries().count(),
            self.group_buffer_entries().count(),
            self.host_entries().count(),
            self.source_capacities().count(),
            self.metadata_entries().count(),
            self.slot_entries().count(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .map(Some)
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
    }

    /// Validates the complete fixed-table population before any adoption. The
    /// expected token comes only from this model quote's original source slot.
    /// This performs no witness collection or registration mutation.
    pub(crate) fn validate_original_table(
        &self,
        pool: &MemoryLedger,
        expected: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
    ) -> Result<(), Error> {
        self.validate_original_sources(pool, expected, None)
    }

    /// The supplied B profile comes from the actual accepted quote's source
    /// projection. Every original cache must match that materialization and pool;
    /// authenticating residence alone never permits an unrelated original source.
    pub(crate) fn validate_original_sources(
        &self,
        pool: &MemoryLedger,
        expected: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    ) -> Result<(), Error> {
        if prepared.is_some_and(|source| !source.pool().same_ledger(pool)) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        for (_, (_, metadata)) in self.metadata_entries() {
            if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
                if let Some(result) = input.original_residence(pool) {
                    let Some(source) = prepared else {
                        // Keep the table-only/ordinary caller's existing refusal
                        // and source projection unchanged.
                        result.map_err(|error| Error::Other(Box::new(error)))?;
                        return Err(Error::text_admission(WorkingMemoryError::UnknownBound));
                    };
                    result.map_err(Error::PrefillControl)?;
                    if !source.matches_cache_identity(input) {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                }
            }
        }
        self.validate_inventory_completeness()?;
        if let Some(expected) = expected {
            pool.pin_original_reset_slots(expected.metadata())
                .map_err(|e| Error::Other(Box::new(e)))?;
        }
        let mut seen_root = false;
        for (_, token) in self.slot_entries().map(|(_, value)| value) {
            let source = pool
                .classify_host_slot_source(token)
                .map_err(|e| Error::Other(Box::new(e)))?;
            if source.registered().is_none() {
                let exact = expected.is_some_and(|expected| expected.same_constructor(token));
                if !exact {
                    return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
                }
                seen_root |=
                    expected.is_some_and(|expected| expected.metadata().same_storage(token));
            }
        }
        if seen_root != expected.is_some() {
            return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
        }
        Ok(())
    }

    /// Pins only already-published origins from this complete actual inventory.
    /// It does not admit or attach new source storage. The caller separately
    /// retains actual payload ownership through copying and native recovery.
    pub(crate) fn pin_registered(
        &self,
        pool: &MemoryLedger,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, Error> {
        let maximum = self.generic_publication_rows(pool)?;
        let prepared = prepare_storage_publication(pool, maximum).map_err(Error::PrefillControl)?;
        let (entries, mut original) =
            self.source_entries_prepared(pool, None, maximum, prepared.host_authority())?;
        prepared
            .pin_registered_storage(
                entries
                    .into_iter()
                    .map(|(key, allocation)| (key, allocation.capacity_bytes())),
            )
            .and_then(|storage| storage.with_original_reset_sources(&mut original))
            .map_err(|error| original_source_failure(Error::Other(Box::new(error)), original))
    }

    /// Existing-only source pinning under an already accepted enclosing host
    /// plan. Original tables and prepared inputs retain their real constructor
    /// custody; they never become ordinary registrations or new capacity credit.
    pub(crate) fn pin_registered_with_host(
        &self,
        pool: &MemoryLedger,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, Error> {
        self.source_pin_plan(pool)?.pin(self, pool, host)
    }

    /// Registers certified existing payloads atomically in the ledger that
    /// admits requests. Separate model/source inventories share charges for
    /// identical backing, including host/device aliases on unified memory.
    /// Unknown storage is rejected before accounting changes or native work.
    pub fn register(self, pool: &MemoryLedger) -> Result<RetainedStorageReservation, Error> {
        let maximum = self.generic_publication_rows(pool)?;
        let prepared = prepare_storage_publication(pool, maximum).map_err(Error::PrefillControl)?;
        let (entries, mut original) =
            self.source_entries_prepared(pool, None, maximum, prepared.host_authority())?;
        let charge = prepared
            .register_storage_with_gguf_sources(entries)
            .and_then(|storage| storage.with_original_reset_sources(&mut original))
            .map_err(|error| original_source_failure(Error::Other(Box::new(error)), original))?;
        Ok(RetainedStorageReservation(std::rc::Rc::new(
            OrdinaryRetirement::new(RegisteredStorage {
                inventory: self,
                charge,
            }),
        )))
    }

    // Original population construction is available only with the actual
    // ordinary participant. Pool-only/funded callers reject before any Vec.
    fn source_entries(
        &self,
        pool: &MemoryLedger,
        owner: Option<&crate::backend::managed_memory::NativeMemoryOwner>,
    ) -> Result<
        (
            Vec<(StorageIdentity, StorageAllocation)>,
            UnquotedOriginalSlotSources,
        ),
        Error,
    > {
        self.source_entries_inner(pool, owner, None)
    }

    fn source_entries_prepared(
        &self,
        pool: &MemoryLedger,
        owner: Option<&crate::backend::managed_memory::NativeMemoryOwner>,
        maximum: usize,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<
        (
            Vec<(StorageIdentity, StorageAllocation)>,
            UnquotedOriginalSlotSources,
        ),
        Error,
    > {
        self.source_entries_inner(pool, owner, Some((maximum, host)))
    }

    fn source_entries_inner(
        &self,
        pool: &MemoryLedger,
        owner: Option<&crate::backend::managed_memory::NativeMemoryOwner>,
        preparation: Option<(usize, &eredu_core::HostPreparationAuthority)>,
    ) -> Result<
        (
            Vec<(StorageIdentity, StorageAllocation)>,
            UnquotedOriginalSlotSources,
        ),
        Error,
    > {
        self.validate_inventory_completeness()?;
        // Every originally constructed cache remains an explicit inventory
        // owner. Ordinary publication can consume its authenticated residence;
        // pool-only/funded callers need D's separate source profile first.
        for (_, (_, metadata)) in self.metadata_entries() {
            if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
                if let Some(result) = input.original_residence(pool) {
                    result.map_err(|error| Error::Other(Box::new(error)))?;
                    let owner = owner
                        .ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?;
                    if !owner.pool().same_ledger(pool) {
                        return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
                    }
                }
            }
        }
        let mut has_original = false;
        for (_, token) in self.slot_entries().map(|(_, value)| value) {
            let source = pool
                .classify_host_slot_source(token)
                .map_err(|error| Error::Other(Box::new(error)))?;
            has_original |= source.registered().is_none();
        }
        let mut original = if has_original {
            let owner =
                owner.ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?;
            if !owner.pool().same_ledger(pool) {
                return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
            }
            let (maximum, host) =
                preparation.ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            UnquotedOriginalSlotSources::prepare(&owner.unquoted_lease()?, maximum, host)
                .map_err(Error::PrefillControl)?
        } else {
            UnquotedOriginalSlotSources::default()
        };
        for (_, token) in self.slot_entries().map(|(_, value)| value) {
            let source = match pool.classify_host_slot_source(token) {
                Ok(source) => source,
                Err(error) => {
                    return Err(original_source_failure(
                        Error::Other(Box::new(error)),
                        original,
                    ));
                }
            };
            if source.registered().is_none() {
                if let Err(error) = original.push(token) {
                    return Err(original_source_failure(
                        Error::Other(Box::new(error)),
                        original,
                    ));
                }
            }
        }
        let mut entries = match self
            .placed_storage_entries_bounded(pool, preparation.map(|(maximum, _)| maximum))
        {
            Ok(entries) => entries,
            Err(error) => return Err(original_source_failure(error, original)),
        };
        entries.retain(|(key, _)| !original.sources().iter().any(|source| {
            matches!(key, StorageIdentity::HostMetadata(key) if key == source.metadata().identity().registry_key())
        }));
        entries.retain(|(key,_)| !self.metadata_entries().any(|(identity,(_,metadata))| {
            matches!(metadata,eredu_runtime::SharedHostMetadata::Input(input) if input.original_source().is_some())
                && key==&StorageIdentity::HostMetadata(identity.registry_key().clone())
        }));
        Ok((entries, original))
    }

    /// Counts every possible row from borrowed allocation facts without allocating.
    pub(crate) fn generic_publication_rows(&self, pool: &MemoryLedger) -> Result<usize, Error> {
        if let Some(census) = &self.census {
            if census.overflowed {
                return Err(Error::PrefillControl(WorkingMemoryError::Overflow));
            }
            if !census.pool.same_ledger(pool) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            return (!self.incomplete)
                .then_some(census.rows)
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        self.validate_physical_attribution(pool)?;
        let mut rows = 0usize;
        let mut add = |n| {
            rows = rows
                .checked_add(n)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            Ok::<_, Error>(())
        };
        for (_, (_, array)) in self.array_entries() {
            let facts = array
                .try_allocation_info()
                .map_err(array_inspection_error)?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            add(1 + usize::from(facts.host_control_bytes() != 0))?;
        }
        for (identity, (_, host)) in self.host_entries() {
            if self.array_entry(identity).is_none() {
                let facts = host.try_allocation_info().map_err(host_inspection_error)?;
                add(1 + usize::from(facts.host_control_bytes() != 0))?;
            }
        }
        for count in [
            self.group_buffer_entries().count(),
            self.source_capacities().count(),
            self.byte_values().count(),
            self.metadata_entries().count(),
            self.slot_entries().count(),
            self.capture_entries().count(),
        ] {
            add(count)?;
        }
        Ok(rows)
    }

    fn storage_entries(&self) -> Result<Vec<(StorageIdentity, u64)>, Error> {
        self.storage_entries_with_capacity(self.original_maximum_rows())
    }

    /// Validates each original backing's physical attribution without forming
    /// an aggregate byte total or allocating a second inventory.
    pub(crate) fn validate_physical_attribution(&self, pool: &MemoryLedger) -> Result<(), Error> {
        self.validate_inventory_completeness()?;
        for (identity, (bytes, array)) in self.array_entries() {
            let info = array
                .allocation_info()?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            if info.identity() != *identity || u64::try_from(info.bytes()).ok() != Some(*bytes) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            crate::backend::managed_memory::allocation_placement_handle(&info, pool)
                .map_err(|cause| Error::PrefillControl(cause.into()))?;
        }
        for (identity, (bytes, host)) in self.host_entries() {
            let info = host.allocation_info()?;
            if info.identity() != *identity || u64::try_from(info.bytes()).ok() != Some(*bytes) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            crate::backend::managed_memory::allocation_placement_handle(&info, pool)
                .map_err(|cause| Error::PrefillControl(cause.into()))?;
        }
        for (_, (_, buffer)) in self.group_buffer_entries() {
            if buffer.allocation_placement() != safemlx::AllocationPlacement::Host {
                return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
            }
        }
        // Remaining rows are controlled host allocations; publication uses the
        // same host placement and authenticates their source identities.
        Ok(())
    }

    pub(crate) fn has_no_payload(&self) -> Result<bool, Error> {
        self.validate_inventory_completeness()?;
        Ok(!self.array_entries().any(|(_, (bytes, _))| *bytes != 0)
            && !self.host_entries().any(|(_, (bytes, _))| *bytes != 0)
            && !self
                .group_buffer_entries()
                .any(|(_, (bytes, _))| *bytes != 0)
            && !self.metadata_entries().any(|(_, (bytes, _))| *bytes != 0)
            && !self.slot_entries().any(|(_, (bytes, _))| *bytes != 0)
            && !self.source_capacities().any(|(_, bytes)| bytes != 0)
            && !self.byte_values().any(|bytes| !bytes.is_empty())
            && !self.capture_entries().any(|(_, (bytes, _))| *bytes != 0))
    }

    fn validate_inventory_completeness(&self) -> Result<(), Error> {
        if self.census.is_some()
            || self.incomplete
            || !self.unknown_arrays.is_empty()
            || self
                .original
                .as_ref()
                .is_some_and(|v| !v.has_complete_capacity_facts())
            || !self.unknown_metadata.is_empty()
            || !self.unknown_slot_metadata.is_empty()
            || !self.unknown_capture_plans.is_empty()
        {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        Ok(())
    }

    fn placed_storage_entries(
        &self,
        pool: &MemoryLedger,
    ) -> Result<Vec<(StorageIdentity, StorageAllocation)>, Error> {
        self.placed_storage_entries_bounded(pool, None)
    }

    fn placed_storage_entries_bounded(
        &self,
        pool: &MemoryLedger,
        maximum: Option<usize>,
    ) -> Result<Vec<(StorageIdentity, StorageAllocation)>, Error> {
        let entries = if let Some(maximum) = maximum {
            self.storage_entries_with_capacity_mode(Some(maximum), false)?
        } else {
            self.storage_entries()?
        };
        let mut placed = Vec::new();
        placed
            .try_reserve_exact(entries.capacity())
            .map_err(|cause| {
                Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(cause))
            })?;
        if placed.capacity() != entries.capacity() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        for (key, bytes) in entries {
            let placement = match &key {
                StorageIdentity::Native(identity) => {
                    let info = if let Some((_, array)) = self.array_entry(identity) {
                        array
                            .allocation_info()?
                            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?
                    } else if let Some((_, host)) = self.host_entry(identity) {
                        host.allocation_info()?
                    } else {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    };
                    if info.identity() != *identity
                        || u64::try_from(info.bytes()).ok() != Some(bytes)
                    {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                    crate::backend::managed_memory::allocation_placement_handle(&info, pool)
                        .map_err(|cause| Error::PrefillControl(cause.into()))?
                }
                StorageIdentity::GroupBuffer(identity) => {
                    let (_, (_, buffer)) = self
                        .group_buffer_entries()
                        .find(|(id, _)| *id == identity)
                        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
                    if buffer.allocation_placement() != safemlx::AllocationPlacement::Host {
                        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
                    }
                    pool.host_placement_handle()
                }
                StorageIdentity::NativeControl(_)
                | StorageIdentity::Source(_)
                | StorageIdentity::Bytes(_)
                | StorageIdentity::HostMetadata(_)
                | StorageIdentity::CapturePlan(_) => pool.host_placement_handle(),
            };
            placed.push((key, StorageAllocation::new(bytes, placement)));
        }
        Ok(placed)
    }

    fn storage_entries_with_capacity(
        &self,
        maximum: Option<usize>,
    ) -> Result<Vec<(StorageIdentity, u64)>, Error> {
        self.storage_entries_with_capacity_mode(maximum, maximum.is_some())
    }

    fn storage_entries_with_capacity_mode(
        &self,
        maximum: Option<usize>,
        original_only: bool,
    ) -> Result<Vec<(StorageIdentity, u64)>, Error> {
        self.validate_inventory_completeness()?;
        let mut entries = Vec::new();
        if let Some(rows) = maximum {
            entries
                .try_reserve_exact(rows)
                .map_err(|e| Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(e)))?;
            if entries.capacity() != rows {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        let fixed = maximum.is_some();
        let mut push =
            |key, bytes| {
                if fixed && entries.len() == entries.capacity() {
                    return Err(Error::PrefillControl(WorkingMemoryError::CollectorCapacity {
                    kind: eredu_runtime::working_memory::CollectorCapacityKind::PublicationEntries,
                    used: entries.len(),
                    capacity: entries.capacity(),
                }));
                }
                entries.push((key, bytes));
                Ok(())
            };
        for (identity, (bytes, _)) in self.group_buffer_entries() {
            push(StorageIdentity::GroupBuffer(*identity), *bytes)?;
        }
        for (identity, (bytes, array)) in self.array_entries() {
            push(StorageIdentity::Native(*identity), *bytes)?;
            let facts = array
                .try_allocation_info()
                .map_err(array_inspection_error)?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            if facts.identity() != *identity || u64::try_from(facts.bytes()).ok() != Some(*bytes) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            let controls = checked_bytes(facts.host_control_bytes())?;
            if controls != 0 {
                push(StorageIdentity::NativeControl(*identity), controls)?;
            }
        }
        for (identity, (bytes, host)) in self.host_entries() {
            if self.array_entry(identity).is_none() {
                push(StorageIdentity::Native(*identity), *bytes)?;
                let facts = host.try_allocation_info().map_err(host_inspection_error)?;
                if facts.identity() != *identity
                    || u64::try_from(facts.bytes()).ok() != Some(*bytes)
                {
                    return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                }
                let controls = checked_bytes(facts.host_control_bytes())?;
                if controls != 0 {
                    push(StorageIdentity::NativeControl(*identity), controls)?;
                }
            }
        }
        for (identity, bytes) in self.source_capacities() {
            push(StorageIdentity::Source(identity), bytes)?;
        }
        for bytes in self.byte_values() {
            if original_only {
                return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
            }
            push(
                StorageIdentity::Bytes(ByteStorageIdentity::from_bytes(bytes)),
                checked_bytes(bytes.len())?,
            )?;
        }
        for (identity, (bytes, _)) in self.metadata_entries() {
            push(
                StorageIdentity::HostMetadata(identity.registry_key().clone()),
                *bytes,
            )?;
        }
        for (identity, (bytes, _)) in self.slot_entries() {
            push(
                StorageIdentity::HostMetadata(identity.registry_key().clone()),
                *bytes,
            )?;
        }
        for (identity, (bytes, _)) in self.capture_entries() {
            if original_only {
                return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
            }
            push(StorageIdentity::CapturePlan(identity.clone()), *bytes)?;
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests;

impl eredu_runtime::working_memory::CapturePlanStorageKey for StorageIdentity {
    fn capture_plan_identity(&self) -> Option<&eredu_core::SharedStorageIdentity> {
        match self {
            Self::CapturePlan(identity) => Some(identity),
            _ => None,
        }
    }
}

impl eredu_runtime::working_memory::GgufSourceStorageKey for StorageIdentity {
    fn gguf_source_identity(&self) -> Option<&eredu_checkpoint::store::SourceStorageIdentity> {
        match self {
            Self::Source(identity) => Some(identity),
            _ => None,
        }
    }
}

/// Shared producer quotation for generic descriptors, publication attachments,
/// and their retained native owner. Original prepared publication has its own layout.
pub(crate) fn generic_storage_publication_layout(
    maximum: usize,
) -> Result<
    eredu_runtime::working_memory::StoragePublicationLayout<StorageIdentity>,
    WorkingMemoryError,
> {
    use std::{alloc::Layout, mem::size_of};
    let descriptors = Layout::array::<(StorageIdentity, u64)>(maximum)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size()
        .checked_add(
            Layout::array::<(StorageIdentity, StorageAllocation)>(maximum)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
        )
        .ok_or(WorkingMemoryError::Overflow)?;
    let registered = Layout::new::<[usize; 2]>()
        .extend(Layout::new::<OrdinaryRetirement<RegisteredStorage>>())
        .map_err(|_| WorkingMemoryError::Overflow)?
        .0
        .pad_to_align()
        .size();
    let extra = u64::try_from(descriptors)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .checked_add(
            publication::publication_control_bytes(maximum)
                .ok_or(WorkingMemoryError::UnknownBound)?,
        )
        .and_then(|n| {
            n.checked_add(publication::generic_native_attachment_control_bytes(
                maximum,
            )?)
        })
        .and_then(|n| n.checked_add(registered as u64))
        .and_then(|n| n.checked_add(size_of::<RegisteredStorage>() as u64))
        .and_then(|n| n.checked_add(OrdinaryRetirement::<RegisteredStorage>::control_bytes()?))
        .and_then(|n| n.checked_add(UnquotedOriginalSlotSources::constructor_bytes(maximum).ok()?))
        .ok_or(WorkingMemoryError::Overflow)?;
    eredu_runtime::working_memory::StoragePublicationLayout::new(maximum)?
        .with_additional_host_metadata(extra)
}
pub(crate) fn prepare_storage_publication(
    pool: &MemoryLedger,
    maximum: usize,
) -> Result<
    eredu_runtime::working_memory::PreparedStoragePublication<StorageIdentity>,
    WorkingMemoryError,
> {
    generic_storage_publication_layout(maximum)?.fund(pool)
}
pub(crate) fn prepare_funded_storage_publication(
    scope: &eredu_runtime::working_memory::WorkingMemoryFundingScope,
    maximum: usize,
) -> Result<
    eredu_runtime::working_memory::PreparedStoragePublication<StorageIdentity>,
    WorkingMemoryError,
> {
    generic_storage_publication_layout(maximum)?.fund_from(scope)
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
