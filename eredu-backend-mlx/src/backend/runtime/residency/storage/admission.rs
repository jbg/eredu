//! Native inventory ownership alongside neutral shared-memory accounting.

use super::*;
use crate::backend::error::Error;
use crate::backend::ordinary_retirement::OrdinaryRetirement;
use eredu_runtime::working_memory::{
    UnquotedOriginalSlotSources, WorkingMemoryError, WorkingMemoryPool, WorkingMemoryStorage,
};

mod publication;
mod source_pin;
pub(crate) use source_pin::SourcePinPlan;
pub(crate) use publication::{
    PendingCopyPublication, CopyPublicationLayout, PendingNativePublication,
    RetainedStoragePublication, retain_copy_publication_failure,
};

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
            Self::Native(_) | Self::GroupBuffer(_) | Self::Source(_) | Self::Bytes(_) | Self::CapturePlan(_) => None,
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
    pub fn bytes(&self) -> u64 {
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
        pool: &WorkingMemoryPool,
    ) -> Result<Option<usize>, Error> {
        if let Some(census) = &self.census {
            return Ok((!self.incomplete).then_some(census.rows));
        }
        if self.byte_bound()?.is_none()
            || !self.byte_buffers.is_empty()
            || !self.capture_plans.is_empty()
        {
            return Ok(None);
        }
        for (identity, bytes) in self.source_capacities() {
            match pool.validate_retained_source_inventory(&StorageIdentity::Source(identity), bytes) {
                Ok(()) => {}
                Err(WorkingMemoryError::UnknownBound) => return Ok(None),
                Err(cause) => return Err(Error::PrefillControl(cause)),
            }
        }
        // Counting both sides of a captured alias is conservative and uses
        // actual rows. Registry commit still deduplicates physical identities.
        [
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
        pool: &WorkingMemoryPool,
        expected: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
    ) -> Result<(), Error> {
        self.validate_original_sources(pool, expected, None)
    }

    /// The supplied B profile comes from the actual accepted quote's source
    /// projection. Every original cache must match that materialization and pool;
    /// authenticating residence alone never permits an unrelated original source.
    pub(crate) fn validate_original_sources(
        &self,
        pool: &WorkingMemoryPool,
        expected: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    ) -> Result<(), Error> {
        if prepared.is_some_and(|source| !source.pool().same_domain(pool)) {
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
        self.byte_bound()?
            .ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?;
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
        pool: &WorkingMemoryPool,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, Error> {
        let (entries, mut original) = self.source_entries(pool, None)?;
        pool.pin_registered_storage(entries)
            .and_then(|storage| storage.with_original_reset_sources(&mut original))
            .map_err(|error| original_source_failure(Error::Other(Box::new(error)), original))
    }

    /// Existing-only source pinning under an already accepted enclosing host
    /// plan. Original tables and prepared inputs retain their real constructor
    /// custody; they never become ordinary registrations or new capacity credit.
    pub(crate) fn pin_registered_with_host(
        &self,
        pool: &WorkingMemoryPool,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, Error> {
        self.source_pin_plan(pool)?.pin(self, pool, host)
    }

    /// Registers certified existing payloads atomically against the same domain
    /// that admits requests. Separate model/source inventories share charges for
    /// identical backing, including host/device aliases on unified memory.
    /// Unknown storage is rejected before accounting changes or native work.
    pub fn register(self, pool: &WorkingMemoryPool) -> Result<RetainedStorageReservation, Error> {
        let (entries, mut original) = self.source_entries(pool, None)?;
        let charge = pool
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
        pool: &WorkingMemoryPool,
        owner: Option<&crate::backend::managed_memory::NativeMemoryOwner>,
    ) -> Result<(Vec<(StorageIdentity, u64)>, UnquotedOriginalSlotSources), Error> {
        self.byte_bound()?
            .ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?;
        // Every originally constructed cache remains an explicit inventory
        // owner. Ordinary publication can consume its authenticated residence;
        // pool-only/funded callers need D's separate source profile first.
        for (_, (_, metadata)) in self.metadata_entries() {
            if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
                if let Some(result) = input.original_residence(pool) {
                    result.map_err(|error| Error::Other(Box::new(error)))?;
                    let owner = owner
                        .ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?;
                    if !owner.pool().same_domain(pool) {
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
            if !owner.pool().same_domain(pool) {
                return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
            }
            UnquotedOriginalSlotSources::prepare(&owner.unquoted_lease()?)
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
        let mut entries = match self.storage_entries() {
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

    fn storage_entries(&self) -> Result<Vec<(StorageIdentity, u64)>, Error> {
        self.storage_entries_with_capacity(self.original_maximum_rows())
    }

    fn storage_entries_with_capacity(
        &self,
        maximum: Option<usize>,
    ) -> Result<Vec<(StorageIdentity, u64)>, Error> {
        self.byte_bound()?
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
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
        for (identity,(bytes,_)) in self.group_buffer_entries(){push(StorageIdentity::GroupBuffer(*identity),*bytes)?;}
        for (identity, (bytes, _)) in self.array_entries() {
            push(StorageIdentity::Native(*identity), *bytes)?;
        }
        for (identity, (bytes, _)) in self.host_entries() {
            if self.array_entry(identity).is_none() {
                push(StorageIdentity::Native(*identity), *bytes)?;
            }
        }
        for (identity, bytes) in self.source_capacities() {
            push(StorageIdentity::Source(identity), bytes)?;
        }
        for bytes in self.byte_values() {
            if fixed {
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
            if fixed {
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
