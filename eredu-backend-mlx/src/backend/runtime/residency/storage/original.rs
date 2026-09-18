//! Fixed destination for the same retained-storage inclusion workers.
use super::*;
use eredu_runtime::working_memory::{
    CollectorCapacityKind, OriginalTextMetadataCustody, WorkingMemoryError,
};
use std::{alloc::Layout, mem::size_of};

mod custody;
use custody::Custody;

pub(super) struct Census {
    pub(super) rows: usize,
    pub(super) pool: eredu_runtime::working_memory::WorkingMemoryPool,
}
impl Census {
    pub(super) fn add(&mut self) -> Result<(), ResidencyError> {
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or_else(|| failure(WorkingMemoryError::Overflow))?;
        Ok(())
    }
}

pub(super) enum Value {
    Array((u64, RetainedArray)),
    GroupBuffer((u64,safemlx::distributed::RetainedGroupBuffer)),
    Host((u64, RetainedHostBuffer)),
    Bytes(Arc<[u8]>),
    Metadata((u64, eredu_runtime::SharedHostMetadata)),
    Slot((u64, eredu_runtime::HostSlotMetadata)),
    Capture((u64, eredu_core::capture::SharedCapturePlan)),
    Source(eredu_checkpoint::store::SourceStorageOwner),
    UnknownArray(Array),
    UnknownMetadata(eredu_runtime::SharedHostMetadata),
    UnknownSlot(eredu_runtime::HostSlotMetadata),
    UnknownCapture(eredu_core::capture::SharedCapturePlan),
}
impl Value {
    fn kind(&self) -> u8 {
        match self {
            Self::Array(_) => 0,
            Self::GroupBuffer(_) => 11,
            Self::Host(_) => 1,
            Self::Bytes(_) => 2,
            Self::Metadata(_) => 3,
            Self::Slot(_) => 4,
            Self::Capture(_) => 5,
            Self::Source(_) => 6,
            Self::UnknownArray(_) => 7,
            Self::UnknownMetadata(_) => 8,
            Self::UnknownSlot(_) => 9,
            Self::UnknownCapture(_) => 10,
        }
    }
    fn bytes(&self) -> Option<u64> {
        match self {
            Self::Array((n, _))
            | Self::GroupBuffer((n,_))
            | Self::Host((n, _))
            | Self::Metadata((n, _))
            | Self::Slot((n, _))
            | Self::Capture((n, _)) => Some(*n),
            Self::Bytes(v) => u64::try_from(v.len()).ok(),
            Self::Source(v) => Some(v.bytes()),
            _ => None,
        }
    }
}
struct Row {
    key: Option<StorageIdentity>,
    value: Value,
}
impl Row {
    fn compare(&self, kind: u8, key: &Option<StorageIdentity>) -> std::cmp::Ordering {
        self.value.kind().cmp(&kind).then_with(|| self.key.cmp(key))
    }
}

pub(super) struct Inventory {
    rows: Vec<Row>,
    clones: Vec<safemlx::PreparedArrayClone>,
    refused: Option<Value>,
    #[cfg(test)]
    clone_attempts: usize,
    // Last: even an escaped inventory cannot retire its allocation after Q.
    custody: Custody,
}
impl Inventory {
    pub(super) fn new(rows: usize, custody: Custody) -> Result<Self, ResidencyError> {
        if !qualified() {
            return Err(failure(WorkingMemoryError::UnknownBound));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(rows)
            .map_err(|e| failure(WorkingMemoryError::ControlStorageReserve(e)))?;
        if values.capacity() != rows {
            return Err(failure(WorkingMemoryError::IdentityMismatch));
        }
        let mut result = Self {
            rows: values,
            clones: Vec::new(),
            refused: None,
            #[cfg(test)]
            clone_attempts: 0,
            custody,
        };
        result
            .clones
            .try_reserve_exact(rows)
            .map_err(|e| failure(WorkingMemoryError::ControlStorageReserve(e)))?;
        if result.clones.capacity() != rows {
            return Err(failure(WorkingMemoryError::IdentityMismatch));
        }
        for _ in 0..rows {
            result.clones.push(
                safemlx::PreparedArrayClone::try_prepare_for_inspection()
                    .map_err(ResidencyError::OriginalClone)?,
            );
        }
        Ok(result)
    }
    pub(super) fn clone_array(&mut self, source: &Array) -> Result<Array, ResidencyError> {
        #[cfg(test)]
        {
            self.clone_attempts += 1;
        }
        let capacity = self.clones.capacity();
        let used = capacity - self.clones.len();
        let slot = self.clones.last_mut().ok_or_else(|| {
            failure(WorkingMemoryError::CollectorCapacity {
                kind: CollectorCapacityKind::InventoryCloneShells,
                used,
                capacity,
            })
        })?;
        let array = slot
            .fill_for_inspection(source)
            .map_err(ResidencyError::OriginalClone)?;
        self.clones.pop();
        Ok(array)
    }
    pub(super) fn maximum_rows(&self) -> usize {
        self.rows.capacity()
    }
    pub(super) fn values(&self) -> impl Iterator<Item = (&Option<StorageIdentity>, &Value)> {
        self.rows.iter().map(|r| (&r.key, &r.value))
    }
    pub(super) fn get(&self, kind: u8, key: &StorageIdentity) -> Option<&Value> {
        self.rows
            .binary_search_by(|r| {
                r.value
                    .kind()
                    .cmp(&kind)
                    .then_with(|| r.key.as_ref().cmp(&Some(key)))
            })
            .ok()
            .map(|i| &self.rows[i].value)
    }
    pub(super) fn get_mut(&mut self, kind: u8, key: &StorageIdentity) -> Option<&mut Value> {
        self.rows.binary_search_by(|r| r.value.kind().cmp(&kind).then_with(|| r.key.as_ref().cmp(&Some(key))))
            .ok().map(|i| &mut self.rows[i].value)
    }
    pub(super) fn custody(&self) -> &Custody {
        &self.custody
    }
    pub(super) fn refuse(&mut self, value: Value) -> ResidencyError {
        if self.refused.is_none() {
            self.refused = Some(value);
        }
        failure(WorkingMemoryError::UnknownBound)
    }
    pub(super) fn ensure_room(&self) -> Result<(), ResidencyError> {
        if self.refused.is_some() {
            return Err(failure(WorkingMemoryError::ExecutionFenced));
        }
        if self.rows.len() == self.rows.capacity() {
            Err(self.row_capacity_failure())
        } else {
            Ok(())
        }
    }
    fn row_capacity_failure(&self) -> ResidencyError {
        failure(WorkingMemoryError::CollectorCapacity {
            kind: CollectorCapacityKind::InventoryRows,
            used: self.rows.len(),
            capacity: self.rows.capacity(),
        })
    }
    pub(super) fn insert(
        &mut self,
        key: Option<StorageIdentity>,
        value: Value,
    ) -> Result<(), ResidencyError> {
        if self.refused.is_some() {
            return Err(failure(WorkingMemoryError::ExecutionFenced));
        }
        let position = self
            .rows
            .binary_search_by(|r| r.compare(value.kind(), &key));
        if let Ok(i) = position {
            if key.is_some() {
                if self.rows[i].value.bytes() != value.bytes() {
                    return Err(failure(WorkingMemoryError::IdentityMismatch));
                }
                return Ok(());
            }
        }
        if self.rows.len() == self.rows.capacity() {
            self.refused = Some(value);
            return Err(self.row_capacity_failure());
        }
        self.rows
            .insert(position.unwrap_or_else(|i| i), Row { key, value });
        Ok(())
    }
    pub(super) fn byte_bound(&self) -> Result<Option<u64>, ResidencyError> {
        if self.refused.is_some() {
            return Ok(None);
        }
        let mut total = 0u64;
        for row in &self.rows {
            let Some(bytes) = row.value.bytes() else {
                return Ok(None);
            };
            if matches!(&row.value, Value::Host(_))
                && row.key.as_ref().is_some_and(|k| self.get(0, k).is_some())
            {
                continue;
            }
            total = total
                .checked_add(bytes)
                .ok_or_else(|| failure(WorkingMemoryError::Overflow))?;
        }
        Ok(Some(total))
    }
    pub(super) fn finish(&mut self) {
        self.rows.retain(|row| matches!(&row.value, Value::Bytes(_) | Value::Source(_) | Value::GroupBuffer(_)) ||
            matches!(&row.value, Value::Metadata((_, eredu_runtime::SharedHostMetadata::Input(v))) if v.original_source().is_some()));
    }
    pub(super) fn controls(rows: usize) -> Option<usize> {
        if !qualified() {
            return None;
        }
        let keys = usize::try_from(eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes()?)
            .ok()?
            .checked_mul(rows)?;
        let clones = Layout::array::<safemlx::PreparedArrayClone>(rows)
            .ok()?
            .size()
            .checked_add(rows.checked_mul(Array::inspection_clone_handle_bytes())?)?
            // Preparation/fill/drop run serially over these slots. Their
            // frames are reused; the slot vector and native handles above
            // remain charged for every row, including failed prefixes.
            .checked_add(safemlx::PreparedArrayClone::control_bytes()?)?;
        Layout::array::<Row>(rows)
            .ok()?
            .size()
            .checked_add(clones)?
            .checked_add(RetainedArray::control_bytes()?)?
            .checked_add(keys)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<Row>())?
            .checked_add(safemlx::distributed::RetainedGroupBuffer::clone_control_bytes())?
            .checked_add(size_of::<Option<&safemlx::distributed::RetainedGroupBuffer>>())?
            .checked_add(size_of::<safemlx::distributed::GroupBufferIdentity>())?
            .checked_add(size_of::<Result<Self, ResidencyError>>())?
            .checked_add(size_of::<Result<(), ResidencyError>>())?
            .checked_add(size_of::<WorkingMemoryError>())?
            .checked_add(size_of::<CollectorCapacityKind>())?
            .checked_add(size_of::<usize>().checked_mul(2)?)?
            .checked_add(size_of::<std::collections::TryReserveError>())?
            .checked_add(size_of::<std::slice::Iter<'static, Row>>())?
            .checked_add(size_of::<Custody>())?
            .checked_add(size_of::<StorageIdentity>())?
            .checked_add(eredu_runtime::working_memory::WorkingMemoryPool::retained_source_inventory_control_bytes::<StorageIdentity>()?)?
            .checked_add(size_of::<
                Result<OriginalTextMetadataCustody, WorkingMemoryError>,
            >())?
            .checked_add(size_of::<
                Result<Option<OriginalTextMetadataCustody>, WorkingMemoryError>,
            >())?
            .checked_add(size_of::<(&Custody, &eredu_runtime::SharedHostMetadata)>())?
            .checked_add(size_of::<(&Custody, &eredu_runtime::HostSlotMetadata)>())?
            .checked_add(size_of::<(
                &Custody,
                &eredu_checkpoint::store::SourceStorageIdentity,
                u64,
            )>())
    }
}
pub(super) fn qualified() -> bool {
    // The existing managed-owner query is backed by the pinned fresh Global
    // Vec/Arc/Rc qualification; it neither constructs nor issues a budget.
    native_storage::Bank::shared_borrowed_owner_bytes().is_some()
}
pub(super) fn failure(cause: WorkingMemoryError) -> ResidencyError {
    ResidencyError::OriginalInventory(cause)
}

impl RetainedStorage {
    /// The private selected Work supplies the already accepted population and
    /// raw custody. No ordinary inventory is promoted after its construction.
    pub(crate) fn original_census(pool: &eredu_runtime::working_memory::WorkingMemoryPool) -> Self {
        Self {
            census: Some(Census {
                rows: 0,
                pool: pool.clone(),
            }),
            ..Self::default()
        }
    }
    pub(crate) fn prepare_original(
        rows: usize,
        custody: OriginalTextMetadataCustody,
    ) -> Result<Self, ResidencyError> {
        Ok(Self {
            original: Some(Inventory::new(rows, Custody::Text(custody))?),
            ..Self::default()
        })
    }
    /// The caller has admitted the owning collector amount under this host
    /// authority. Source validation remains tied to the supplied source pool;
    /// this custody can never be projected into a native publication role.
    pub(crate) fn prepare_snapshot_collector(
        rows: usize,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, ResidencyError> {
        Ok(Self {
            original: Some(Inventory::new(
                rows,
                Custody::Snapshot {
                    pool: pool.clone(),
                    _host: host.clone(),
                },
            )?),
            ..Self::default()
        })
    }

    pub(crate) fn prepare_snapshot_publication(
        rows: usize,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, ResidencyError> {
        Ok(Self {
            original: Some(Inventory::new(
                rows,
                Custody::SnapshotPublication {
                    pool: pool.clone(),
                    _host: host.clone(),
                },
            )?),
            ..Self::default()
        })
    }
    pub(super) fn validate_snapshot_publication(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.original
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .custody
            .snapshot_publication(pool)
    }

    pub(super) fn publication_custody(
        &self,
    ) -> Result<Option<OriginalTextMetadataCustody>, WorkingMemoryError> {
        self.original
            .as_ref()
            .map(|value| value.custody.publication())
            .transpose()
    }

    pub(crate) fn original_collector_control_bytes(rows: usize) -> Option<usize> {
        Inventory::controls(rows)
    }
    #[cfg(test)]
    pub(crate) fn original_clone_progress(&self) -> Option<(usize, usize, usize)> {
        self.original
            .as_ref()
            .map(|value| (value.rows.len(), value.clones.len(), value.clone_attempts))
    }
    pub(crate) fn original_maximum_rows(&self) -> Option<usize> {
        self.original.as_ref().map(Inventory::maximum_rows)
    }
    pub(crate) fn is_original_collector(&self) -> bool {
        self.original.is_some()
    }
    pub(crate) fn requires_borrowed_source_storage(&self) -> bool {
        self.original.is_some() || self.census.is_some()
    }
    pub(crate) fn include_checkpoint_source(
        &mut self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<(), ResidencyError> {
        if !self.requires_borrowed_source_storage() {
            return self.include_sources(source.source_storage()?);
        }
        let mut failure = None;
        let result = source.visit_source_storage(&mut |value| {
            if failure.is_none() {
                failure = self.include_source_ref(value).err();
            }
        });
        if let Some(cause) = failure {
            return Err(cause);
        }
        if !result? {
            self.mark_incomplete();
        }
        Ok(())
    }
    pub(crate) fn include_source_ref(
        &mut self,
        source: eredu_checkpoint::store::SourceStorageRef<'_>,
    ) -> Result<(), ResidencyError> {
        let identity = source.identity();
        let key = StorageIdentity::Source(identity);
        if let Some(census) = &mut self.census {
            match census
                .pool
                .validate_retained_source_inventory(&key, source.bytes())
            {
                Ok(()) => census.add()?,
                Err(WorkingMemoryError::UnknownBound) => self.incomplete = true,
                Err(cause) => return Err(failure(cause)),
            }
            return Ok(());
        }
        if let Some(original) = &mut self.original {
            original
                .custody
                .validate_source_inventory(&key, source.bytes())
                .map_err(|cause| failure(cause))?;
            return original.insert(
                Some(key),
                Value::Source(source.retain()),
            );
        }
        self.sources.include_ref(source)?;
        Ok(())
    }

    pub(super) fn group_buffer_entries(&self)->impl Iterator<Item=(&safemlx::distributed::GroupBufferIdentity,&(u64,safemlx::distributed::RetainedGroupBuffer))>{
        self.group_buffers.iter().chain(self.original.iter().flat_map(|source|source.values()).filter_map(|(key,value)|match(key,value){
            (Some(StorageIdentity::GroupBuffer(key)),Value::GroupBuffer(value))=>Some((key,value)),_=>None,
        }))
    }
    pub(super) fn array_entries(
        &self,
    ) -> impl Iterator<Item = (&safemlx::AllocationIdentity, &(u64, RetainedArray))> {
        self.arrays
            .iter()
            .filter_map(|(key, value)| value.owned().map(|value| (key, value)))
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(k, v)| match (k, v) {
                        (Some(StorageIdentity::Native(k)), Value::Array(v)) => Some((k, v)),
                        _ => None,
                    }),
            )
    }
    pub(super) fn host_entries(
        &self,
    ) -> impl Iterator<Item = (&safemlx::AllocationIdentity, &(u64, RetainedHostBuffer))> {
        self.hosts
            .iter()
            .filter_map(|(key, entry)| entry.owned().map(|value| (key, value)))
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(k, v)| match (k, v) {
                        (Some(StorageIdentity::Native(k)), Value::Host(v)) => Some((k, v)),
                        _ => None,
                    }),
            )
    }
    pub(super) fn array_entry(&self, id: &safemlx::AllocationIdentity) -> Option<&(u64, RetainedArray)> {
        self.arrays.get(id).and_then(NativeEntry::owned).or_else(|| {
            match self
                .original
                .as_ref()?
                .get(0, &StorageIdentity::Native(*id))?
            {
                Value::Array(v) => Some(v),
                _ => None,
            }
        })
    }
    pub(super) fn host_entry(
        &self,
        id: &safemlx::AllocationIdentity,
    ) -> Option<&(u64, RetainedHostBuffer)> {
        self.hosts.get(id).and_then(NativeEntry::owned).or_else(|| {
            match self
                .original
                .as_ref()?
                .get(1, &StorageIdentity::Native(*id))?
            {
                Value::Host(v) => Some(v),
                _ => None,
            }
        })
    }
    pub(super) fn metadata_entries(
        &self,
    ) -> impl Iterator<
        Item = (
            &eredu_runtime::HostMetadataIdentity,
            &(u64, eredu_runtime::SharedHostMetadata),
        ),
    > {
        self.metadata
            .iter()
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(_, v)| match v {
                        Value::Metadata(v) => Some((v.1.identity(), v)),
                        _ => None,
                    }),
            )
    }
    pub(super) fn slot_entries(
        &self,
    ) -> impl Iterator<
        Item = (
            &eredu_runtime::HostMetadataIdentity,
            &(u64, eredu_runtime::HostSlotMetadata),
        ),
    > {
        self.slot_metadata
            .iter()
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(_, v)| match v {
                        Value::Slot(v) => Some((v.1.identity(), v)),
                        _ => None,
                    }),
            )
    }
    pub(super) fn capture_entries(
        &self,
    ) -> impl Iterator<
        Item = (
            &eredu_core::SharedStorageIdentity,
            &(u64, eredu_core::capture::SharedCapturePlan),
        ),
    > {
        self.capture_plans
            .iter()
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(_, v)| match v {
                        Value::Capture(v) => Some((v.1.storage_identity(), v)),
                        _ => None,
                    }),
            )
    }
    pub(super) fn byte_values(&self) -> impl Iterator<Item = &Arc<[u8]>> {
        self.byte_buffers
            .values()
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(_, v)| match v {
                        Value::Bytes(v) => Some(v),
                        _ => None,
                    }),
            )
    }
    pub(super) fn source_capacities(
        &self,
    ) -> impl Iterator<Item = (eredu_checkpoint::store::SourceStorageIdentity, u64)> + '_ {
        self.sources
            .capacities()
            .chain(
                self.original
                    .iter()
                    .flat_map(|s| s.values())
                    .filter_map(|(_, v)| match v {
                        Value::Source(v) => Some((v.identity(), v.bytes())),
                        _ => None,
                    }),
            )
    }
}
