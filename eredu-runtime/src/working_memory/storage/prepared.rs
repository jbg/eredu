//! Caller-funded bounded preparation for ordinary storage publication.
use super::*;
use crate::working_memory::{StorageMetadataFunding, qualified_storage};
use eredu_core::{HostMetadataFunding, HostPreparationAuthority};

/// Managed host layout for one bounded inventory publication or pin.
/// Constructing the layout allocates nothing and grants no execution authority.
#[derive(Debug)]
pub struct StoragePublicationLayout<K: Ord + Send + 'static> {
    maximum: usize,
    bytes: u64,
    pending_allocation: bool,
    marker: std::marker::PhantomData<fn() -> K>,
}
impl<K: Ord + Send + 'static> StoragePublicationLayout<K> {
    /// Quotes the bounded vectors, canonical rows, namespace and independent
    /// registration owners before their constructor can allocate.
    pub fn new(maximum_backings: usize) -> Result<Self, WorkingMemoryError> {
        let individual = StorageRegistrations::<K>::publication_control_bytes(
            maximum_backings,
            maximum_backings,
            maximum_backings,
        )?;
        let grouped = cold_metadata::registration_bytes::<K>(maximum_backings)?;
        let pin = u64::try_from(finite_pin::construction_bytes::<K>(maximum_backings)?)
            .map_err(|_| WorkingMemoryError::Overflow)?
            .checked_add(cold_metadata::vector_bytes::<(K, StorageAllocation)>(
                maximum_backings,
            )?)
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<PreparedStoragePublication<K>>()))
            .and_then(|n| n.checked_add(HostMetadataFunding::reservation_control_bytes()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = individual
            .checked_add(grouped)
            .ok_or(WorkingMemoryError::Overflow)?
            .max(pin)
            .checked_add(controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            maximum: maximum_backings,
            bytes,
            pending_allocation: false,
            marker: std::marker::PhantomData,
        })
    }
    /// Quotes the additional preallocated conversion rows for native backing
    /// that is reserved now and published only after successful allocation.
    pub fn pending(maximum_backings: usize) -> Result<Self, WorkingMemoryError> {
        let mut layout = Self::new(maximum_backings)?;
        layout.bytes = layout
            .bytes
            .checked_add(pending_allocation::control_bytes(maximum_backings)?)
            .and_then(|n| n.checked_add(std::mem::size_of::<PendingStorageAllocation<K>>() as u64))
            .ok_or(WorkingMemoryError::Overflow)?;
        layout.pending_allocation = true;
        Ok(layout)
    }
    /// Includes caller-owned host attachment controls in this same prepared owner.
    pub fn with_additional_host_metadata(mut self, bytes: u64) -> Result<Self, WorkingMemoryError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(self)
    }

    /// Complete preparation allowance; allocation begins only after it is funded.
    pub fn requested_bytes(&self) -> u64 {
        self.bytes
    }
    /// Starts a distinct host preparation operation before physical publication.
    pub fn fund(
        self,
        pool: &MemoryLedger,
    ) -> Result<PreparedStoragePublication<K>, WorkingMemoryError> {
        let funding = pool
            .prepare_storage_metadata()
            .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        self.prepare(pool, &funding)
    }
    /// Funds preparation from an admitted account's assigned host allowance.
    pub fn fund_from(
        self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<PreparedStoragePublication<K>, WorkingMemoryError> {
        let funding = scope
            .prepare_storage_metadata()
            .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        self.prepare(scope.pool(), &funding)
    }
    /// Reserves this one constructor on an authenticated ledger metadata owner.
    pub fn prepare(
        self,
        pool: &MemoryLedger,
        funding: &StorageMetadataFunding,
    ) -> Result<PreparedStoragePublication<K>, WorkingMemoryError> {
        funding.validate_ledger(pool)?;
        funding
            .funding()
            .reserve_metadata(
                usize::try_from(self.bytes).map_err(|_| WorkingMemoryError::Overflow)?,
            )
            .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        let host = HostPreparationAuthority::retain(funding.funding().clone());
        Ok(PreparedStoragePublication {
            pool: pool.clone(),
            maximum: self.maximum,
            pending_allocation: self.pending_allocation,
            host,
            marker: std::marker::PhantomData,
        })
    }
}
/// Move-only prepared constructor. Publication updates every payload domain
/// atomically. The preparation charge follows its retained metadata owner.
#[derive(Debug)]
pub struct PreparedStoragePublication<K: Ord + Send + 'static> {
    pool: MemoryLedger,
    maximum: usize,
    pending_allocation: bool,
    host: HostPreparationAuthority,
    marker: std::marker::PhantomData<fn() -> K>,
}
impl<K: Ord + Send + 'static> PreparedStoragePublication<K> {
    /// Retains the paid preparation envelope for caller-owned attachment controls.
    pub fn host_authority(&self) -> &HostPreparationAuthority {
        &self.host
    }
    fn collect<T>(
        &self,
        values: impl IntoIterator<Item = T>,
    ) -> Result<Vec<T>, WorkingMemoryError> {
        let mut result = qualified_storage::vector(self.maximum, true)?;
        for value in values {
            if result.len() == self.maximum {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            result.push(value);
        }
        Ok(result)
    }
    /// Pins existing authenticated capacities. No backing receives a new charge.
    pub fn pin_registered_storage(
        self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let inputs = self.collect(storage)?;
        let mut result = self.pool.pin_registered_storage_owned(inputs, true)?;
        Arc::get_mut(&mut result.0)
            .expect("unpublished pin")
            .preparation = Some(self.host.clone());
        Ok(result)
    }
    /// Pins existing complete capacity and placement descriptors atomically.
    pub fn pin_registered_storage_with_placement(
        self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let inputs = self.collect(storage)?;
        let mut result = self
            .pool
            .pin_registered_storage_placed_owned(inputs, true)?;
        Arc::get_mut(&mut result.0)
            .expect("unpublished pin")
            .preparation = Some(self.host.clone());
        Ok(result)
    }
}
impl<K: Clone + Ord + Send + 'static> PreparedStoragePublication<K> {
    /// Atomically reserves future backing capacity in every placement domain.
    /// Pending identities cannot be pinned or used as existing storage. The
    /// producer calls `publish` only after native allocation establishes backing.
    pub fn reserve_storage(
        self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<PendingStorageAllocation<K>, WorkingMemoryError> {
        if !self.pending_allocation {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let unique = physical::prepare(&self.pool, self.collect(storage)?)?;
        pending_allocation::reserve(&self.pool, unique, &self.host, None)
    }

    /// Converts this exact live scope's assigned allowance into pending backing.
    /// No second physical reservation occurs. Publication metadata must already
    /// be paid; native allocation failure returns capacity to the same account.
    pub fn reserve_storage_from(
        self,
        funding: &super::super::WorkingMemoryAllocationFunding,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<PendingStorageAllocation<K>, WorkingMemoryError> {
        if !self.pending_allocation {
            return Err(WorkingMemoryError::UnknownBound);
        }
        if !self.pool.same_ledger(funding.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let unique = physical::prepare(&self.pool, self.collect(storage)?)?;
        pending_allocation::reserve(&self.pool, unique, &self.host, Some(funding))
    }

    /// Publishes one retained inventory through the ledger's physical transaction.
    pub fn register_storage(
        self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let inputs = self.collect(storage)?;
        self.pool.register_storage_prepared(inputs, &self.host)
    }
    /// Publishes independent owners while retaining the prepared host constructor.
    pub fn register_storage_individually(
        self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let inputs = self.collect(storage)?;
        self.pool
            .publish_physical_individually(inputs, None, &self.host)
    }
    /// Converts an admitted payload allowance with separately funded host controls.
    pub fn adopt_storage_individually(
        self,
        scope: &WorkingMemoryFundingScope,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        scope.validate_ledger(&self.pool)?;
        scope.validate_native_purpose()?;
        let inputs = self.collect(storage)?;
        self.pool
            .publish_physical_individually(inputs, Some(scope), &self.host)
    }
    /// Publishes host backings with placement explicitly supplied by this ledger.
    pub fn register_host_storage(
        self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let placement = self.pool.host_placement_handle();
        self.register_storage(
            storage
                .into_iter()
                .map(|(key, bytes)| (key, StorageAllocation::new(bytes, Arc::clone(&placement)))),
        )
    }
}

impl<
    K: Clone + Ord + Send + Sync + 'static + crate::working_memory::gguf_source::GgufSourceStorageKey,
> PreparedStoragePublication<K>
{
    /// Publishes a checkpoint inventory while retaining authenticated prepaid sources.
    pub fn register_storage_with_gguf_sources(
        self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let inputs = self.collect(storage)?;
        self.pool
            .register_storage_with_gguf_sources_prepared(inputs, &self.host)
    }
}

#[cfg(test)]
mod tests;
