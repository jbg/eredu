//! Closed Host-table allocation and publication through the physical registry.
use super::{PendingStorageAllocation, StorageAllocation, StoragePublicationLayout};
use crate::working_memory::{
    HostSlotStorageKey, MemoryLedger, StorageMetadataFunding, WorkingMemoryError, qualified_storage,
};
use crate::{HostMetadataIdentity, HostSlotMetadata, HostSlotTable};
use eredu_core::HostPreparationAuthority;
use std::mem::{size_of, size_of_val};

/// One exact Host-table destination. Its original physical reservation precedes
/// allocation and remains live until the table backing retires. Nested element
/// resources require their own owners; this covers only the inline slot extent.
#[derive(Debug)]
pub struct PreparedStorageHostSlots<T, K: HostSlotStorageKey> {
    // Partial values and their buffer retire before the reservation and payer.
    values: Vec<T>,
    identity: HostMetadataIdentity,
    pending: PendingStorageAllocation<K>,
    host: HostPreparationAuthority,
    pool: MemoryLedger,
    maximum: usize,
}

impl StorageMetadataFunding {
    /// Reserves the exact backing through the existing canonical registry before
    /// allocating its slots. Metadata consumes this source's real Host account;
    /// the payload is charged only by its independent physical reservation.
    pub fn prepare_host_slots<T, K: HostSlotStorageKey>(
        &self,
        len: usize,
    ) -> Result<PreparedStorageHostSlots<T, K>, WorkingMemoryError> {
        let backing = qualified_storage::array_bytes::<T>(len)?;
        let publication = StoragePublicationLayout::<K>::pending(1)?
            .with_additional_host_metadata(control_bytes::<T, K>()?)?
            .prepare(self.pool(), self)?;
        let host = publication.host_authority().clone();
        let identity = HostMetadataIdentity::prepared_host(&host)?;
        let key = K::from_host_slot_identity(identity.registry_key().clone())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if key.host_slot_identity() != Some(identity.registry_key()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pending = publication.reserve_storage([(
            key,
            StorageAllocation::new(backing, self.pool().host_placement_handle()),
        )])?;
        let values = qualified_storage::vector(len, true)?;
        Ok(PreparedStorageHostSlots {
            values,
            identity,
            pending,
            host,
            pool: self.pool().clone(),
            maximum: len,
        })
    }
}

fn control_bytes<T, K: HostSlotStorageKey>() -> Result<u64, WorkingMemoryError> {
    let parts = [
        HostSlotMetadata::prepared_host_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        HostSlotMetadata::copy_attachment_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        usize::try_from(qualified_storage::vector_control_bytes::<T>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?,
        size_of::<PreparedStorageHostSlots<T, K>>(),
        size_of::<Result<PreparedStorageHostSlots<T, K>, WorkingMemoryError>>(),
        size_of::<PublicationAttempt<T, K>>(),
        size_of::<PendingStorageAllocation<K>>(), // Box payload at handoff
        size_of::<Box<dyn Send + Sync>>(),
        size_of::<HostSlotTable<T>>(),
        size_of::<Box<[T]>>(),
        size_of::<(HostMetadataIdentity, HostPreparationAuthority, MemoryLedger)>(),
        size_of::<(&StorageMetadataFunding, usize)>(),
        size_of::<(&mut PreparedStorageHostSlots<T, K>, T)>(),
        size_of::<Result<HostSlotTable<T>, WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<bool, crate::HostSlotAttachmentError<WorkingMemoryError>>>(),
        size_of::<Result<Box<dyn Send + Sync>, WorkingMemoryError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}

// Failed publication destroys the complete table before its independent pending
// registration. Neither object can be destroyed inside the attachment locks.
struct PublicationAttempt<T, K: HostSlotStorageKey> {
    table: HostSlotTable<T>,
    pending: Option<Box<PendingStorageAllocation<K>>>,
}

impl<T, K: HostSlotStorageKey> PreparedStorageHostSlots<T, K> {
    /// Appends one existing value without growth. Refusal leaves the initialized
    /// prefix and its original reservation in this builder.
    pub fn push(&mut self, value: T) -> Result<(), WorkingMemoryError> {
        if self.values.len() == self.maximum {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.values.push(value);
        Ok(())
    }

    /// Publishes the actual complete extent and attaches its canonical owner.
    /// No payload charge is added by this reservation-to-registration handoff.
    pub fn finish(self) -> Result<HostSlotTable<T>, WorkingMemoryError> {
        if self.values.len() != self.maximum {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let Self {
            values,
            identity,
            pending,
            host,
            pool,
            maximum: _,
        } = self;
        let mut attempt = PublicationAttempt {
            table: HostSlotTable::new_prepared_host(values.into_boxed_slice(), identity, &host),
            pending: Some(Box::new(pending)),
        };
        attempt
            .table
            .metadata()
            .prepare_copy_attachment(pool.shared_storage_accounting_id())?;
        let pending = &mut attempt.pending;
        let attached = attempt.table.metadata().try_attach_prepared_copy(
            pool.shared_storage_accounting_id(),
            || {
                pending
                    .as_mut()
                    .expect("single table publication")
                    .publish()?;
                Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(
                    pending.take().expect("published table owner"),
                )
            },
        );
        match attached {
            Ok(true) => Ok(attempt.table),
            Ok(false) => Err(WorkingMemoryError::IdentityMismatch),
            Err(crate::HostSlotAttachmentError::Attachment(
                eredu_core::SharedStorageAttachmentError::Provider(cause),
            )) => Err(cause),
            Err(_) => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}

#[cfg(test)]
mod tests;
