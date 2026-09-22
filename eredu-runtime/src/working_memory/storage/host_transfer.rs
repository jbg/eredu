//! Exact completed-table transfer from an existing protected host hold.

use super::*;
use crate::working_memory::{
    HostSlotStorageKey, InferenceExecutionIdentity, funding::WorkingMemoryDecoderHostScope,
};
use crate::{HostSlotAttachmentError, InitializedDenseHostSlots};
use eredu_core::SharedStorageAttachmentError;

fn rejected(error: WorkingMemoryError) -> HostSlotAttachmentError<WorkingMemoryError> {
    HostSlotAttachmentError::Attachment(SharedStorageAttachmentError::Provider(error))
}

/// Only the closed, completed fresh decoder owner calls this. The actual table
/// remains borrowed and its full P hold stays live until a successful transfer.
/// No caller-supplied amount or metadata token can reach this private path.
pub(in crate::working_memory) fn publish_dense_host_slots<T, K: HostSlotStorageKey>(
    slots: &InitializedDenseHostSlots<T>,
    custody: &mut WorkingMemoryDecoderHostScope,
    execution: &InferenceExecutionIdentity,
    retained: u64,
    protected: u64,
    key: K,
    host: Option<&eredu_core::HostPreparationAuthority>,
) -> Result<(), HostSlotAttachmentError<WorkingMemoryError>> {
    let metadata = slots.metadata();
    if key.host_slot_identity() != Some(metadata.identity().registry_key())
        || metadata.capacity_bytes() != Some(retained)
        || metadata.slot_size() != std::mem::size_of::<T>()
        || metadata.len() != slots.len()
        || protected < retained
    {
        return Err(rejected(WorkingMemoryError::IdentityMismatch));
    }
    let pool = custody.pool().clone();
    // Ordinary destinations convert their admitted constructor allowance before
    // allocating registry staging, the attachment owner, or its map node.
    let ordinary = host.is_none();
    let dynamic_host = if ordinary {
        let funding = custody
            .prepare_storage_metadata(execution)
            .map_err(|e| rejected(metadata_error(e)))?;
        Some(
            funding
                .prepare_host_owner(ordinary_host_owner_bytes::<K>().map_err(rejected)?)
                .map_err(|e| rejected(metadata_error(e)))?,
        )
    } else {
        None
    };
    let staging_host = host.or(dynamic_host.as_ref());
    // Pending keys/handles belong to this outer staging frame, never the
    // provider closure. A rejected provider must release both token locks
    // before any of these destructors can reenter accounting or owner code.
    let prepared = Some(match staging_host {
        Some(host) => (
            RegistryBatch::prepare_copy_exact(1, host).map_err(rejected)?,
            directory::PreparedNamespace::prepare_copy::<K>(host),
        ),
        None => (
            RegistryBatch::prepare_source_registration(1),
            directory::PreparedNamespace::prepare::<K>(None),
        ),
    });
    let registration = if let Some(host) = staging_host {
        let mut keys =
            crate::working_memory::qualified_storage::vector(1, true).map_err(rejected)?;
        keys.push(key.clone());
        WorkingMemoryStorage::pending_prepared(keys, retained, host)
    } else {
        WorkingMemoryStorage::pending(vec![key.clone()], retained)
    };
    let mut transfer = PreparedTransfer {
        key: Arc::new(key),
        registration: Some(Box::new(registration)),
        prepared,
        unused_namespace: None,
        retained,
        protected,
    };
    let attached = if let Some(host) = dynamic_host.as_ref() {
        metadata.try_attach_owned_prepared(pool.shared_storage_accounting_id(), |layout| {
            if layout.requested_bytes() > ordinary_attachment_bytes::<K>()? {
                return Err(WorkingMemoryError::Overflow);
            }
            let registration = transfer.commit(&pool, custody, execution)?;
            Ok::<_, WorkingMemoryError>(eredu_core::SharedStorageOwner::new(DenseHostPublication {
                registration: Some(registration),
                _host: host.clone(),
            }))
        })?
    } else {
        metadata
            .prepare_copy_attachment(pool.shared_storage_accounting_id())
            .map_err(rejected)?;
        metadata.try_attach_prepared_copy(pool.shared_storage_accounting_id(), || {
            transfer
                .commit(&pool, custody, execution)
                .map(|r| r as Box<dyn Send + Sync>)
        })?
    };
    if !attached {
        // An arbitrary preexisting attachment is not this exact registration.
        // The closure did not run, so both actual and ledger holds remain P.
        return Err(rejected(WorkingMemoryError::IdentityMismatch));
    }
    Ok(())
}

struct PreparedTransfer<K: Ord + Send + Sync + 'static> {
    // Keep an owner outside try_attach, including while BTreeMap::entry calls
    // provider Ord. Its temporary key can then unwind under the locks without
    // running K::drop there. HostSlotStorageKey already requires Send + Sync;
    // the registry's ordinary Send-only key API does not gain a Sync bound.
    key: Arc<dyn Borrow<K> + Send + Sync>,
    registration: Option<Box<WorkingMemoryStorage<K>>>,
    prepared: Option<(Box<RegistryBatch<K>>, directory::PreparedNamespace)>,
    unused_namespace: Option<directory::PreparedNamespace>,
    retained: u64,
    protected: u64,
}

impl<K: HostSlotStorageKey> PreparedTransfer<K> {
    fn commit(
        &mut self,
        pool: &MemoryLedger,
        custody: &mut WorkingMemoryDecoderHostScope,
        execution: &InferenceExecutionIdentity,
    ) -> Result<Box<WorkingMemoryStorage<K>>, WorkingMemoryError> {
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let (id, held) = custody.validate_host_transfer(&usage, execution)?;
        if held != self.protected || self.retained > held {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Only a fresh physical destination can consume the protected hold.
        // Ordinary later inventory adoption may alias this established entry.
        let key: &K = self.key.as_ref().borrow();
        if usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|registry| {
                registry
                    .downcast_ref::<Registry<K>>()
                    .expect("typed storage registry")
                    .get(key)
            })
            .is_some()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Preserve the existing domain invariant. This operation shifts already
        // reserved coverage, including for a zero-byte table's origin lifetime.
        let _ = pool.0.check_host_increment(&usage, 0)?;
        let state = usage.funding.get(&id).expect("validated host transfer");
        let remaining = state
            .remaining
            .checked_sub(self.retained)
            .ok_or(WorkingMemoryError::Poisoned)?;
        let host_held = state
            .host_held
            .checked_sub(self.retained)
            .ok_or(WorkingMemoryError::Poisoned)?;
        let allocations = state
            .allocations
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let registrations = state
            .registrations
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let reserved = usage
            .reserved
            .checked_sub(self.retained)
            .ok_or(WorkingMemoryError::Poisoned)?;
        let registered = usage
            .registered
            .checked_add(self.retained)
            .ok_or(WorkingMemoryError::Overflow)?;
        let protected = state
            .protected_remaining()?
            .checked_sub(self.retained)
            .ok_or(WorkingMemoryError::Poisoned)?;
        if protected > remaining {
            return Err(WorkingMemoryError::Poisoned);
        }
        // All rejecting checks and provider comparisons precede counter commit.
        // Linking prepared slots does not compare provider keys again. If
        // source search unwinds, the outer staged Arc remains the key's owner
        // until both accounting and table attachment locks have unwound.
        if let Some((mut batch, namespace)) = self.prepared.take() {
            batch.entries[0] = Some((
                RegistryKey::Shared(Arc::clone(&self.key)),
                Entry {
                    reset_layout_id: None,
                    placement: Arc::clone(&pool.0.host_placement),
                    prepaid: None,
                    bytes: self.retained,
                    owners: 1,
                    funding: Some(id),
                    native_retired: false,
                    pending_allocation: false,
                    funding_allowance_bytes: 0,
                },
            ));
            if usage.storage.get(&TypeId::of::<K>()).is_none() {
                usage.storage.install(namespace);
            } else {
                // Retain the unused namespace in the staging owner; its H and
                // boxes must retire only after Usage and attachment locks.
                self.unused_namespace = Some(namespace);
            }
            usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .expect("prepared namespace")
                .downcast_mut::<Registry<K>>()
                .expect("typed storage registry")
                .link(batch);
        } else {
            unreachable!("fresh transfer owns its prepared registry destinations");
        }
        let state = usage.funding.get_mut(&id).expect("validated host transfer");
        state.remaining = remaining;
        state.host_held = host_held;
        state.allocations = allocations;
        state.registrations = registrations;
        custody.transfer_host_hold(self.retained);
        usage.reserved = reserved;
        usage.registered = registered;
        let mut registration = self.registration.take().expect("staged host registration");
        registration.activate(pool.clone(), Some(id));
        drop(usage);
        // Coercing the preallocated Box adds no fallible publication step. The
        // token's attachment slot was reserved before this provider ran.
        Ok(registration)
    }
}

struct DenseHostPublication<K: Ord + Send + Sync + 'static> {
    registration: Option<Box<WorkingMemoryStorage<K>>>,
    _host: eredu_core::HostPreparationAuthority,
}
impl<K: Ord + Send + Sync + 'static> eredu_core::SharedStorageRetirement
    for DenseHostPublication<K>
{
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
impl<K: Ord + Send + Sync + 'static> Drop for DenseHostPublication<K> {
    fn drop(&mut self) {
        if let Some(owner) = self.registration.take() {
            drop(*owner);
        }
    }
}
fn metadata_error(error: eredu_core::HostMetadataFundingError) -> WorkingMemoryError {
    WorkingMemoryError::MetadataConstruction(eredu_nn::workspace::WorkspaceMetadataError::Funding(
        error,
    ))
}
fn ordinary_attachment_bytes<K: HostSlotStorageKey>() -> Result<usize, WorkingMemoryError> {
    eredu_core::SharedStorageAttachmentTable::owned_attachment_control_bytes::<
        DenseHostPublication<K>,
        WorkingMemoryError,
    >()
    .ok_or(WorkingMemoryError::Overflow)
}
fn ordinary_host_owner_bytes<K: HostSlotStorageKey>() -> Result<usize, WorkingMemoryError> {
    let owner = usize::try_from(super::super::qualified_storage::shared_bytes::<
        DenseHostPublication<K>,
    >()?)
    .map_err(|_| WorkingMemoryError::Overflow)?;
    // Keys used by this closed transfer are cloned only from a caller-paid
    // actual key; nested key payload remains the caller's separate quotation.
    dense_host_transfer_control_bytes::<K>(0)?
        .checked_add(ordinary_attachment_bytes::<K>()?)
        .and_then(|n| n.checked_add(owner))
        .ok_or(WorkingMemoryError::Overflow)
}
pub(in crate::working_memory) fn prepare_dense_host_metadata<S, D, K: HostSlotStorageKey>(
    custody: &WorkingMemoryDecoderHostScope,
    execution: &InferenceExecutionIdentity,
) -> Result<eredu_core::HostPreparationAuthority, WorkingMemoryError> {
    let bytes = ordinary_dense_preparation_owner_bytes::<S, D, K>()?;
    let funding = custody
        .prepare_storage_metadata(execution)
        .map_err(metadata_error)?;
    funding.prepare_host_owner(bytes).map_err(metadata_error)
}
fn ordinary_dense_preparation_owner_bytes<S, D, K: HostSlotStorageKey>()
-> Result<usize, WorkingMemoryError> {
    ordinary_host_owner_bytes::<K>()?
        .checked_add(
            crate::DenseHostSlotInitialization::<S, D>::preparation_control_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
        )
        .ok_or(WorkingMemoryError::Overflow)
}
pub(in crate::working_memory) fn ordinary_dense_preparation_bytes<S, D, K: HostSlotStorageKey>()
-> Result<u64, WorkingMemoryError> {
    let bytes = super::super::StorageMetadataFunding::host_owner_bytes(
        ordinary_dense_preparation_owner_bytes::<S, D, K>()?,
    )
    .map_err(metadata_error)?;
    MemoryLedger::storage_metadata_control_bytes()?
        .checked_add(u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)?)
        .ok_or(WorkingMemoryError::Overflow)
}

/// Complete constructor allowance consumed from an ordinary dense prompt's
/// unprotected host envelope before publication. Actual nested key cloning is
/// quoted separately by the caller's concrete key producer.
pub(in crate::working_memory) fn ordinary_dense_host_publication_bytes<K: HostSlotStorageKey>()
-> Result<u64, WorkingMemoryError> {
    let host =
        super::super::StorageMetadataFunding::host_owner_bytes(ordinary_host_owner_bytes::<K>()?)
            .map_err(metadata_error)?;
    MemoryLedger::storage_metadata_control_bytes()?
        .checked_add(u64::try_from(host).map_err(|_| WorkingMemoryError::Overflow)?)
        .ok_or(WorkingMemoryError::Overflow)
}

pub(in crate::working_memory) fn dense_host_transfer_control_bytes<K: HostSlotStorageKey>(
    nested_key_bytes: u64,
) -> Result<usize, WorkingMemoryError> {
    use crate::working_memory::qualified_storage as q;
    use std::mem::{size_of, size_of_val};
    let n = |bytes: u64| usize::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow);
    let parts = [
        n(q::shared_bytes::<K>()?)?,
        n(q::shared_bytes::<Registration<K>>()?)?,
        n(q::array_bytes::<K>(1)?)?,
        n(q::vector_control_bytes::<K>()?)?,
        n(q::array_bytes::<Option<(RegistryKey<K>, Entry)>>(1)?)?,
        n(q::vector_control_bytes::<Option<(RegistryKey<K>, Entry)>>()?)?,
        size_of::<RegistryBatch<K>>(),
        size_of::<WorkingMemoryStorage<K>>(),
        n(directory::PreparedNamespace::requested_control_bytes::<K>()?)?,
        crate::HostSlotMetadata::copy_attachment_control_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?,
        n(nested_key_bytes
            .checked_mul(2)
            .ok_or(WorkingMemoryError::Overflow)?)?,
        size_of::<PreparedTransfer<K>>(),
        size_of::<Box<dyn Send + Sync>>(),
        size_of::<Result<bool, HostSlotAttachmentError<WorkingMemoryError>>>(),
        size_of::<Result<Box<dyn Send + Sync>, WorkingMemoryError>>(),
        size_of::<std::sync::MutexGuard<'_, crate::working_memory::Usage>>(),
        size_of::<Option<eredu_core::HostPreparationAuthority>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
}
