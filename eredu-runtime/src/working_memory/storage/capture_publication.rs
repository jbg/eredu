//! Single exact capture source publication, using the ordinary registry namespace.
use super::*;
use crate::working_memory::{
    InferenceSpanWorkspacePlan, Usage,
    funding::{RawSpanHostOwner, SpanHostCustody},
};
use eredu_core::{
    SharedStorageIdentity, SharedStorageOwner, SharedStorageRetirement, capture::SharedCapturePlan,
};
mod owner;
pub(in crate::working_memory) use owner::CaptureSourceOwner;
use std::{any::Any, fmt, marker::PhantomData, mem::size_of, sync::OnceLock};

/// Backend key for one physical admitted capture plan. The returned identity
/// must be that key's capture-plan namespace. Keys must retain only identity,
/// never the shared plan or its custody, and obey the ordinary registry contract.
/// Key cloning/comparison is cold provider code; its heap allocations remain the
/// provider's original preparation obligation, not covered by fixed key slots.
pub trait CapturePlanStorageKey: Clone + Ord + Send + Sync + 'static {
    /// Exact physical capture-plan identity, or None for another key namespace.
    fn capture_plan_identity(&self) -> Option<&SharedStorageIdentity>;
}

/// Typed terminal failure of one originally priced publication attempt.
#[derive(Debug, thiserror::Error)]
pub enum CapturePlanPublicationCause {
    #[error("original capture-plan publication is busy")]
    /// One source or accounting mutex is already borrowed.
    Busy,
    #[error("capture-plan domain has incompatible attachment custody")]
    /// Existing custody is opaque or has a different concrete owner type.
    AttachmentMismatch,
    #[error("capture-plan attachment metadata allocation failed: {0}")]
    /// The pre-existing attachment registry could not reserve metadata.
    Allocation(#[source] std::collections::TryReserveError),
    #[error("original capture-plan publication rejected: {0}")]
    /// Original account, identity, capacity or source-origin rejection.
    Storage(#[from] WorkingMemoryError),
}

pub(in crate::working_memory) struct PublicationLayout {
    pub(in crate::working_memory) source: SharedStorageIdentity,
    pub(in crate::working_memory) capacity: u64,
    pub(in crate::working_memory) new_bytes: u64,
    controls: u64,
    key: Arc<dyn Any + Send + Sync>,
    pub(in crate::working_memory) existing:
        Option<Arc<dyn crate::working_memory::saved_source::SavedSourceValidation + Send + Sync>>,
}
impl fmt::Debug for PublicationLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicationLayout")
            .field("source", &self.source)
            .field("capacity", &self.capacity)
            .field("new_bytes", &self.new_bytes)
            .field("controls", &self.controls)
            .finish_non_exhaustive()
    }
}
impl PublicationLayout {
    pub(in crate::working_memory) fn controls(&self) -> u64 {
        self.controls
    }
    pub(in crate::working_memory) fn key<K: CapturePlanStorageKey>(
        &self,
    ) -> Result<&K, WorkingMemoryError> {
        self.key
            .downcast_ref::<K>()
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
}

/// Cold typed one-key layout. It owns no publication authority. Bind it to the
/// actual text control plan before original acceptance; no later amount enters.
#[derive(Debug)]
pub struct PreparedCapturePlanPublication<K: CapturePlanStorageKey> {
    layout: Arc<PublicationLayout>,
    plan: InferenceSpanWorkspacePlan,
    key: PhantomData<K>,
}
impl<K: CapturePlanStorageKey> PreparedCapturePlanPublication<K> {
    /// Derive C from the actual source. Only an exact healthy existing one-key
    /// registration in this pool permits zero new C; no caller byte/boolean does.
    /// The caller must keep that existing registration in the quote's source
    /// joins (this constructor additionally retains it in the prepared layout).
    pub fn prepare(
        pool: &MemoryLedger,
        plan: &InferenceSpanWorkspacePlan,
        source: &SharedCapturePlan,
        key: K,
        existing: Option<&WorkingMemoryStorage<K>>,
    ) -> Result<Self, WorkingMemoryError> {
        if key.capture_plan_identity() != Some(source.storage_identity()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let capacity = source
            .capacity_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if let Some(existing) = existing {
            let usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            existing.validate_copy_source(pool, &usage)?;
            if existing.0.keys.len() != 1
                || existing.0.keys[0].cmp(&key) != Ordering::Equal
                || existing.bytes() != Some(capacity)
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        // No generic source factory survives into publication. A separate
        // erased key payload keeps the concrete key available without cloning
        // its logical source, and optional existing custody is retained below.
        let new_bytes = if existing.is_some() { 0 } else { capacity };
        let controls = publication_control_bytes::<K>().ok_or(WorkingMemoryError::Overflow)?;
        let layout = Arc::new(PublicationLayout {
            source: source.storage_identity().clone(),
            capacity,
            new_bytes,
            controls,
            key: Arc::new(key),
            existing: existing.map(|pin| {
                Arc::new(ExistingCaptureStorage(pin.clone()))
                    as Arc<
                        dyn crate::working_memory::saved_source::SavedSourceValidation
                            + Send
                            + Sync,
                    >
            }),
        });
        // The exact existing pin is validated again at sealing/reservation via
        // a closed layout validator, without an additional source Vec.
        Ok(Self {
            layout,
            plan: plan.clone(),
            key: PhantomData,
        })
    }
    /// Exact new C reserved by the original seal (zero only for a retained pin).
    pub fn new_source_bytes(&self) -> u64 {
        self.layout.new_bytes
    }
    /// Fixed publication controls, separate from physical C and provider payload.
    pub fn control_peak_bytes(&self) -> u64 {
        self.layout.controls
    }
    pub(in crate::working_memory) fn into_layout(
        self,
        plan: &InferenceSpanWorkspacePlan,
        source: &SharedStorageIdentity,
    ) -> Result<Arc<PublicationLayout>, WorkingMemoryError> {
        if !self.plan.same_plan(plan) || &self.layout.source != source {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(self.layout)
    }
}

// Includes the source owner, prepared registry slots and namespace, and their
// construction/retirement transport. Provider key payloads are separate facts.
fn publication_control_bytes<K: CapturePlanStorageKey>() -> Option<u64> {
    let bytes = size_of::<PublicationLayout>()
        .checked_add(size_of::<K>().checked_mul(3)?)?
        .checked_add(size_of::<Registration<K>>())?
        .checked_add(size_of::<ExistingCaptureStorage<K>>())?
        .checked_add(size_of::<WorkingMemoryStorage<K>>())? // reserved source pin payload
        .checked_add(size_of::<
            [crate::working_memory::residual::RegisteredStoragePin; 2],
        >())?
        .checked_add(size_of::<PublishedCaptureStorage<K>>())?
        .checked_add(size_of::<SharedStorageOwner<PublishedCaptureStorage<K>>>())? // returned typed attachment alias
        .checked_add(size_of::<CaptureSourceOwner>())?
        .checked_add(SharedCapturePlan::owned_attachment_control_bytes::<
            PublishedCaptureStorage<K>,
            CapturePlanPublicationCause,
        >()?)?
        .checked_add(size_of::<PreparedCapturePlanPublication<K>>())?
        .checked_add(size_of::<PreparedCaptureStorage<K>>())?
        .checked_add(size_of::<
            crate::working_memory::PendingCapturePlanPublication<K>,
        >())?
        .checked_add(size_of::<
            crate::working_memory::FailedCapturePlanPublication<K>,
        >())?
        .checked_mul(3)?
        // Eight Arc headers: layout, erased key, staged key, optional existing
        // validator, registration, published owner, reserved pin and fixed pair.
        .checked_add(16 * size_of::<usize>())?;
    u64::try_from(bytes)
        .ok()?
        .checked_add(directory::PreparedNamespace::requested_control_bytes::<K>().ok()?)?
        .checked_add(
            u64::try_from(
                size_of::<RegistryBatch<K>>() + size_of::<Option<(RegistryKey<K>, Entry)>>(),
            )
            .ok()?,
        )
}

struct ExistingCaptureStorage<K: CapturePlanStorageKey>(WorkingMemoryStorage<K>);
impl<K: CapturePlanStorageKey> crate::working_memory::saved_source::SavedSourceValidation
    for ExistingCaptureStorage<K>
{
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError> {
        self.0.validate_copy_source(pool, usage)
    }
    fn pin(&self) -> crate::working_memory::residual::RegisteredStoragePin {
        crate::working_memory::residual::RegisteredStoragePin::new(self.0.clone())
    }
    fn pin_control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        crate::working_memory::residual::RegisteredStoragePin::single_control_bytes::<K>(
            self.0.has_source_preparation(),
        )
    }
}

pub(in crate::working_memory) trait CaptureSourceValidation:
    fmt::Debug + Send + Sync
{
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError>;
}
// Registration/payload keys must retire before raw custody. No plan, full
// source association or control guard is reachable from this source sidecar.
pub(in crate::working_memory) struct PublishedCaptureStorage<K: CapturePlanStorageKey> {
    source: SharedStorageIdentity,
    capacity: u64,
    registration: OnceLock<CaptureRegistration<K>>,
    raw: RawSpanHostOwner,
}
enum CaptureRegistration<K: CapturePlanStorageKey> {
    Original(WorkingMemoryStorage<K>),
    // The source's actual typed owner remains the original one. A later
    // request's fixed witness retains that owner AND its own original raw hold.
    Alias(SharedStorageOwner<PublishedCaptureStorage<K>>),
}
impl<K: CapturePlanStorageKey> SharedStorageRetirement for PublishedCaptureStorage<K> {
    fn retire(self: Arc<Self>) {
        // All typed/erased aliases use this path; no Weak of this allocation exists.
        drop(Arc::into_inner(self));
    }
}
impl<K: CapturePlanStorageKey> PublishedCaptureStorage<K> {
    fn original_registration(&self) -> Result<&WorkingMemoryStorage<K>, WorkingMemoryError> {
        match self.registration.get() {
            Some(CaptureRegistration::Original(storage)) => Ok(storage),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}
impl<K: CapturePlanStorageKey> fmt::Debug for PublishedCaptureStorage<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublishedCaptureStorage")
            .field("published", &self.registration.get().is_some())
            .finish_non_exhaustive()
    }
}
impl<K: CapturePlanStorageKey> CaptureSourceValidation for PublishedCaptureStorage<K> {
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError> {
        self.raw.validate_origin_locked(pool, usage)?;
        match self
            .registration
            .get()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?
        {
            CaptureRegistration::Original(storage) => storage.validate_copy_source(pool, usage),
            CaptureRegistration::Alias(owner) => owner.validate(pool, usage),
        }
    }
}

pub(in crate::working_memory) struct PreparedCaptureStorage<K: CapturePlanStorageKey> {
    key: Arc<dyn Borrow<K> + Send + Sync>,
    registration: Option<WorkingMemoryStorage<K>>,
    pub(in crate::working_memory) published: SharedStorageOwner<PublishedCaptureStorage<K>>,
    capacity: u64,
    new_bytes: u64,
    batch: Option<Box<RegistryBatch<K>>>,
    namespace: Option<std::sync::Mutex<directory::PreparedNamespace>>,
}
impl<K: CapturePlanStorageKey> fmt::Debug for PreparedCaptureStorage<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCaptureStorage")
            .field("capacity", &self.capacity)
            .field("published", &self.published.registration.get().is_some())
            .finish_non_exhaustive()
    }
}
impl<K: CapturePlanStorageKey> PreparedCaptureStorage<K> {
    pub(in crate::working_memory) fn new(
        layout: &PublicationLayout,
        raw: RawSpanHostOwner,
    ) -> Result<Self, WorkingMemoryError> {
        let key = layout.key::<K>()?;
        // Provider clones run before raw registry custody exists. The accepted
        // span hold covers their allocations and any failed clone prefix; an
        // unpublished empty namespace cannot establish a native quarantine.
        let staged_key = Arc::new(key.clone());
        let registration = WorkingMemoryStorage::pending(vec![key.clone()], layout.capacity);
        let batch = RegistryBatch::prepare(1, raw.clone());
        let namespace = directory::PreparedNamespace::prepare::<K>(Some(raw.clone()));
        let published = SharedStorageOwner::new(PublishedCaptureStorage {
            source: layout.source.clone(),
            capacity: layout.capacity,
            registration: OnceLock::new(),
            raw,
        });
        Ok(Self {
            key: staged_key,
            registration: Some(registration),
            published,
            capacity: layout.capacity,
            new_bytes: layout.new_bytes,
            batch: Some(batch),
            namespace: Some(std::sync::Mutex::new(namespace)),
        })
    }
    pub(in crate::working_memory) fn publish(
        &mut self,
        source: &SharedCapturePlan,
        native: &WorkingMemoryFundingScope,
        custody: &SpanHostCustody,
    ) -> Result<(), CapturePlanPublicationCause> {
        let domain = native.pool().shared_storage_accounting_id().clone();
        let owner = source
            .try_attach_owned_nonblocking(&domain, || {
                self.commit(native, custody)?;
                Ok(self.published.clone())
            })
            .map_err(|error| match error {
                eredu_core::SharedStorageAttachmentError::Busy => CapturePlanPublicationCause::Busy,
                eredu_core::SharedStorageAttachmentError::AttachmentMismatch => {
                    CapturePlanPublicationCause::AttachmentMismatch
                }
                eredu_core::SharedStorageAttachmentError::Poisoned => {
                    WorkingMemoryError::Poisoned.into()
                }
                eredu_core::SharedStorageAttachmentError::Overflow => {
                    WorkingMemoryError::Overflow.into()
                }
                eredu_core::SharedStorageAttachmentError::Provider(error) => error,
            })?;
        if !owner.same_owner(&self.published) {
            // Source lock is released. Only this private original owner can
            // authenticate reuse; opaque/different typed attachments never get
            // here and cannot cause a new unattached registry entry.
            let pool = native.pool().clone();
            let usage = pool.0.usage.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => CapturePlanPublicationCause::Busy,
                std::sync::TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned.into(),
            })?;
            self.published
                .raw
                .validate_publication_locked(native, &usage)?;
            custody.validate_prepublication_locked(&pool, &usage)?;
            if owner.source != self.published.source || owner.capacity != self.capacity {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            let registered = owner.original_registration()?;
            let key: &K = self.key.as_ref().borrow();
            if registered.0.keys.len() != 1
                || registered.0.keys[0].cmp(key) != Ordering::Equal
                || registered.bytes() != Some(self.capacity)
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            owner.validate(&pool, &usage)?;
            drop(usage);
            self.published
                .registration
                .set(CaptureRegistration::Alias(owner))
                .unwrap_or_else(|_| unreachable!("single typed source alias"));
        }
        Ok(())
    }
    fn commit(
        &mut self,
        native: &WorkingMemoryFundingScope,
        custody: &SpanHostCustody,
    ) -> Result<(), CapturePlanPublicationCause> {
        if self.published.registration.get().is_some() || self.registration.is_none() {
            return Err(WorkingMemoryError::PreparationAlreadyStarted.into());
        }
        let pool = native.pool().clone();
        let activation_pool = pool.clone();
        let mut usage = pool.0.usage.try_lock().map_err(|e| match e {
            std::sync::TryLockError::WouldBlock => CapturePlanPublicationCause::Busy,
            std::sync::TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned.into(),
        })?;
        self.published
            .raw
            .validate_publication_locked(native, &usage)?;
        custody.validate_prepublication_locked(&pool, &usage)?;
        let key: &K = self.key.as_ref().borrow();
        let prior = usage.storage.get(&TypeId::of::<K>()).and_then(|r| {
            r.downcast_ref::<Registry<K>>()
                .expect("typed registry")
                .locate(key)
        });
        let locator = prior.map(|(locator, _)| locator);
        let existing = if let Some((_, entry)) = prior {
            same_capacity(entry.bytes, self.capacity)?;
            if entry.placement != pool.0.host_placement {
                return Err(WorkingMemoryError::StoragePlacementMismatch.into());
            }
            validate_entry_origin(entry, &usage)?;
            entry
                .owners
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?;
            true
        } else {
            false
        };
        if !existing && self.new_bytes != self.capacity {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let _ = pool.0.check_host_increment(&usage, 0)?;
        let debit = if existing { 0 } else { self.capacity };
        let state = usage
            .funding
            .get(&native.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let available = state.spendable_remaining()?;
        if debit > available {
            return Err(WorkingMemoryError::DomainAllowanceExceeded {
                domain: pool.topology().host_domain(),
                required_bytes: debit,
                available_bytes: available,
            }
            .into());
        }
        let remaining = state
            .remaining
            .checked_sub(debit)
            .ok_or(WorkingMemoryError::Poisoned)?;
        let allocations = state
            .allocations
            .checked_add(usize::from(!existing))
            .ok_or(WorkingMemoryError::Overflow)?;
        let registrations = state
            .registrations
            .checked_add(usize::from(!existing))
            .ok_or(WorkingMemoryError::Overflow)?;
        let reserved = usage
            .reserved
            .checked_sub(debit)
            .ok_or(WorkingMemoryError::Poisoned)?;
        let registered = usage
            .registered
            .checked_add(debit)
            .ok_or(WorkingMemoryError::Overflow)?;
        // All comparisons and fallible checks precede linking prepared slots.
        if usage.storage.get(&TypeId::of::<K>()).is_none() {
            usage.storage.install(
                self.namespace
                    .take()
                    .expect("prepared capture namespace")
                    .into_inner()
                    .unwrap_or_else(|poison| poison.into_inner()),
            );
        }
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .unwrap()
            .downcast_mut::<Registry<K>>()
            .expect("typed capture namespace");
        if let Some(locator) = locator {
            registry.at_mut(locator).owners += 1;
        } else {
            let mut batch = self.batch.take().expect("prepared capture row");
            batch.entries[0] = Some((
                RegistryKey::Shared(Arc::clone(&self.key)),
                Entry {
                    reset_layout_id: None,
                    placement: Arc::clone(&pool.0.host_placement),
                    prepaid: None,
                    bytes: self.capacity,
                    owners: 1,
                    funding: Some(native.id),
                    native_retired: false,
                    pending_allocation: false,
                    funding_allowance_bytes: 0,
                },
            ));
            registry.link(batch);
        }
        let state = usage
            .funding
            .get_mut(&native.id)
            .expect("validated original account");
        state.remaining = remaining;
        state.allocations = allocations;
        state.registrations = registrations;
        usage.reserved = reserved;
        usage.registered = registered;
        self.registration
            .as_mut()
            .expect("staged registration")
            .activate(activation_pool, (!existing).then_some(native.id));
        drop(usage);
        self.published
            .registration
            .set(CaptureRegistration::Original(
                self.registration.take().expect("committed registration"),
            ))
            .unwrap_or_else(|_| unreachable!("single private source publication"));
        Ok(())
    }
}
