//! A separate source channel for one closed scheduled capture segment.
use super::{capture_tensor::CaptureSourceRollback, *};
use crate::working_memory::saved_source::SavedSourceValidation;

mod opening;
mod prefill;
use crate::working_memory::storage::bounded_pin::OpeningPinOwner;
pub(in crate::working_memory) use prefill::CapturePinIdentity;
mod span;
pub use prefill::{
    PreparedPrefillChunkRetention, SettledCaptureSourceParcel, SettledPrefillChunkRetention,
};
pub(super) use span::ActiveCaptureSpan;
#[derive(Debug)]
struct CaptureSourceIdentity {
    prefill: Option<prefill::PrefillStamp>,
}

pub(super) fn control_bytes() -> usize {
    // Heap payload plus the actual Box handle; the enclosing H formula adds
    // two constructor/move overlaps for these fixed representations.
    std::mem::size_of::<CaptureSourceSlot>()
        + std::mem::size_of::<Box<CaptureSourceSlot>>()
        + std::mem::size_of::<SettledCaptureSourceParcel>()
        + std::mem::size_of::<CaptureSourceIdentity>()
        + std::mem::size_of::<Option<PreparedPrefillChunkRetention>>()
        + std::mem::size_of::<Option<SettledPrefillChunkRetention>>()
        + std::mem::size_of::<crate::inspection::PrefillChunkRetentionContext<'_>>()
        // One extra erased pin element in the existing quarantine aggregate,
        // including the caller-held installation Option representation.
        + std::mem::size_of::<RegisteredStoragePin>()
        + std::mem::size_of::<Option<OpeningPinOwner>>()
        // Shared binding worker nested below the strict or projected entry.
        + std::mem::size_of::<(&CaptureTensorCustody, &WorkingMemoryFundingScope,
            &CaptureSourceSegment, &WorkingMemoryStorage<u8>,
            &eredu_core::capture::AdmittedCapturePlan,
            Option<&crate::working_memory::OriginalInterventionSource>)>()
        + std::mem::size_of::<Result<CaptureSourceRollback<'_>, WorkingMemoryError>>()
}

pub(super) type CaptureSourceEntries = Vec<Arc<dyn SavedSourceValidation + Send + Sync>>;

// Only this closed wrapper can enter the segment. No external validator or
// callback executes under Usage; it is exactly the existing storage check.
struct Source<K: Ord + Send + 'static>(WorkingMemoryStorage<K>);
impl<K: Clone + Ord + Send + Sync + 'static> SavedSourceValidation for Source<K> {
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError> {
        self.0.validate_copy_source(pool, usage)
    }
    fn pin(&self) -> RegisteredStoragePin {
        RegisteredStoragePin::new(self.0.clone())
    }
    fn pin_control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        crate::working_memory::residual::RegisteredStoragePin::single_control_bytes::<K>(
            self.0.has_source_preparation(),
        )
    }
}

pub(super) struct CaptureSourceSlot {
    pub(super) sources: CaptureSourceEntries,
    opening: Option<OpeningPinOwner>,
    identity: Arc<CaptureSourceIdentity>,
}
impl std::fmt::Debug for CaptureSourceSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptureSourceSlot")
            .field("sources", &self.sources.len())
            .field("opening_group", &self.opening.is_some())
            .finish_non_exhaustive()
    }
}
impl CaptureSourceSlot {
    pub(super) fn validate(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_sources(pool, usage)?;
        if let Some(opening) = &self.opening {
            opening.validate_origins(pool, usage)?;
        }
        Ok(())
    }
    fn validate_sources(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        for source in &self.sources {
            source.validate(pool, usage)?;
        }
        Ok(())
    }
    // Scope quarantine owns all source origins, including empty and zero-byte
    // inventories. Both preparation and destruction occur outside Usage.
    pub(super) fn into_pin(self) -> RegisteredStoragePin {
        let Self {
            sources,
            opening,
            identity,
        } = self;
        let pin = RegisteredStoragePin::aggregate(
            sources
                .iter()
                .map(|source| source.pin())
                .chain(opening.map(|owner| owner.into_pin())),
        );
        drop(sources);
        drop(identity);
        pin
    }
}

/// Move-only association with one scheduled capture channel in one actual scope.
///
/// This is accounting-only custody, not a host/native allowance, permission or
/// completion certificate. Dropping this handle never clears the scope's slot
/// or its pins. Failed/abandoned work therefore remains in the same recovery and
/// quarantine path. There is deliberately no public release or reset operation.
#[derive(Debug)]
#[must_use = "retain alongside exact capture roots; Drop does not release unresolved pins"]
pub struct CaptureSourceSegment {
    identity: Arc<CaptureSourceIdentity>,
    custody: CaptureTensorCustody,
}
impl CaptureSourceSegment {
    fn slot<'s>(
        &self,
        native: &'s WorkingMemoryFundingScope,
    ) -> Result<&'s CaptureSourceSlot, WorkingMemoryError> {
        native
            .capture_source
            .as_deref()
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }

    /// Recheck exact scope, original schedule/account and all attached origins.
    /// This is cold validation only, with no publication or completion effect.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        let slot = self.slot(native)?;
        let usage = native
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.custody
            .validate_scheduled_native_locked(native, &usage)?;
        slot.validate(&native.pool, &usage)
    }

    // Foundation only: no production caller yet. The future canonical successful
    // finish_guarded_chunk path must establish exact scope/segment association,
    // completed capture, and release actual roots outside native owner locks
    // before reaching this operation. A semantic callback alone is insufficient.
    // No new certificate is produced and no scope is certified or amount refunded.
    pub(crate) fn retire_after_settled_boundary(
        &mut self,
        native: &mut WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        let pool = native.pool.clone();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.custody
            .validate_scheduled_native_locked(native, &usage)?;
        let slot = self.slot(native)?;
        if slot.opening.is_some() {
            // An installed opening group has only the canonical parcel path.
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        slot.validate(&pool, &usage)?;
        // This foundation path has no canonical ticket. It remains usable only
        // for channels that never activated exclusive account spending.
        usage
            .funding
            .get(&native.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .validate_span_spend(None)?;
        let retired = native.capture_source.take();
        drop(usage);
        // Keys/registrations may acquire Usage during destruction. Physical
        // owners elsewhere still retain their charge; no assumed credit here.
        drop(retired);
        Ok(())
    }
}
impl CaptureTensorCustody {
    pub(in crate::working_memory) fn begin_source_segment(
        &self,
        native: &mut WorkingMemoryFundingScope,
    ) -> Result<CaptureSourceSegment, WorkingMemoryError> {
        // Identity is created lazily, only for this scheduled channel. Ordinary
        // scopes allocate nothing for their absent slot. Build before Usage.
        let identity = Arc::new(CaptureSourceIdentity { prefill: None });
        let custody = self.share_scheduled();
        // Stage the complete heap slot before Usage or any scope mutation.
        let slot = Box::new(CaptureSourceSlot {
            sources: Vec::new(),
            opening: None,
            identity: identity.clone(),
        });
        let pool = native.pool.clone();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_scheduled_native_locked(native, &usage)?;
        if native.capture_source.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        native.capture_source = Some(slot);
        drop(usage);
        Ok(CaptureSourceSegment { identity, custody })
    }

    pub(in crate::working_memory) fn bind_segment_source<
        's,
        K: Clone + Ord + Send + Sync + 'static,
    >(
        &self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
        source: &WorkingMemoryStorage<K>,
    ) -> Result<CaptureSourceRollback<'s>, WorkingMemoryError> {
        if !self.same_schedule(&segment.custody) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.bind_validated_segment_source(native, segment, source)
    }

    // Only receipt-issued projected claims call this after validating their
    // exact canonical fragment. Their destination H is independently paid;
    // the original source channel still belongs to the global scheduled bank.
    pub(in crate::working_memory) fn bind_projected_segment_source<
        's,
        K: Clone + Ord + Send + Sync + 'static,
    >(
        &self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
        source: &WorkingMemoryStorage<K>,
        admission: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<CaptureSourceRollback<'s>, WorkingMemoryError> {
        segment.validate_projected_admission(admission)?;
        self.bind_validated_segment_source(native, segment, source)
    }

    fn bind_validated_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        &self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
        source: &WorkingMemoryStorage<K>,
    ) -> Result<CaptureSourceRollback<'s>, WorkingMemoryError> {
        let previous = segment.slot(native)?;
        let count = previous
            .sources
            .len()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut sources = Vec::with_capacity(count);
        sources.extend(previous.sources.iter().cloned());
        sources
            .push(Arc::new(Source(source.clone())) as Arc<dyn SavedSourceValidation + Send + Sync>);
        let pool = native.pool.clone();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_scheduled_native_locked(native, &usage)?;
        segment
            .custody
            .validate_scheduled_native_locked(native, &usage)?;
        // Opening pins are independent of the rollback-owned transform list.
        if let Some(opening) = &previous.opening {
            opening.validate_origins(&pool, &usage)?;
        }
        // Revalidate every origin atomically, including earlier contributions.
        for source in &sources {
            source.validate(&pool, &usage)?;
        }
        let slot = native
            .capture_source
            .as_mut()
            .expect("exclusively borrowed matching slot");
        let previous = std::mem::replace(&mut slot.sources, sources);
        drop(usage);
        Ok(CaptureSourceRollback::segment(native, previous))
    }
}
