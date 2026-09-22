//! Exact scheduled-bank association; canonical runtime success remains separate.
use super::*;
use crate::inspection::PrefillChunkRetentionContext;
use crate::prefill::PrefillChunk;
use crate::working_memory::InferenceRequest;
use eredu_core::{DistributedCommitEpoch, DistributedCommitOutcome, capture::SharedCapturePlan};

pub(super) struct PrefillStamp {
    request: InferenceRequest,
    source: SharedCapturePlan,
    // Exact original source admitted by the parent bank. Its two-side geometry
    // sources remain distinct; equal freshly rebuilt companions cannot match.
    interventions: Option<crate::working_memory::OriginalInterventionSource>,
    chunk: PrefillChunk,
    epoch: DistributedCommitEpoch,
    // Every escaped registration/ticket retains the same original host H.
    custody: CaptureTensorCustody,
}
impl PrefillStamp {
    fn matches_admission(&self, admission: &eredu_core::capture::AdmittedCapturePlan) -> bool {
        std::ptr::eq(admission, self.source.admission())
            || self.interventions.as_ref().is_some_and(|original| {
                original
                    .plan()
                    .admission()
                    .plan()
                    .operations
                    .iter()
                    .enumerate()
                    .any(|(index, operation)| {
                        operation
                            .schedule
                            .includes(eredu_core::capture::CapturePhase::Prefill, 0)
                            && original.plan().evidence(index).is_some_and(|companion| {
                                std::ptr::eq(
                                    admission,
                                    companion.shared_geometry_source().admission(),
                                )
                            })
                    })
            })
    }
    pub(super) fn validate_context_coordinates(
        &self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.request.validate_same_request(context.request())?;
        let chunk = context.chunk();
        if self.epoch != context.epoch()
            || self.chunk.input != chunk.input
            || self.chunk.position != chunk.position
            || self.chunk.output != chunk.output
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}
impl std::fmt::Debug for PrefillStamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillStamp")
            .field("chunk", &self.chunk)
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

/// Registration of one actual source slot. Only a bank-derived bootstrap can
/// create it. It supplies no success evidence; Drop leaves all scope pins live.
#[derive(Debug)]
#[must_use = "the canonical runtime retains this registration through chunk completion"]
pub struct PreparedPrefillChunkRetention {
    identity: Arc<CaptureSourceIdentity>,
}
/// One-use success issued after the exact model epoch and both existing chunk
/// reservation guards settled. Drop does not retire roots/pins or certify work.
/// Native record retirement and matching carrier ownership remain separate.
#[derive(Debug)]
#[must_use = "consume only at the canonical post-chunk retirement callback"]
pub struct SettledPrefillChunkRetention {
    identity: Arc<CaptureSourceIdentity>,
}
impl PreparedPrefillChunkRetention {
    pub(crate) fn validate_context(
        &self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        stamp.validate_context_coordinates(context)?;
        stamp.custody.validate()
    }
    pub(crate) fn validate_commit(
        &self,
        context: &PrefillChunkRetentionContext<'_>,
        outcome: Option<DistributedCommitOutcome>,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_context(context)?;
        if outcome != Some(DistributedCommitOutcome::Committed(context.epoch())) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    // Only the canonical SessionPrefill success branch calls this after checking
    // validate_commit and both guard settlements. No public constructor exists.
    pub(crate) fn into_settled(self) -> SettledPrefillChunkRetention {
        SettledPrefillChunkRetention {
            identity: self.identity,
        }
    }
}
impl SettledPrefillChunkRetention {
    /// Successful chunk coordinates, retained without a caller byte/epoch input.
    pub fn chunk(&self) -> &PrefillChunk {
        &self
            .identity
            .prefill
            .as_ref()
            .expect("stamped ticket")
            .chunk
    }
    /// The actual successfully committed model epoch.
    pub fn epoch(&self) -> DistributedCommitEpoch {
        self.identity
            .prefill
            .as_ref()
            .expect("stamped ticket")
            .epoch
    }
    pub(crate) fn validate_request(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self.identity.prefill.as_ref().expect("stamped ticket");
        stamp.request.validate_same_request(request)?;
        stamp.custody.validate()
    }
}
impl CaptureTensorCustody {
    pub(in crate::working_memory) fn begin_prefill_source_segment(
        &self,
        source: &SharedCapturePlan,
        interventions: Option<&crate::working_memory::OriginalInterventionSource>,
        native: &mut WorkingMemoryFundingScope,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(CaptureSourceSegment, PreparedPrefillChunkRetention), WorkingMemoryError> {
        let reservation = context.request().memory_reservation();
        if reservation.0.funding != Some(native.id) || !reservation.0.pool.same_ledger(&native.pool)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(original) = interventions {
            original.validate_pool(&native.pool)?;
            let plan = original.plan().admission();
            if plan.request() != source.admission().request()
                || plan.text_origin() != source.admission().text_origin()
                || plan.invocation_bounds() != source.admission().invocation_bounds()
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        // Every shared source/account owner is prepared before Usage. A failed
        // validation drops these only after the lock guard has gone away.
        let identity = Arc::new(CaptureSourceIdentity {
            prefill: Some(PrefillStamp {
                request: context.request().clone(),
                source: source.clone(),
                interventions: interventions.cloned(),
                chunk: context.chunk().clone(),
                epoch: context.epoch(),
                custody: self.share_scheduled(),
            }),
        });
        let segment = CaptureSourceSegment {
            identity: identity.clone(),
            custody: self.share_scheduled(),
        };
        let registration = PreparedPrefillChunkRetention {
            identity: identity.clone(),
        };
        // H priced this actual slot/Box before admission. Failed validation
        // destroys it only after Usage has been released.
        let slot = Box::new(CaptureSourceSlot {
            sources: Vec::new(),
            opening: None,
            identity,
        });
        let pool = native.pool.clone();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_scheduled_native_locked(native, &usage)?;
        FundingSource::NativeScope(native).validate(&usage, &reservation.0.execution)?;
        if native.capture_source.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        native.capture_source = Some(slot);
        drop(usage);
        Ok((segment, registration))
    }
}
impl CaptureSourceSegment {
    // A separately paid projected destination must still come from this exact
    // retained admission. Equal declarations and another request do not match.
    pub(super) fn validate_projected_admission(
        &self,
        admission: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !stamp.matches_admission(admission) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(in crate::working_memory) fn validate_summary_geometry(
        &self,
        geometry: &eredu_core::capture::CaptureSummaryGeometry<'_>,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !stamp.matches_admission(geometry.admission())
            || geometry.phase() != eredu_core::capture::CapturePhase::Prefill
            || geometry.prediction() != 0
            || !geometry.matches_prefill(
                stamp.request.geometry(),
                &stamp.chunk.input,
                stamp.chunk.position,
                stamp.chunk.output,
            )
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        stamp.custody.validate()
    }
    pub(in crate::working_memory) fn validate_histogram_geometry(
        &self,
        geometry: &eredu_core::capture::CaptureHistogramGeometry<'_>,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !stamp.matches_admission(geometry.admission())
            || geometry.phase() != eredu_core::capture::CapturePhase::Prefill
            || geometry.prediction() != 0
            || !geometry.matches_prefill(
                stamp.request.geometry(),
                &stamp.chunk.input,
                stamp.chunk.position,
                stamp.chunk.output,
            )
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        stamp.custody.validate()
    }
    pub(in crate::working_memory) fn validate_candidate_geometry(
        &self,
        geometry: &eredu_core::capture::CaptureCandidateGeometry<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_terminal_geometry(
            geometry.admission(),
            geometry.phase(),
            geometry.prediction(),
            geometry.source_shape()[1],
        )
    }
    pub(in crate::working_memory) fn validate_token_score_geometry(
        &self,
        geometry: &eredu_core::capture::CaptureTokenScoreGeometry<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_terminal_geometry(
            geometry.admission(),
            geometry.phase(),
            geometry.prediction(),
            geometry.source_shape()[1],
        )
    }
    fn validate_terminal_geometry(
        &self,
        admission: &eredu_core::capture::AdmittedCapturePlan,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        rows: usize,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let inference = stamp.request.geometry();
        let expected_rows = if stamp.chunk.output == eredu_core::OutputDemand::Sequence {
            stamp.chunk.input.end - stamp.chunk.input.start
        } else {
            1
        };
        if !stamp.matches_admission(admission)
            || phase != eredu_core::capture::CapturePhase::Prefill
            || prediction != 0
            || stamp.chunk.input.end != inference.input_positions
            || rows as u64 != expected_rows
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        stamp.custody.validate()
    }
    /// Cold placement association only. The actual fragment must belong to the
    /// retained admission and this exact announced native chunk. No target,
    /// native source, allocation, completion or causal-equivalence proof follows.
    pub fn validate_prefill_fragment(
        &self,
        fragment: &eredu_core::capture::CapturePrefillFragment<'_, '_>,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !stamp.matches_admission(fragment.assembly().logical_geometry().admission())
            || fragment.assembly().inference_geometry() != stamp.request.geometry()
            || !fragment.matches_chunk(&stamp.chunk.input, stamp.chunk.position, stamp.chunk.output)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        stamp.custody.validate()
    }
    /// Validate sparse batch/token coordinates against the same original request,
    /// source owner and announced chunk as dense capture. This grants no source
    /// publication, native completion or logical provider-coverage evidence.
    pub fn validate_routed_prefill_fragment(
        &self,
        fragment: &eredu_core::capture::CaptureRoutedPrefillFragment<'_, '_>,
    ) -> Result<(), WorkingMemoryError> {
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !std::ptr::eq(
            fragment.plan().geometry().admission(),
            stamp.source.admission(),
        ) || fragment.plan().inference_geometry() != stamp.request.geometry()
            || !fragment.matches_chunk(&stamp.chunk.input, stamp.chunk.position, stamp.chunk.output)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        stamp.custody.validate()
    }
    /// Read-only exact success association. This neither removes the channel nor
    /// certifies the native scope. The future closed native carrier must also
    /// prove record retirement and payload-before-pin destruction.
    pub fn validate_settled_ticket(
        &self,
        native: &WorkingMemoryFundingScope,
        ticket: &SettledPrefillChunkRetention,
    ) -> Result<(), WorkingMemoryError> {
        if !Arc::ptr_eq(&self.identity, &ticket.identity) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_native_scope(native)
    }
}

/// Detached custody of one canonically completed capture source channel.
///
/// This opaque, non-Clone parcel owns the exact source entries and original
/// stamp. It exposes no byte credit, keys, scope, native allowance or completion
/// authority. The backend must independently establish exact native settlement
/// and record retirement, then retire actual native roots/publications before
/// dropping this parcel outside its locks. Independent aliases keep their charge.
#[must_use = "keep source custody until matching native payloads retire outside locks"]
pub struct SettledCaptureSourceParcel {
    // Field ownership, not a destructor callback or a scalar refund.
    _sources: CaptureSourceEntries,
    _opening: Option<OpeningPinOwner>,
    _identity: Arc<CaptureSourceIdentity>,
}

impl std::fmt::Debug for SettledCaptureSourceParcel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettledCaptureSourceParcel")
            .field("source_origins", &self._sources.len())
            .finish_non_exhaustive()
    }
}

impl CaptureSourceSegment {
    /// Move only this matching capture channel into an opaque custody parcel.
    ///
    /// The exact runtime-issued ticket is necessary but does not prove native
    /// record/Array retirement. Those remain the backend's obligations. Validate
    /// and acquire every caller loan before this operation; after success retain
    /// the parcel through native payload destruction outside all caller locks.
    /// The slot can be taken once. Permanent scope pins are never modified.
    pub fn take_settled_sources(
        &self,
        native: &mut WorkingMemoryFundingScope,
        ticket: &SettledPrefillChunkRetention,
    ) -> Result<SettledCaptureSourceParcel, WorkingMemoryError> {
        if !Arc::ptr_eq(&self.identity, &ticket.identity) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pool = native.pool.clone();
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.custody
            .validate_scheduled_native_locked(native, &usage)?;
        self.slot(native)?.validate(&pool, &usage)?;
        let slot = native
            .capture_source
            .take()
            .expect("validated exact source slot");
        // validate_scheduled_native_locked checked exact marker ownership under
        // this same lock. Detach once together with the exact source slot. The
        // weak owner is dropped only after unlocking, while slot remains strong.
        let marker = usage
            .funding
            .get_mut(&native.id)
            .expect("validated source account")
            .active_span
            .take();
        drop(usage);
        drop(marker);
        let CaptureSourceSlot {
            sources,
            opening,
            identity,
        } = *slot;
        Ok(SettledCaptureSourceParcel {
            _sources: sources,
            _opening: opening,
            _identity: identity,
        })
    }
}

/// Existing stamped identity only. No new allocation or caller-mintable stamp.
#[derive(Debug)]
pub(in crate::working_memory) struct CapturePinIdentity(Arc<CaptureSourceIdentity>);
impl CapturePinIdentity {
    // Successful rows may escape together. S covers each retained stamp Arc's
    // actual payload/counters, independently of H's active-channel move peak.
    pub(in crate::working_memory) fn retained_control_bytes() -> usize {
        std::mem::size_of::<CaptureSourceIdentity>() + 2 * std::mem::size_of::<usize>()
    }

    pub(in crate::working_memory) fn request(
        &self,
    ) -> Result<&InferenceRequest, WorkingMemoryError> {
        Ok(&self
            .0
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .request)
    }
}
impl CaptureSourceSegment {
    pub(in crate::working_memory) fn pin_identity(&self) -> CapturePinIdentity {
        CapturePinIdentity(self.identity.clone())
    }
    pub(in crate::working_memory) fn validate_pin_locked(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
        identity: &CapturePinIdentity,
        context: Option<&PrefillChunkRetentionContext<'_>>,
        source: &eredu_core::SharedStorageIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_pin_association_locked(native, usage, identity, context, source)?;
        self.slot(native)?.validate(&native.pool, usage)
    }
    // Association only: never recurses through an installed opening group.
    pub(in crate::working_memory) fn validate_pin_association_locked(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
        identity: &CapturePinIdentity,
        context: Option<&PrefillChunkRetentionContext<'_>>,
        source: &eredu_core::SharedStorageIdentity,
    ) -> Result<(), WorkingMemoryError> {
        if !Arc::ptr_eq(&self.identity, &identity.0) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if stamp.source.storage_identity() != source {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(context) = context {
            stamp.validate_context_coordinates(context)?;
        }
        let reservation = stamp.request.memory_reservation();
        if reservation.0.funding != Some(native.id) || !reservation.0.pool.same_ledger(&native.pool)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.custody
            .validate_scheduled_native_locked(native, usage)?;
        FundingSource::NativeScope(native).validate(usage, &reservation.0.execution)?;
        self.slot(native)?;
        Ok(())
    }
}
