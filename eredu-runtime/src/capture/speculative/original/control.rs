//! Saved immutable authority over the collector's live, nonrefunding lineage.
use super::*;
use crate::capture::{PreparedSpeculativeActivationRestore, SpeculativeActivationCheckpoint};
use super::super::control::Authority;
use std::{alloc::Layout, sync::Arc};

pub(super) fn owner_control_bytes() -> Option<usize> {
    Some(Layout::new::<[usize; 2]>()
        .extend(Layout::new::<crate::capture::CaptureHostOwner>()).ok()?.0
        .pad_to_align().size())
}

#[derive(Debug)]
struct Payload {
    identity: String,
    scopes: Vec<SpeculativeCaptureScope>,
    interventions: Option<OriginalInterventionSource>,
    source: OriginalCaptureSource,
    lineage: crate::working_memory::OriginalEmbeddedCaptureLineage,
    // No observer, native state, mutable ledger snapshot or invocation counter.
    funding: HostMetadataFunding,
}
#[derive(Clone, Debug)]
pub(in crate::capture::speculative) struct Saved(Option<Arc<Payload>>);
impl Saved {
    fn get(&self) -> &Payload { self.0.as_ref().expect("live saved capture authority") }
}
impl Drop for Saved {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            // Release the actual shared allocation before its source/H can retire.
            drop(Arc::into_inner(value));
        }
    }
}
struct Prepared<'a> {
    observer: &'a mut OriginalSpeculativeCapture,
    identity: String,
    interventions: Option<InterventionSelection>,
}
impl PreparedSpeculativeActivationRestore for Prepared<'_> {
    fn commit(self: Box<Self>) {
        self.observer.identity = self.identity;
        self.observer.interventions = self.interventions;
        // Origins are scheduler annotations. IDs, C and the cumulative ledger
        // remain the live collector's values, including post-snapshot spending.
    }
}

impl OriginalSpeculativeCapture {
    fn drained_control_boundary(&self) -> bool {
        self.checkpoint_ready && self.active.is_none() && self.received.is_none()
            && self.prefix.is_none() && self.reductions.is_none() && self.held_prefill.is_none()
            && self.records.is_empty() && self.reduction_geometry.is_none()
    }
    fn control_boundary(&self) -> Result<(), OriginalSpeculativeCaptureError> {
        if !self.drained_control_boundary() { return Err(self.protocol(CaptureProtocolError::PreviousStep)); }
        let lineage = self.lineage.as_ref()
            .ok_or_else(|| self.protocol(CaptureProtocolError::Invocation))?;
        lineage.validate_source(&self.source)
            .and_then(|()| lineage.ledger().inspect_usage().map(|_| ()))
            .map_err(|cause| self.reject(self.funding.metadata_source(cause)))
    }
    fn control_frames() -> Option<usize> {
        let parts = [
            Layout::new::<[usize; 2]>().extend(Layout::new::<Payload>()).ok()?.0.pad_to_align().size(),
            size_of::<Payload>(), size_of::<Saved>(), size_of::<Authority>(),
            size_of::<SpeculativeActivationCheckpoint>(), size_of::<Prepared<'_>>(),
            size_of::<Box<Prepared<'_>>>(), size_of::<Box<dyn PreparedSpeculativeActivationRestore>>(),
            size_of::<Option<InterventionSelection>>(), size_of::<InterventionSelection>(),
            size_of::<Result<SpeculativeActivationCheckpoint, OriginalSpeculativeCaptureError>>(),
            size_of::<Result<Box<dyn PreparedSpeculativeActivationRestore>, OriginalSpeculativeCaptureError>>(),
            size_of::<(&Self, &SpeculativeActivationCheckpoint)>(),
            size_of::<(Option<OriginalInterventionSource>, &[SpeculativeCaptureScope], &str)>(),
            size_of::<Result<(), OriginalSpeculativeCaptureError>>(),
            size_of::<Result<CaptureUsage, crate::working_memory::WorkingMemoryError>>(),
            size_of::<OriginalSpeculativeCaptureError>(),
            crate::working_memory::CaptureRunLedger::inspection_control_bytes()?,
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn reserve_control(&self) -> Result<(), OriginalSpeculativeCaptureError> {
        self.funding.reserve_metadata(Self::control_frames()
            .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?)
            .map_err(|cause| self.reject(WorkspaceMetadataError::from(cause).into()))
    }
    /// Exact host checkpoint/prepared-restore census at a successful drained
    /// boundary. No native state or cumulative allowance is copied.
    pub fn control_storage_bytes(&self) -> Option<u64> {
        if !self.drained_control_boundary() { return None; }
        let lineage = self.lineage.as_ref()?;
        lineage.validate_source(&self.source).ok()?;
        lineage.ledger().inspect_usage().ok()?;
        let count = self.interventions.as_ref().map_or(0, |value| value.scopes.len());
        let one = Self::control_frames()?
            .checked_add(WorkspaceContext::metadata_string_bytes(self.identity.len())?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<SpeculativeCaptureScope>(count)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<bool>(count)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<[Option<CaptureSkipReason>; 2]>(count)?)?;
        u64::try_from(one.checked_mul(2)?).ok()
    }
    /// Saves only immutable edit authority and exact source/collector identity.
    /// The retained lineage is the existing live account, never saved usage.
    pub fn save_control(&self) -> Result<SpeculativeActivationCheckpoint, OriginalSpeculativeCaptureError> {
        self.reserve_control()?;
        self.control_boundary()?;
        let scopes = self.interventions.as_ref().map_or(&[][..], |value| value.scopes.as_slice());
        let mut copied = self.funding.metadata_vec(scopes.len()).map_err(|cause| self.reject(cause))?;
        copied.extend_from_slice(scopes);
        let identity = self.funding.metadata_string(format_args!("{}", self.identity))
            .map_err(|cause| self.reject(cause))?;
        Ok(SpeculativeActivationCheckpoint {
            authority: Authority::Original(Saved(Some(Arc::new(Payload {
                identity, scopes: copied,
                interventions: self.interventions.as_ref().map(|value| value.source.clone()),
                source: self.source.clone(),
                lineage: self.lineage.as_ref().expect("validated lineage").clone(),
                funding: self.funding.clone(),
            })))),
            host_owner: self.control_owner.clone(),
            original_funding: Some(self.funding.clone()),
        })
    }
    /// Authenticates the unchanged C authority at a fully drained boundary.
    /// Loaded artifact/session/declaration checks remain with the actual source
    /// projection, before compiling any replacement edit plan.
    pub fn validate_control_plan(&self, plan: &eredu_core::speculative::AdmittedSpeculativeActivations)
        -> Result<(), OriginalSpeculativeCaptureError>
    {
        self.reserve_control()?;
        self.control_boundary()?;
        if plan.captures().identity() != self.source.plan().admission().identity()
            || plan.capture_scopes() != self.scopes
        { return Err(self.protocol(CaptureProtocolError::Geometry)); }
        Ok(())
    }
    /// Read settled usage from this collector's exact request-minted lineage.
    /// This lends no request issuance authority and cannot reopen a closed run.
    /// Readmission rechecks the same live ledger before accepting any spending.
    pub fn control_usage(&self) -> Result<CaptureUsage, OriginalSpeculativeCaptureError> {
        self.reserve_control()?;
        self.control_boundary()?;
        self.lineage.as_ref().expect("validated source lineage").ledger().inspect_usage()
            .map_err(|cause| self.reject(self.funding.metadata_source(cause)))
    }
    /// Native composition has authenticated the loaded declaration and checked
    /// preflight with `inherited`. This joins its actual compiled I source to C,
    /// reserves ordinary logical readmission storage, and prepares atomic commit.
    pub fn prepare_readmitted_control<'a>(&'a mut self,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        source: Option<OriginalInterventionSource>, pool: &crate::working_memory::WorkingMemoryPool,
        inherited: CaptureUsage,
    ) -> Result<Box<dyn PreparedSpeculativeActivationRestore + 'a>, OriginalSpeculativeCaptureError> {
        let frames = [
            size_of::<(&Self, &eredu_core::speculative::AdmittedSpeculativeActivations,
                Option<OriginalInterventionSource>, &crate::working_memory::WorkingMemoryPool, CaptureUsage)>(),
            size_of::<CaptureLedger>(), size_of::<crate::working_memory::CaptureRunLedgerGuard<'_>>(),
            size_of::<(CaptureUsage, Option<CaptureSkipReason>)>(),
            size_of::<Result<Option<CaptureSkipReason>, CaptureError>>(),
            size_of::<Result<(), crate::working_memory::WorkingMemoryError>>(),
            size_of::<[&str; 9]>(), size_of::<[u64; 6]>(),
            OriginalCaptureSource::validation_control_bytes()
                .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?,
            OriginalInterventionSource::validation_control_bytes()
                .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?,
        ];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?)
            .map_err(|cause| self.reject(WorkspaceMetadataError::from(cause).into()))?;
        self.validate_control_plan(plan)?;
        self.source.validate_pool(pool).map_err(|cause| self.reject(self.funding.metadata_source(cause)))?;
        if let Some(source) = &source {
            source.validate_pool(pool).map_err(|cause| self.reject(self.funding.metadata_source(cause)))?;
            if source.plan().admission().identity() != plan.interventions().identity() {
                return Err(self.protocol(CaptureProtocolError::Geometry));
            }
        } else if !plan.interventions().is_empty() { return Err(self.protocol(CaptureProtocolError::Geometry)); }
        let lineage = self.lineage.as_ref().expect("validated lineage").clone();
        let mut guard = lineage.ledger().borrow().map_err(|cause| self.protocol(cause))?;
        if guard.usage() != inherited { return Err(self.protocol(CaptureProtocolError::Invocation)); }
        let mut ledger = CaptureLedger::with_inherited_usage(self.source.plan().admission(), inherited)
            .map_err(|cause| self.reject(self.funding.metadata_source(cause)))?;
        let bytes = super::super::control::storage(plan)
            .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?;
        let result = ledger.reserve(CaptureUsage { host_bytes: bytes, ..Default::default() });
        guard.record(ledger.total());
        drop(guard);
        if let Some(CaptureSkipReason::Limit { budget, cumulative }) = result
            .map_err(|cause| self.reject(self.funding.metadata_source(cause)))?
        { return Err(self.reject(self.funding.metadata_source(CaptureError::Limit { budget, cumulative }))); }
        self.prepare_control_authority(source, plan.intervention_scopes(), plan.identity())
    }
    /// Fully prepares replacement masks/identity before the enclosing controller
    /// copies native state. Dropping this value preserves the current authority.
    pub fn prepare_control<'a>(&'a mut self, saved: &SpeculativeActivationCheckpoint)
        -> Result<Box<dyn PreparedSpeculativeActivationRestore + 'a>, OriginalSpeculativeCaptureError>
    {
        self.reserve_control()?;
        self.control_boundary()?;
        let Authority::Original(saved_authority) = &saved.authority else {
            return Err(self.protocol(CaptureProtocolError::Invocation));
        };
        let value = saved_authority.get();
        if !Arc::ptr_eq(&self.control_owner, &saved.host_owner)
            || !self.source.same_source(&value.source)
            || !self.lineage.as_ref().expect("validated lineage").ledger().same_storage(value.lineage.ledger())
        { return Err(self.protocol(CaptureProtocolError::Invocation)); }
        self.prepare_control_authority(value.interventions.clone(), &value.scopes, &value.identity)
    }
    /// Installs an already compiled, source-qualified edit candidate through the
    /// same infallible commit used by restore. C and current lineage cannot change.
    pub(super) fn prepare_control_authority<'a>(&'a mut self,
        source: Option<OriginalInterventionSource>, scopes: &[SpeculativeCaptureScope], identity: &str,
    ) -> Result<Box<dyn PreparedSpeculativeActivationRestore + 'a>, OriginalSpeculativeCaptureError> {
        self.reserve_control()?;
        self.control_boundary()?;
        let capture = self.source.plan().admission();
        if identity.is_empty() || source.as_ref().map_or(!scopes.is_empty(), |source| {
            let plan = source.plan().admission();
            plan.request() != capture.request() || plan.invocation_bounds() != capture.invocation_bounds()
                || scopes.len() != plan.points().len() || scopes.len() != plan.plan().operations.len()
        }) { return Err(self.protocol(CaptureProtocolError::Geometry)); }
        let identity = self.funding.metadata_string(format_args!("{identity}"))
            .map_err(|cause| self.reject(cause))?;
        let interventions = if let Some(source) = source {
            let mut copied = self.funding.metadata_vec(scopes.len()).map_err(|cause| self.reject(cause))?;
            copied.extend_from_slice(scopes);
            let mut selected = self.funding.metadata_vec(scopes.len()).map_err(|cause| self.reject(cause))?;
            selected.resize(scopes.len(), false);
            let mut evidence_skips = self.funding.metadata_vec(scopes.len()).map_err(|cause| self.reject(cause))?;
            evidence_skips.resize_with(scopes.len(), || [None, None]);
            Some(InterventionSelection { scopes: copied, selected, evidence_skips, source })
        } else { None };
        Ok(Box::new(Prepared { observer: self, identity, interventions }))
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
