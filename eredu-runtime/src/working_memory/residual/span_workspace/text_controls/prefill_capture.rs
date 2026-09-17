//! Original path-source seal and borrowed accepted physical prefill contract.
use super::*;
use crate::layered::BoundCaptureSelection;
use crate::working_memory::InferenceRequest;

impl PreparedTextControlWorkspace {
    /// Seal the actual architecture path source before original acceptance.
    /// Equal path contents, a later binding, or a second association cannot
    /// replace it. This cold declaration grants no native operation.
    pub fn with_prefill_capture_selection(
        mut self,
        bound: BoundCaptureSelection<'_>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.prefill_paths.is_some()
            || self.binding.source.as_ref() != Some(bound.selection().source().storage_identity())
            || self.geometry() != bound.geometry()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.binding.prefill_paths = Some(bound.selection().paths().identity().clone());
        Ok(self)
    }
}

/// Original saved-token capture at an absolute logical decode coordinate.
/// Its first physical operation is the shared one-token prefill worker. The
/// source, paths, origin and request were sealed together before admission.
#[derive(Debug)]
pub struct AdmittedCaptureContinuation<'a> {
    reserved: ReservedTextSpanWorkspace<'a>,
    selection: &'a crate::layered::PreparedCaptureSelection,
    first: u64,
}
impl PreparedTextControlWorkspace {
    /// Bind the actual saved source's one-token opening to this same plan.
    pub fn with_capture_continuation_selection(
        mut self,
        checkpoint: &crate::capture::FundedCaptureCheckpoint,
        selection: &crate::layered::PreparedCaptureSelection,
    ) -> Result<Self, WorkingMemoryError> {
        checkpoint
            .validate_continuation_geometry(self.geometry())
            .map_err(checkpoint_geometry_error)?;
        if checkpoint.next_prediction() == 0
            || self.geometry().input_positions != 1
            || self.binding.prefill_paths.is_some()
            || self.binding.capture_first.is_some()
            || self.binding.source.as_ref() != Some(checkpoint.source().storage_identity())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        selection
            .validate_sources(checkpoint.source(), selection.paths())
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        self.binding.prefill_paths = Some(selection.paths().identity().clone());
        self.binding.capture_first = Some(checkpoint.next_prediction());
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Borrow the already admitted source/absolute-coordinate association.
    pub fn capture_continuation<'a>(
        &'a self,
        selection: &'a crate::layered::PreparedCaptureSelection,
        first: u64,
    ) -> Result<AdmittedCaptureContinuation<'a>, WorkingMemoryError> {
        let admitted = AdmittedCaptureContinuation {
            reserved: self.as_reserved_text_span_workspace(),
            selection,
            first,
        };
        admitted.validate_original()?;
        Ok(admitted)
    }
}
impl AdmittedCaptureContinuation<'_> {
    /// Actual immutable declaration source retained by the installed collector.
    pub fn selection(&self) -> &crate::layered::PreparedCaptureSelection {
        self.selection
    }
    /// Fresh physical geometry authenticated by the original reservation.
    pub fn geometry(&self) -> InferenceGeometry {
        self.reserved.workspace().plan().geometry()
    }
    /// Absolute logical decode coordinate of the first physical prefill.
    pub fn first_prediction(&self) -> u64 {
        self.first
    }
    fn validate_original(&self) -> Result<(), WorkingMemoryError> {
        let workspace = self.reserved.workspace();
        let binding = workspace
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if self.first == 0
            || self.geometry().input_positions != 1
            || binding.source.as_ref() != Some(self.selection.source().storage_identity())
            || binding.prefill_paths.as_ref() != Some(self.selection.paths().identity())
            || binding.capture_first != Some(self.first)
            || binding.geometry != self.geometry()
            || self.reserved.reservation().0.geometry != self.geometry()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pool = &self.reserved.reservation().0.pool;
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.reserved.validate_locked(pool, &usage)
    }
    pub(crate) fn validate_request(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        self.reserved.span.validate_request(request)?;
        if request.geometry() != self.geometry() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_original()
    }
}

/// Borrow of an original accepted physical capture contract. Only the consuming
/// original text owner can lend this view. It authenticates source, original
/// path identity, geometry, plan and reservation; the gateway must independently
/// validate the current execution's prepared token before preparing a source.
/// It proves neither complete opening inventory nor native quiescence.
#[derive(Debug)]
pub struct AdmittedPrefillCapture<'a> {
    reserved: ReservedTextSpanWorkspace<'a>,
    selection: BoundCaptureSelection<'a>,
}
impl OwnedTextSpanWorkspace {
    /// Lend the exact originally sealed companion. No quote, allocation, owner
    /// clone, reservation, scope or spend permission is created here.
    pub fn prefill_capture<'a>(
        &'a self,
        selection: BoundCaptureSelection<'a>,
    ) -> Result<AdmittedPrefillCapture<'a>, WorkingMemoryError> {
        let admitted = AdmittedPrefillCapture {
            reserved: self.as_reserved_text_span_workspace(),
            selection,
        };
        admitted.validate_original()?;
        Ok(admitted)
    }
}
impl<'a> AdmittedPrefillCapture<'a> {
    /// Actual immutable selection and original physical geometry.
    pub fn selection(&self) -> BoundCaptureSelection<'a> {
        self.selection
    }
    fn validate_original(&self) -> Result<(), WorkingMemoryError> {
        let workspace = self.reserved.workspace();
        let binding = workspace
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if binding.source.as_ref() != Some(self.selection.selection().source().storage_identity())
            || binding.prefill_paths.as_ref() != Some(self.selection.selection().paths().identity())
            || binding.geometry != self.selection.geometry()
            || workspace.plan().geometry() != self.selection.geometry()
            || self.reserved.reservation().0.geometry != self.selection.geometry()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Reuse the same accounting lock and exact accepted source validators.
        // This read-only health check is not a native completion witness.
        let pool = &self.reserved.reservation().0.pool;
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.reserved.validate_locked(pool, &usage)
    }
    pub(crate) fn validate_request(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        self.reserved.span.validate_request(request)?;
        if request.geometry() != self.selection.geometry() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_original()
    }
}

// One constructor local and its returned Result, the borrowed observer option,
// the read-only validation result, returned selection borrow and gateway match
// flag coexist with the original P controls.
// Native installed/call pair overlaps remain in the native Q producer.
pub(super) fn control_peak_bytes() -> Option<usize> {
    size_of::<AdmittedPrefillCapture<'static>>()
        .checked_add(size_of::<
            Result<AdmittedPrefillCapture<'static>, WorkingMemoryError>,
        >())?
        .checked_add(size_of::<Option<&AdmittedPrefillCapture<'static>>>())?
        .checked_add(size_of::<Result<(), WorkingMemoryError>>())?
        .checked_add(size_of::<BoundCaptureSelection<'static>>())?
        .checked_add(size_of::<bool>())?
        .checked_add(size_of::<AdmittedCaptureContinuation<'static>>())?
        .checked_add(size_of::<
            Result<AdmittedCaptureContinuation<'static>, WorkingMemoryError>,
        >())?
        .checked_add(size_of::<Option<&AdmittedCaptureContinuation<'static>>>())
}
