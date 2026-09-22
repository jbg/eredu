//! Immutable cold declarations for exact upcoming physical model invocations.
use super::*;

/// Borrowed source declaration at a drained invocation boundary. Quotation may
/// retain prospective descriptors without beginning or spending an invocation.
#[derive(Clone, Copy)]
pub struct OriginalSpeculativeCapturePreview<'a>(&'a OriginalSpeculativeCapture);

/// An immutable, paid description of one prospective physical invocation. It
/// owns no host capture frame, native role, or completion authority. Aggregate
/// suppression is deliberately not assumed: actual execution can use less than
/// this selected source population when earlier windows already supplied data.
pub struct OriginalSpeculativeCaptureProspect {
    selected: Vec<bool>,
    skipped: Vec<Option<CaptureSkipReason>>,
    identity: String,
    active: Active,
    interventions: Option<InterventionSelection>,
    source: OriginalCaptureSource,
    lineage: Option<crate::working_memory::OriginalModelCaptureLineage>,
    _funding: HostMetadataFunding,
}
impl OriginalSpeculativeCaptureProspect {
    /// Lends the same immutable phase source at one exact future prefill span.
    /// All physical extents and ordinals are checked; this spends no invocation.
    pub fn prefill_invocation(
        &self,
        span: SpeculativePrefillSpan,
        offset: u64,
    ) -> Result<OriginalSpeculativeCaptureInvocation<'_>, CaptureProtocolError> {
        let width = span
            .input_end
            .checked_sub(span.input_start)
            .ok_or(CaptureProtocolError::Geometry)?;
        let sequence = u64::try_from(width).map_err(|_| CaptureProtocolError::Geometry)?;
        let bounds = self
            .source
            .plan()
            .admission()
            .invocation_bounds()
            .ok_or(CaptureProtocolError::Invocation)?;
        if sequence == 0
            || sequence > bounds.max_sequence
            || !usize::try_from(width)
                .ok()
                .is_some_and(|width| span.validate(self.active.phase, width))
        {
            return Err(CaptureProtocolError::Geometry);
        }
        let mut descriptor = self.invocation();
        descriptor.active.invocation = self
            .active
            .invocation
            .checked_add(offset)
            .filter(|n| n.checked_add(1).is_some())
            .ok_or(CaptureProtocolError::Geometry)?;
        descriptor.active.sequence = sequence;
        descriptor.active.span = Some(span);
        Ok(descriptor)
    }
    /// Descriptive input for the same cold capture worker used by begun calls.
    /// Runtime must still authenticate and begin the actual source invocation.
    pub fn invocation(&self) -> OriginalSpeculativeCaptureInvocation<'_> {
        OriginalSpeculativeCaptureInvocation {
            source: &self.source,
            selected: &self.selected,
            skipped: &self.skipped,
            identity: &self.identity,
            interventions: self
                .interventions
                .as_ref()
                .map(|edits| (&edits.source, edits.selected.as_slice())),
            evidence_skips: self
                .interventions
                .as_ref()
                .map(|edits| edits.evidence_skips.as_slice()),
            active: self.active,
            lineage: self.lineage.as_ref(),
            prefix: None,
        }
    }
}
impl OriginalSpeculativeCapture {
    /// Lends the current immutable source only when no physical call is active.
    pub fn preview(&self) -> Option<OriginalSpeculativeCapturePreview<'_>> {
        (self.active.is_none() && self.received.is_none() && self.origin.is_some())
            .then_some(OriginalSpeculativeCapturePreview(self))
    }
}
impl OriginalSpeculativeCapturePreview<'_> {
    /// Copies one exact physical declaration using the existing source metadata
    /// payer. `offset` is its order among the upcoming calls, not a fresh claim.
    /// The real model role independently validates all span and cache geometry.
    pub fn prepare_invocation(
        self,
        phase: SpeculativeActivationPhase,
        sequence: usize,
        span: Option<SpeculativePrefillSpan>,
        offset: u64,
    ) -> Result<OriginalSpeculativeCaptureProspect, OriginalSpeculativeCaptureError> {
        let state = self.0;
        let origin = state
            .origin
            .ok_or_else(|| state.protocol(CaptureProtocolError::Invocation))?;
        let sequence =
            u64::try_from(sequence).map_err(|_| state.protocol(CaptureProtocolError::Geometry))?;
        let bounds = state
            .source
            .plan()
            .admission()
            .invocation_bounds()
            .ok_or_else(|| state.protocol(CaptureProtocolError::Invocation))?;
        if origin.request != state.request
            || origin.prediction < origin.committed_tokens
            || u64::try_from(origin.prediction)
                .ok()
                .is_none_or(|n| n >= bounds.max_predictions)
            || sequence == 0
            || sequence > bounds.max_sequence
            || span.is_some_and(|span| !span.validate(phase, sequence as usize))
        {
            return Err(state.protocol(CaptureProtocolError::Geometry));
        }
        let invocation = state
            .next_invocation
            .checked_add(offset)
            .filter(|n| n.checked_add(1).is_some())
            .ok_or_else(|| state.reject(WorkspaceMetadataError::Overflow.into()))?;
        let controls = [
            size_of::<Self>(),
            size_of::<OriginalSpeculativeCaptureProspect>(),
            size_of::<OriginalSpeculativeCaptureInvocation<'_>>(),
            size_of::<Result<OriginalSpeculativeCaptureProspect, OriginalSpeculativeCaptureError>>(
            ),
            size_of::<(
                SpeculativeActivationPhase,
                usize,
                Option<SpeculativePrefillSpan>,
                u64,
            )>(),
            size_of::<Option<InterventionSelection>>(),
            size_of::<
                std::iter::Zip<
                    std::slice::IterMut<'_, bool>,
                    std::slice::Iter<'_, SpeculativeCaptureScope>,
                >,
            >(),
        ];
        state
            .funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| state.reject(WorkspaceMetadataError::Overflow.into()))?,
            )
            .map_err(|cause| state.reject(WorkspaceMetadataError::from(cause).into()))?;
        let mut selected = state
            .funding
            .metadata_vec(state.scopes.len())
            .map_err(|cause| state.reject(cause))?;
        selected.extend(state.scopes.iter().map(|scope| scope.applies(phase)));
        let mut skipped = state
            .funding
            .metadata_vec(state.scopes.len())
            .map_err(|cause| state.reject(cause))?;
        skipped.resize(state.scopes.len(), None);
        let identity = state
            .funding
            .metadata_string(format_args!("{}", state.identity))
            .map_err(|cause| state.reject(cause))?;
        let interventions = state
            .interventions
            .as_ref()
            .map(|edits| {
                let mut scopes = state
                    .funding
                    .metadata_vec(edits.scopes.len())
                    .map_err(|cause| state.reject(cause))?;
                scopes.extend_from_slice(&edits.scopes);
                let mut selected = state
                    .funding
                    .metadata_vec(scopes.len())
                    .map_err(|cause| state.reject(cause))?;
                selected.extend(scopes.iter().map(|scope| scope.applies(phase)));
                let mut evidence_skips = state
                    .funding
                    .metadata_vec(scopes.len())
                    .map_err(|cause| state.reject(cause))?;
                evidence_skips.resize_with(scopes.len(), || [None, None]);
                Ok(InterventionSelection {
                    scopes,
                    selected,
                    evidence_skips,
                    source: edits.source.clone(),
                })
            })
            .transpose()?;
        Ok(OriginalSpeculativeCaptureProspect {
            selected,
            skipped,
            identity,
            active: Active {
                invocation,
                origin,
                phase,
                sequence,
                span,
            },
            interventions,
            source: state.source.clone(),
            lineage: state.lineage.clone(),
            _funding: state.funding.clone(),
        })
    }
}
