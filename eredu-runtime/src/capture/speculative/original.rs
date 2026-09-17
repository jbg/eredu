//! Source-only internal observer state. Native work belongs to the model phase.
use crate::{
    capture::{CaptureObservationStep, CaptureProtocolError},
    working_memory::{OriginalCaptureSource, OriginalInterventionSource},
};
use eredu_core::{
    SpeculativeRequestId,
    capture::*,
    speculative::{
        SpeculativeActivationCapture, SpeculativeActivationOrigin, SpeculativeActivationPhase,
        SpeculativeCaptureScope, SpeculativePrefillSpan,
    },
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceMetadataFunding};
use std::mem::{size_of, size_of_val};

mod aggregate;
pub(super) mod control;
mod prefix;
pub use prefix::OriginalSpeculativeCapturePrefix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Active {
    invocation: u64,
    origin: SpeculativeActivationOrigin,
    phase: SpeculativeActivationPhase,
    sequence: u64,
    span: Option<SpeculativePrefillSpan>,
}

/// Borrowed exact source, applicability and attribution for one admitted outer
/// invocation. It describes the native phase's input; it grants no model role,
/// source publication, capture frame, numerical work or completion authority.
#[derive(Clone, Copy, Debug)]
pub struct OriginalSpeculativeCaptureInvocation<'a> {
    source: &'a OriginalCaptureSource,
    selected: &'a [bool],
    skipped: &'a [Option<CaptureSkipReason>],
    identity: &'a str,
    interventions: Option<(&'a OriginalInterventionSource, &'a [bool])>,
    evidence_skips: Option<&'a [[Option<CaptureSkipReason>; 2]]>,
    active: Active,
    lineage: Option<&'a crate::working_memory::OriginalEmbeddedCaptureLineage>,
    prefix: Option<&'a OriginalSpeculativeCapturePrefix>,
}
impl<'a> OriginalSpeculativeCaptureInvocation<'a> {
    /// Already charged logical frame/envelope/aggregate prefix for this exact
    /// invocation. It grants no host destination, native work, or completion.
    pub fn prepared_prefix(self) -> Option<&'a OriginalSpeculativeCapturePrefix> {
        self.prefix
    }
    /// Exact request-owned cumulative source, when initialized before quoting.
    pub fn lineage(self) -> Option<&'a crate::working_memory::OriginalEmbeddedCaptureLineage> {
        self.lineage
    }
    /// Actual immutable C source, not an equivalent independently copied plan.
    pub fn source(self) -> &'a OriginalCaptureSource {
        self.source
    }
    /// Actual architecture scopes selected for this physical phase.
    pub fn selected(self) -> &'a [bool] {
        self.selected
    }
    /// Exact logical aggregate decisions for inactive capture rows. These
    /// remain separate from edit scopes and never authorize native work.
    pub fn skip_reasons(self) -> &'a [Option<CaptureSkipReason>] {
        self.skipped
    }
    /// Actual independently compiled edit source and this phase's exact scope mask.
    pub fn interventions(self) -> Option<(&'a OriginalInterventionSource, &'a [bool])> {
        self.interventions
    }
    /// Exact logical Before/After evidence decisions for each declared edit.
    /// This never disables the edit itself or grants a physical evidence claim.
    pub fn intervention_evidence_skips(self) -> Option<&'a [[Option<CaptureSkipReason>; 2]]> {
        self.evidence_skips
    }
    /// Loaded admission identity for the enclosing envelope.
    pub fn admission_identity(self) -> &'a str {
        self.identity
    }
    /// Monotone outer collector occurrence, independent of logical coordinates.
    pub fn invocation(self) -> u64 {
        self.active.invocation
    }
    /// Existing shared scheduler provenance.
    pub fn origin(self) -> SpeculativeActivationOrigin {
        self.active.origin
    }
    /// Actual target/prediction equation phase.
    pub fn phase(self) -> SpeculativeActivationPhase {
        self.active.phase
    }
    /// Physical input rows, independent of cache extent and prediction index.
    pub fn sequence(self) -> u64 {
        self.active.sequence
    }
    /// Existing physical prefill span, when the driver provided it.
    pub fn prefill_span(self) -> Option<SpeculativePrefillSpan> {
        self.active.span
    }
    /// Global row placement from the actual shared scheduler span.
    pub fn window(self) -> Result<Option<CaptureInvocationWindow>, CaptureError> {
        super::invocation_window(self.active.phase, self.active.span)
    }
    /// Ordinary capture phase for this physical equation.
    pub fn capture_phase(self) -> CapturePhase {
        phase(self.active.phase)
    }
    /// Same selected readout predicate as the funded/cold capture worker.
    /// Context is supplied later by the actual model equation.
    pub fn requires_sequence_readout(self) -> Result<bool, CaptureProtocolError> {
        if self.interventions.is_some_and(|(source, selected)| {
            source
                .plan()
                .admission()
                .plan()
                .operations
                .iter()
                .zip(selected)
                .any(|(operation, selected)| {
                    *selected
                        && operation
                            .schedule
                            .includes(self.capture_phase(), self.active.origin.prediction as u64)
                })
        }) {
            return Ok(true);
        }
        CaptureObservationStep::with_invocation(
            self.source.plan().admission(),
            self.capture_phase(),
            self.active.origin.prediction as u64,
            Some(CaptureInvocationShape {
                batch: 1,
                sequence: self.active.sequence,
                context: None,
            }),
        )?
        .requires_sequence_readout_for(self.selected)
    }
    /// The shared ordinary envelope reservation, separate from physical H.
    /// Cold quotation and the funded ledger consume this same logical policy.
    pub fn envelope_usage(self) -> Result<CaptureUsage, CaptureError> {
        super::envelope_usage(self.identity.len(), self.active.span.is_some())
    }
    /// Named model-role envelope/identity destination. The native phase must add
    /// these controls to its actual role before constructing the String; queue
    /// backing is independently paid by the source-only observer.
    pub fn envelope_control_bytes(self) -> Option<usize> {
        let parts = [
            WorkspaceContext::metadata_string_bytes(self.identity.len())?,
            size_of::<SpeculativeActivationCapture>(),
            OriginalSpeculativeCapturePrefix::control_bytes()?,
            size_of::<(usize, bool, CaptureUsage)>(),
            size_of::<Option<CaptureUsage>>(),
            size_of::<Result<CaptureUsage, CaptureError>>(),
            size_of::<(CapturePhase, CaptureInvocationShape)>(),
            size_of::<Result<(), crate::working_memory::CaptureRunHostError>>(),
            size_of::<CapturedStepDelivery>(),
            size_of::<Option<SpeculativeActivationCapture>>(),
            size_of::<String>(),
            size_of::<Result<SpeculativeActivationCapture, CaptureProtocolError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Moves a fully paid identity and the actual completed funded frame into
    /// the ordinary shared-driver envelope. Its frame retains the model-role H
    /// covering the identity; no second capture payload or Arc is constructed.
    pub fn envelope(
        self,
        identity: String,
        frame: SharedCapturedStep,
        completed: bool,
    ) -> Result<SpeculativeActivationCapture, CaptureProtocolError> {
        if identity != self.identity
            || frame.phase() != self.capture_phase()
            || frame.prediction_index() != self.active.origin.prediction as u64
            || !frame
                .invocation()
                .is_some_and(|shape| shape.batch == 1 && shape.sequence == self.active.sequence)
        {
            return Err(CaptureProtocolError::Geometry);
        }
        Ok(SpeculativeActivationCapture {
            admission_identity: Some(identity),
            invocation: self.active.invocation,
            origin: self.active.origin,
            phase: self.active.phase,
            prefill_span: self.active.span,
            completed,
            captures: CapturedStepDelivery::Shared(frame),
            prefill_reductions: None,
        })
    }
}

fn phase(phase: SpeculativeActivationPhase) -> CapturePhase {
    match phase {
        SpeculativeActivationPhase::TargetPrefill
        | SpeculativeActivationPhase::PredictionPrefill => CapturePhase::Prefill,
        _ => CapturePhase::Decode,
    }
}

/// A fixed or counted preparation failure retaining the source and host account.
/// Dynamic diagnostic allocation uses the same funding producer before escape.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalSpeculativeCaptureError {
    #[source]
    cause: eredu_nn::Error,
    _source: OriginalCaptureSource,
    _interventions: Option<OriginalInterventionSource>,
    _funding: WorkspaceMetadataFunding,
}

/// Array-free source, scope mask and delivery state for one original request.
/// This owner never executes observation callbacks. The native adapter borrows
/// its descriptor, quotes the existing worker and installs a local funded observer.
struct InterventionSelection {
    scopes: Vec<SpeculativeCaptureScope>,
    selected: Vec<bool>,
    evidence_skips: Vec<[Option<CaptureSkipReason>; 2]>,
    source: OriginalInterventionSource,
}
pub struct OriginalSpeculativeCapture {
    scopes: Vec<SpeculativeCaptureScope>,
    selected: Vec<bool>,
    skipped: Vec<Option<CaptureSkipReason>>,
    identity: String,
    request: SpeculativeRequestId,
    origin: Option<SpeculativeActivationOrigin>,
    span: Option<SpeculativePrefillSpan>,
    active: Option<Active>,
    received: Option<SpeculativeActivationCapture>,
    records: Vec<SpeculativeActivationCapture>,
    held_prefill: Option<SpeculativeActivationCapture>,
    reduction_geometry: Option<eredu_core::speculative::SpeculativePrefillReductionGeometry>,
    reductions: Option<aggregate::Aggregate>,
    next_invocation: u64,
    prefix: Option<OriginalSpeculativeCapturePrefix>,
    checkpoint_ready: bool,
    control_owner: std::sync::Arc<crate::capture::CaptureHostOwner>,
    // Payload and queue backing retire before their actual source/H owners.
    source: OriginalCaptureSource,
    lineage: Option<crate::working_memory::OriginalEmbeddedCaptureLineage>,
    interventions: Option<InterventionSelection>,
    funding: WorkspaceMetadataFunding,
}
impl OriginalSpeculativeCapture {
    /// Source-copy-independent constructor frames. Caller Box/adapter/error
    /// transport controls remain the caller's actual producer responsibility.
    pub fn preparation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            control::owner_control_bytes()?,
            crate::working_memory::CaptureRunLedger::inspection_control_bytes()?,
            size_of::<Active>(),
            size_of::<Option<Active>>(),
            size_of::<OriginalSpeculativeCaptureInvocation<'_>>(),
            size_of::<crate::working_memory::OriginalEmbeddedCaptureLineage>(),
            size_of::<(Self, crate::working_memory::OriginalEmbeddedCaptureLineage)>(),
            size_of::<Result<(), crate::working_memory::WorkingMemoryError>>(),
            size_of::<OriginalSpeculativeCaptureError>(),
            size_of::<Result<Self, OriginalSpeculativeCaptureError>>(),
            size_of::<Result<(), OriginalSpeculativeCaptureError>>(),
            size_of::<CaptureProtocolError>(),
            size_of::<(
                OriginalCaptureSource,
                &[SpeculativeCaptureScope],
                &str,
                SpeculativeRequestId,
                WorkspaceMetadataFunding,
            )>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Consumes the already compiled C source. Scope/identity copies use the
    /// counted H producer and the admission plan is never cloned here.
    pub fn prepare(
        source: OriginalCaptureSource,
        scopes: &[SpeculativeCaptureScope],
        identity: &str,
        request: SpeculativeRequestId,
        funding: WorkspaceMetadataFunding,
    ) -> Result<Self, OriginalSpeculativeCaptureError> {
        let reject = |cause| OriginalSpeculativeCaptureError {
            cause,
            _source: source.clone(),
            _interventions: None,
            _funding: funding.clone(),
        };
        funding
            .reserve_metadata(
                Self::preparation_control_bytes()
                    .ok_or_else(|| reject(WorkspaceMetadataError::Overflow.into()))?,
            )
            .map_err(|cause| reject(eredu_nn::Error::from(WorkspaceMetadataError::from(cause))))?;
        if source
            .plan()
            .admission()
            .invocation_bounds()
            .is_none_or(|bounds| bounds.batch != 1)
            || scopes.len() != source.plan().admission().points().len()
            || identity.is_empty()
        {
            return Err(reject(
                funding.metadata_source(CaptureProtocolError::Geometry),
            ));
        }
        let mut scope_copy = funding.metadata_vec(scopes.len()).map_err(reject)?;
        scope_copy.extend_from_slice(scopes);
        let mut selected = funding.metadata_vec(scopes.len()).map_err(reject)?;
        selected.resize(scopes.len(), false);
        let mut skipped = funding.metadata_vec(scopes.len()).map_err(reject)?;
        skipped.resize(scopes.len(), None);
        let identity = funding
            .metadata_string(format_args!("{identity}"))
            .map_err(reject)?;
        Ok(Self {
            scopes: scope_copy,
            selected,
            skipped,
            identity,
            request,
            origin: None,
            span: None,
            active: None,
            received: None,
            records: Vec::new(),
            held_prefill: None,
            reduction_geometry: None,
            reductions: None,
            next_invocation: 0,
            prefix: None,
            checkpoint_ready: true,
            control_owner: std::sync::Arc::new(crate::capture::CaptureHostOwner::default()),
            source,
            lineage: None,
            interventions: None,
            funding,
        })
    }
    /// Attach the immutable edit declaration before the first invocation. The
    /// same source H pays scope/mask storage; this issues no model or edit claim.
    pub fn with_interventions(
        mut self,
        source: OriginalInterventionSource,
        scopes: &[SpeculativeCaptureScope],
    ) -> Result<Self, OriginalSpeculativeCaptureError> {
        let reject = |cause| OriginalSpeculativeCaptureError {
            cause,
            _source: self.source.clone(),
            _interventions: Some(source.clone()),
            _funding: self.funding.clone(),
        };
        self.funding
            .reserve_metadata(std::mem::size_of::<(
                InterventionSelection,
                Option<InterventionSelection>,
                &mut Self,
                OriginalInterventionSource,
                &[SpeculativeCaptureScope],
                Result<Self, OriginalSpeculativeCaptureError>,
            )>())
            .map_err(|cause| reject(WorkspaceMetadataError::from(cause).into()))?;
        let plan = source.plan().admission();
        let capture = self.source.plan().admission();
        if self.next_invocation != 0
            || self.active.is_some()
            || self.interventions.is_some()
            || plan.request() != capture.request()
            || plan.invocation_bounds() != capture.invocation_bounds()
            || scopes.len() != plan.points().len()
            || scopes.len() != plan.plan().operations.len()
        {
            return Err(reject(
                self.funding.metadata_source(CaptureProtocolError::Geometry),
            ));
        }
        let mut copied = self.funding.metadata_vec(scopes.len()).map_err(reject)?;
        copied.extend_from_slice(scopes);
        let mut selected = self.funding.metadata_vec(scopes.len()).map_err(reject)?;
        selected.resize(scopes.len(), false);
        let mut evidence_skips = self.funding.metadata_vec(scopes.len()).map_err(reject)?;
        evidence_skips.resize_with(scopes.len(), || [None, None]);
        self.interventions = Some(InterventionSelection {
            scopes: copied,
            selected,
            evidence_skips,
            source,
        });
        Ok(self)
    }
    /// Attach the already paid exact request ledger before any invocation.
    /// Later native quotation must authenticate this same owner against its
    /// actual request; equal C declarations cannot substitute another lineage.
    pub fn with_lineage(
        mut self,
        lineage: crate::working_memory::OriginalEmbeddedCaptureLineage,
    ) -> Result<Self, OriginalSpeculativeCaptureError> {
        if self.lineage.is_some() || self.active.is_some() || self.next_invocation != 0 {
            return Err(self.protocol(CaptureProtocolError::Invocation));
        }
        lineage
            .validate_source(&self.source)
            .map_err(|cause| self.reject(self.funding.metadata_source(cause)))?;
        self.lineage = Some(lineage);
        Ok(self)
    }
    fn reject(&self, cause: eredu_nn::Error) -> OriginalSpeculativeCaptureError {
        OriginalSpeculativeCaptureError {
            cause,
            _source: self.source.clone(),
            _interventions: self
                .interventions
                .as_ref()
                .map(|value| value.source.clone()),
            _funding: self.funding.clone(),
        }
    }
    fn protocol(&self, cause: CaptureProtocolError) -> OriginalSpeculativeCaptureError {
        self.reject(self.funding.metadata_source(cause))
    }
    /// Source identity, useful for the native caller's exact pool validation.
    pub fn source(&self) -> &OriginalCaptureSource {
        &self.source
    }
    /// Set by the same speculative driver; never inferred from state length.
    pub fn set_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.origin = origin;
    }
    /// Set by the same split-prefill driver and checked against the next phase.
    pub fn set_prefill_span(&mut self, span: Option<SpeculativePrefillSpan>) {
        self.span = span;
    }
    /// Prepares only source-owned queue backing and the existing scope mask.
    /// Physical capture quota and model role are consumed by the native phase.
    pub fn begin(
        &mut self,
        phase: SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), OriginalSpeculativeCaptureError> {
        if self.active.is_some() || self.received.is_some() {
            return Err(self.protocol(CaptureProtocolError::PreviousStep));
        }
        let origin = self
            .origin
            .ok_or_else(|| self.protocol(CaptureProtocolError::Invocation))?;
        let sequence =
            u64::try_from(sequence).map_err(|_| self.protocol(CaptureProtocolError::Geometry))?;
        let bounds = self
            .source
            .plan()
            .admission()
            .invocation_bounds()
            .expect("validated source");
        if origin.request != self.request
            || origin.prediction < origin.committed_tokens
            || origin.prediction as u64 >= bounds.max_predictions
            || sequence == 0
            || sequence > bounds.max_sequence
            || self
                .span
                .is_some_and(|span| !span.validate(phase, sequence as usize))
        {
            return Err(self.protocol(CaptureProtocolError::Geometry));
        }
        let next = self
            .next_invocation
            .checked_add(1)
            .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?;
        let controls = [
            size_of::<Active>(),
            size_of::<Option<Active>>(),
            size_of::<CaptureProtocolError>(),
            size_of::<Result<(), OriginalSpeculativeCaptureError>>(),
            size_of::<(&mut Self, SpeculativeActivationPhase, usize)>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| self.reject(WorkspaceMetadataError::Overflow.into()))?;
        self.funding
            .reserve_metadata(bytes)
            .map_err(|cause| self.reject(WorkspaceMetadataError::from(cause).into()))?;
        self.funding
            .reserve_metadata_vec(
                &mut self.records,
                if self.reduction_geometry.is_some() {
                    2
                } else {
                    1
                },
            )
            .map_err(|cause| self.reject(cause))?;
        self.skipped.fill(None);
        for (selected, scope) in self.selected.iter_mut().zip(&self.scopes) {
            *selected = scope.applies(phase);
        }
        if let Some(edits) = &mut self.interventions {
            edits.evidence_skips.fill_with(|| [None, None]);
            for (selected, scope) in edits.selected.iter_mut().zip(&edits.scopes) {
                *selected = scope.applies(phase);
            }
        }
        let active = Active {
            invocation: self.next_invocation,
            origin,
            phase,
            sequence,
            span: self.span,
        };
        // Once preparation can spend quota, this attempted occurrence cannot be
        // retried as fresh work. The existing finalizer ends it on failure.
        self.checkpoint_ready = false;
        self.active = Some(active);
        self.next_invocation = next;
        self.prepare_aggregate_invocation(active)?;
        Ok(())
    }
    /// Borrows the prepared descriptor through the unchanged observer bridge.
    pub fn invocation(&self) -> Option<OriginalSpeculativeCaptureInvocation<'_>> {
        Some(OriginalSpeculativeCaptureInvocation {
            source: &self.source,
            selected: &self.selected,
            skipped: &self.skipped,
            identity: &self.identity,
            interventions: self
                .interventions
                .as_ref()
                .map(|value| (&value.source, value.selected.as_slice())),
            evidence_skips: self
                .interventions
                .as_ref()
                .map(|value| value.evidence_skips.as_slice()),
            active: self.active?,
            lineage: self.lineage.as_ref(),
            prefix: self.prefix.as_ref(),
        })
    }
    /// Accepts only the native phase's already paid shared frame envelope.
    pub fn receive(
        &mut self,
        envelope: SpeculativeActivationCapture,
    ) -> Result<(), CaptureProtocolError> {
        let active = self.active.ok_or(CaptureProtocolError::Transaction)?;
        if self.received.is_some()
            || envelope.invocation != active.invocation
            || envelope.origin != active.origin
            || envelope.phase != active.phase
            || envelope.prefill_span != active.span
            || envelope.admission_identity.as_deref() != Some(self.identity.as_str())
        {
            return Err(CaptureProtocolError::Geometry);
        }
        let frame = envelope
            .captures
            .shared()
            .ok_or(CaptureProtocolError::Invocation)?;
        if frame.phase() != phase(active.phase)
            || frame.prediction_index() != active.origin.prediction as u64
            || !frame
                .invocation()
                .is_some_and(|shape| shape.batch == 1 && shape.sequence == active.sequence)
        {
            return Err(CaptureProtocolError::Geometry);
        }
        self.received = Some(envelope);
        Ok(())
    }
    /// The surrounding driver only completes after the real model phase supplied
    /// its completed frame. This performs no callback, allocation or native work.
    pub fn complete(&mut self) -> Result<(), OriginalSpeculativeCaptureError> {
        if self.active.is_none()
            || !self
                .received
                .as_ref()
                .is_some_and(|record| record.completed)
        {
            return Err(self.protocol(CaptureProtocolError::Transaction));
        }
        let prepared = match &mut self.reductions {
            Some(aggregate) => {
                let frame = self
                    .received
                    .as_ref()
                    .expect("validated frame")
                    .captures
                    .as_step();
                aggregate
                    .group
                    .prepare_fixed(&frame.records, &frame.interventions)
            }
            None => Ok(()),
        };
        prepared.map_err(|cause| self.aggregate_error(cause))
    }
    /// Infallible final delivery after native completion or retained failure.
    /// Earlier invocation IDs and all source/role/capture spending remain spent.
    pub fn finish(&mut self, success: bool) {
        self.checkpoint_ready = success && self.received.as_ref().is_some_and(|frame| frame.completed);
        self.active = None;
        self.prefix = None;
        if let Some(mut envelope) = self.received.take() {
            envelope.completed &= success;
            if let Some(aggregate) = &mut self.reductions {
                let frame = envelope.captures.as_step();
                aggregate.group.finish_window(
                    &frame.records,
                    &frame.interventions,
                    envelope.invocation,
                    envelope.completed,
                );
                if let Some(previous) = self.held_prefill.replace(envelope) {
                    debug_assert!(self.records.len() < self.records.capacity());
                    self.records.push(previous);
                }
            } else {
                debug_assert!(self.records.len() < self.records.capacity());
                self.records.push(envelope);
            }
        }
    }
    /// Moves the exact envelope; no raw DTO or tensor payload is cloned.
    pub fn take(&mut self) -> Option<SpeculativeActivationCapture> {
        (!self.records.is_empty()).then(|| self.records.remove(0))
    }
}
