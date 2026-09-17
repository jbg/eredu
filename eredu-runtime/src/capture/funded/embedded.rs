//! A model invocation lends the same funded observer and shares request usage.
use super::*;
use crate::working_memory::{OriginalCaptureSource, OriginalEmbeddedSpeculativeRole};
use eredu_core::speculative::SpeculativeActivationOrigin;

/// One explicit model role's capture bank. It owns the same source and request
/// lineage after the quote borrow ends; native scope/completion remain external.
#[derive(Debug)]
pub struct FundedEmbeddedCaptureInvocation {
    session: FundedCaptureSession,
    source: OriginalCaptureSource,
    origin: SpeculativeActivationOrigin,
    role: OriginalEmbeddedSpeculativeRole,
}
impl FundedEmbeddedCaptureInvocation {
    pub(crate) fn from_run(
        run: PreparedCaptureRun,
        lineage: crate::working_memory::CaptureRunLedger,
        source: OriginalCaptureSource,
        role: OriginalEmbeddedSpeculativeRole,
        origin: SpeculativeActivationOrigin,
    ) -> Self {
        // Internal model hooks participate in the same observation transaction
        // as ordinary execution. Its outcome records completed/aborted model
        // work; proposal/verification provenance separately remains tentative.
        let session = FundedCaptureSession::from_run_with_lineage(run, lineage);
        Self {
            session,
            source,
            origin,
            role,
        }
    }
    /// Bind the shared outer envelope policy before claiming this single-use
    /// frame. Only the exact source/origin/physical invocation may attach it;
    /// charging occurs in the same cumulative ledger before native callbacks.
    pub fn prepare_envelope(
        &mut self,
        invocation: crate::capture::OriginalSpeculativeCaptureInvocation<'_>,
    ) -> Result<(), CaptureRunHostError> {
        let Some((phase, shape)) = self.session.run.invocation() else {
            return Err(CaptureRunHostError::Coordinate);
        };
        if self.session.run.spent_steps() != 0
            || self.session.envelope_usage.is_some()
            || !self.source.same_source(invocation.source())
            || !self
                .session
                .run
                .matches_intervention_source(invocation.interventions())
            || !self.session.run.matches_intervention_evidence_skips(
                invocation.intervention_evidence_skips(),
            )
            || self.origin != invocation.origin()
            || phase != invocation.capture_phase()
            || shape.sequence != invocation.sequence()
            || self.session.run.invocation_window()
                != invocation.window().map_err(CaptureStepError::from)?
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if let Some(lineage) = invocation.lineage() {
            if !self.session.lineage.same_storage(lineage.ledger()) {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        }
        if let Some(prefix) = invocation.prepared_prefix() {
            prefix
                .validate_invocation(invocation)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
            self.session.prepared_prefix = Some(prefix.clone());
        }
        self.session.envelope_usage = Some(
            crate::capture::speculative::envelope_usage(
                invocation.admission_identity().len(),
                invocation.prefill_span().is_some(),
            )
            .map_err(CaptureStepError::from)?,
        );
        Ok(())
    }
    /// Actual consumed model occurrence. The native producer must compare it
    /// with the active equation and its original allocator/completion owners.
    pub fn role(&self) -> &OriginalEmbeddedSpeculativeRole {
        &self.role
    }
    /// Exact immutable C source, independent of model role or host capacity.
    pub fn source(&self) -> &OriginalCaptureSource {
        &self.source
    }
    /// Shared driver's logical attribution; native shape remains in the bank.
    pub fn origin(&self) -> SpeculativeActivationOrigin {
        self.origin
    }
    /// Lend the existing observer under its cumulative ledger and fixed claims.
    /// The caller must preserve the same execution/transaction closure, supply a
    /// qualified original native backend, and retire work before draining output.
    pub fn with_observer<T, E, N, R>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        map_error: &dyn Fn(FundedCaptureError<E>) -> N,
        operation: impl FnOnce(&mut dyn crate::ActivationObserver<T, N>) -> R,
    ) -> Result<R, CaptureProtocolError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.session
            .with_observer(backend, self.origin.prediction as u64, map_error, operation)
    }
    /// Drain only after enclosing native recovery/record retirement. Failed
    /// observation frames retain their existing Aborted outcome and custody.
    pub fn take_shared_step(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, FundedCaptureDrainError> {
        self.session.take_shared_step()
    }
    /// Drain only already sealed host failure evidence. This certifies no
    /// native completion; the enclosing Recovery retains all unresolved roots.
    pub fn take_failed_evidence(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, FundedCaptureDrainError> {
        self.session.take_failed_numerical_evidence()
    }
    /// Monotone usage, including failed invocations and source callbacks.
    pub fn usage(&self) -> CaptureUsage {
        self.session.usage()
    }
}
