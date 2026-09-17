//! Speculative phase composition over the ordinary capture/intervention owner.
use super::*;
use crate::inspection::SpeculativeActivationObserver;
use eredu_core::intervention::InterventionBackend;
use eredu_core::speculative::{
    SpeculativeActivationCapture, SpeculativeActivationOrigin, SpeculativeActivationPhase,
    SpeculativeControlError,
};
use std::{cell::RefCell, collections::VecDeque};

mod control;
mod original;
pub use original::{
    OriginalSpeculativeCapture, OriginalSpeculativeCaptureError,
    OriginalSpeculativeCaptureInvocation, OriginalSpeculativeCapturePrefix,
};
mod observer;
mod provider;
mod transport;
pub use transport::SpeculativeCaptureErrorTransport;
pub(in crate::capture) mod reductions;
pub use control::{PreparedSpeculativeActivationRestore, SpeculativeActivationCheckpoint};
pub use provider::PartitionCaptureBackendProvider;

// Shared logical envelope policy. Physical destination/queue H is admitted
// independently; the same CaptureLedger charges this before native callbacks.
pub(super) fn envelope_usage(
    identity_bytes: usize,
    prefill: bool,
) -> Result<CaptureUsage, CaptureError> {
    Ok(CaptureUsage {
        host_bytes: add(
            mul(
                std::mem::size_of::<SpeculativeActivationCapture>() as u64,
                2,
            )?,
            u64::try_from(identity_bytes).map_err(|_| CaptureError::Overflow)?,
        )?,
        encoded_bytes: if prefill { 1536 } else { 1024 },
        ..Default::default()
    })
}

/// Lends native capture primitives for the duration of a synchronous callback.
/// The provider owns its context; model completion remains with the executor.
pub trait CaptureBackendProvider {
    /// Native value borrowed at the observation boundary.
    type Tensor;
    /// Native transformation failure, translated by the composition adapter.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Existing implementation of capture and intervention primitives.
    type Backend<'a>: InterventionBackend<Tensor = Self::Tensor, Error = Self::Error>
    where
        Self: 'a;
    /// Lends primitives without submitting work or allocating native values.
    fn backend(&mut self) -> Self::Backend<'_>;
    /// Binds an inactive capture owner before invocation metadata is reserved.
    /// The default provider is local; partition providers validate exact retained
    /// setup/model identity here without native work or communication.
    fn bind_session(&self, session: &mut CaptureSession) -> Result<(), CaptureError> {
        if session.partition.is_some() {
            return Err(CaptureError::Invalid(
                "local provider cannot bind a partition capture owner".into(),
            ));
        }
        Ok(())
    }

    /// Lends one observer for the entire invocation, including preparation,
    /// coordination, hooks, completion and final commit. Implementations call
    /// `operation` exactly once and drop borrowed work before returning.
    fn with_observer<E>(
        &mut self,
        session: &mut CaptureSession,
        map_error: impl Fn(CaptureExecutionError<Self::Error>) -> E,
        routed_error: &dyn Fn(eredu_nn::Error) -> eredu_nn::Error,
        operation: &mut dyn FnMut(
            &mut dyn crate::ActivationObserver<Self::Tensor, E>,
        ) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut observer =
            crate::intervention::CaptureObserver::new(session, self.backend(), map_error)
                .with_routed_error_handler(routed_error);
        operation(&mut observer)
    }
}

pub use eredu_core::speculative::SpeculativeCaptureScope;

/// One explicitly admitted scheduler request. Every internal invocation shares
/// its capture/intervention ledger. A bounded queue bridges multiple forwards
/// within one scheduler action; all envelopes are reserved before native work.
pub struct SpeculativeCaptureObserver<P: CaptureBackendProvider, F> {
    control_owner: std::sync::Arc<()>,
    admitted: Option<eredu_core::speculative::AdmittedSpeculativeActivations>,
    estimator: Option<std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>>,
    admission_identity: Option<String>,
    provider: P,
    map_error: F,
    request: eredu_core::SpeculativeRequestId,
    capture_scopes: Vec<SpeculativeCaptureScope>,
    intervention_scopes: Vec<SpeculativeCaptureScope>,
    origin: Option<SpeculativeActivationOrigin>,
    prefill_span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
    reduction_geometry: Option<eredu_core::speculative::SpeculativePrefillReductionGeometry>,
    reductions: Option<reductions::WindowReductions>,
    held_prefill: Option<SpeculativeActivationCapture>,
    active: Option<(u64, SpeculativeActivationOrigin, SpeculativeActivationPhase)>,
    next_invocation: u64,
    records: VecDeque<SpeculativeActivationCapture>,
    failure: RefCell<Option<SpeculativeControlError>>,
    routed_path: Option<String>,
    // Last: shared capture custody covers the observer's host payload lifetime.
    session: CaptureSession,
}

impl<P: CaptureBackendProvider, F> SpeculativeCaptureObserver<P, F> {
    /// Constructs a fresh request collector from immutable loaded authority.
    /// Empty authority preserves an absent observer. Existing native estimators
    /// preflight the shared capture/edit plan before any callback can execute.
    pub fn from_admitted(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        provider: P,
        map_error: F,
        request: eredu_core::SpeculativeRequestId,
        estimator: std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
    ) -> Result<Option<Self>, CaptureError> {
        if plan.is_empty() {
            return Ok(None);
        }
        crate::intervention::preflight(plan.captures(), plan.interventions(), &*estimator)?;
        let mut slot = None;
        crate::intervention::install_session(
            &mut slot,
            plan.captures().clone(),
            (!plan.interventions().is_empty())
                .then(|| (plan.interventions().clone(), estimator.clone())),
        )?;
        let mut observer = Self::new(
            slot.ok_or_else(|| {
                CaptureError::Invalid("nonempty speculative authority has no capture owner".into())
            })?,
            provider,
            map_error,
            request,
            plan.capture_scopes().to_vec(),
            plan.intervention_scopes().to_vec(),
        )?;
        observer.admission_identity = Some(plan.identity().into());
        observer.admitted = Some(plan.clone());
        observer.estimator = Some(estimator);
        Ok(Some(observer))
    }

    /// Binds architecture-declared applicability in admitted order. The caller
    /// must supply the exact scope for every selected capture and operation.
    pub fn new(
        mut session: CaptureSession,
        provider: P,
        map_error: F,
        request: eredu_core::SpeculativeRequestId,
        capture_scopes: Vec<SpeculativeCaptureScope>,
        intervention_scopes: Vec<SpeculativeCaptureScope>,
    ) -> Result<Self, CaptureError> {
        let bounds = session.plan.invocation_bounds().ok_or_else(|| {
            CaptureError::Invalid("speculative capture requires invocation admission".into())
        })?;
        if bounds.batch != 1 {
            return Err(CaptureError::Invalid(
                "one speculative lane requires batch one".into(),
            ));
        }
        if session.records.is_some() || session.transaction.is_some() {
            return Err(CaptureError::Invalid(
                "speculative collector requires an inactive capture owner".into(),
            ));
        }
        if capture_scopes.len() != session.plan.points().len()
            || intervention_scopes.len()
                != session.intervention_plan().map_or(0, |p| p.points().len())
        {
            return Err(CaptureError::Invalid(
                "speculative scopes differ from admission".into(),
            ));
        }
        provider.bind_session(&mut session)?;
        Ok(Self {
            control_owner: std::sync::Arc::new(()),
            admitted: None,
            estimator: None,
            admission_identity: None,
            session,
            provider,
            map_error,
            request,
            capture_scopes,
            intervention_scopes,
            origin: None,
            prefill_span: None,
            reduction_geometry: None,
            reductions: None,
            held_prefill: None,
            active: None,
            next_invocation: 0,
            records: VecDeque::new(),
            failure: RefCell::new(None),
            routed_path: None,
        })
    }

    /// Borrows authority and cumulative accounting without exposing native values.
    pub fn session(&self) -> &CaptureSession {
        &self.session
    }

    fn drained_boundary(&self) -> Result<(), CaptureError> {
        if self.origin.is_some()
            || self.active.is_some()
            || !self.records.is_empty()
            || self.reductions.is_some()
            || self.held_prefill.is_some()
            || self.reduction_geometry.is_some()
        {
            return Err(CaptureError::Invalid(
                "speculative capture is not at a drained boundary".into(),
            ));
        }
        Ok(())
    }

    /// Saves the existing immutable authority at a drained operation boundary.
    pub fn checkpoint(
        &self,
        discovery: &CaptureDiscovery,
    ) -> Result<CaptureCheckpoint, CaptureError> {
        self.drained_boundary()?;
        self.session.checkpoint(discovery)
    }

    /// Restores the shared owner's admitted state without refunding allowances,
    /// changing architecture scopes or rewinding the invocation identity.
    pub fn restore(&mut self, checkpoint: &CaptureCheckpoint) -> Result<(), CaptureError> {
        self.drained_boundary()?;
        self.session.restore(checkpoint)
    }

    fn admit(
        &mut self,
        phase: SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), CaptureError> {
        if self.active.is_some() {
            return Err(CaptureError::Invalid(
                "speculative activation already active".into(),
            ));
        }
        let origin = self.origin.ok_or_else(|| {
            CaptureError::Invalid("speculative activation has no scheduler origin".into())
        })?;
        if origin.request != self.request || origin.prediction < origin.committed_tokens {
            return Err(CaptureError::Invalid(
                "speculative activation origin differs from admission".into(),
            ));
        }
        let next = self
            .next_invocation
            .checked_add(1)
            .ok_or(CaptureError::Overflow)?;
        let captures: Vec<_> = self
            .capture_scopes
            .iter()
            .map(|s| s.applies(phase))
            .collect();
        let interventions: Vec<_> = self
            .intervention_scopes
            .iter()
            .map(|s| s.applies(phase))
            .collect();
        let capture_phase = match phase {
            SpeculativeActivationPhase::TargetPrefill
            | SpeculativeActivationPhase::PredictionPrefill => CapturePhase::Prefill,
            _ => CapturePhase::Decode,
        };
        let physical = CaptureInvocationShape {
            batch: 1,
            sequence: u64::try_from(sequence).map_err(|_| CaptureError::Overflow)?,
            context: None,
        };
        let selection = CaptureInvocationSelection {
            captures: Some(&captures),
            interventions: Some(&interventions),
        };
        let prediction = u64::try_from(origin.prediction).map_err(|_| CaptureError::Overflow)?;
        if let Some(span) = self.prefill_span {
            if !span.validate(phase, sequence) {
                return Err(CaptureError::Invalid(
                    "speculative prefill span differs from actual invocation".into(),
                ));
            }
            let window = invocation_window(phase, Some(span))?.expect("present span");
            self.session.begin_invocation_window(
                capture_phase,
                prediction,
                physical,
                selection,
                window,
            )?;
        } else {
            self.session
                .begin_invocation(capture_phase, prediction, physical, selection)?;
        }
        // Includes queue capacity growth and the fixed-size provenance envelope.
        // Value records and their vectors are already charged by CaptureSession.
        let envelope = envelope_usage(
            self.admission_identity.as_ref().map_or(0, |id| id.len()),
            self.prefill_span.is_some(),
        )?;
        if let Some(CaptureSkipReason::Limit { budget, cumulative }) =
            self.session.ledger.reserve(envelope)?
        {
            return Err(CaptureError::Limit { budget, cumulative });
        }
        self.records.reserve(if self.reduction_geometry.is_some() {
            2
        } else {
            1
        });
        if let (Some(geometry), Some(span)) = (self.reduction_geometry, self.prefill_span) {
            if self.reductions.is_none() {
                self.reductions = reductions::WindowReductions::create(
                    &mut self.session,
                    &self.capture_scopes,
                    &self.intervention_scopes,
                    geometry,
                    origin,
                    self.next_invocation,
                )?;
            }
            if let Some(group) = &mut self.reductions {
                group.begin(&mut self.session, phase, span, origin, geometry)?;
            }
        }
        self.active = Some((self.next_invocation, origin, phase));
        self.next_invocation = next;
        self.routed_path = None;
        Ok(())
    }
}

fn signal<E: std::error::Error + Send + Sync + 'static, X>(
    failure: &RefCell<Option<SpeculativeControlError>>,
    map: &impl SpeculativeCaptureErrorTransport<E, X>,
    error: CaptureExecutionError<E>,
) -> X {
    let error = match map.take_retained(error) {
        Ok((signal, control)) => {
            let mut stored = failure.borrow_mut();
            if stored.is_none() {
                *stored = Some(control);
            }
            // A later control alias retires after the cell loan. Its matching
            // execution owner remains alive in the returned signal.
            drop(stored);
            return signal;
        }
        Err(error) => error,
    };
    let signal = map.ordinary(&error);
    let mut stored = failure.borrow_mut();
    if stored.is_none() {
        *stored = Some(match error {
            CaptureExecutionError::Admission(error) => SpeculativeControlError::Capture(error),
            CaptureExecutionError::Backend(error) => SpeculativeControlError::backend(error),
        });
    }
    signal
}

impl<P, F, E> SpeculativeActivationObserver<P::Tensor, E> for SpeculativeCaptureObserver<P, F>
where
    P: CaptureBackendProvider,
    F: SpeculativeCaptureErrorTransport<P::Error, E>,
{
    fn activation_checkpoint_bytes(&self) -> Option<u64> {
        self.control_storage_bytes()
    }
    fn activation_checkpoint(
        &self,
    ) -> Result<SpeculativeActivationCheckpoint, SpeculativeControlError> {
        self.save_control().map_err(Into::into)
    }
    fn prepare_activation_restore<'a>(
        &'a mut self,
        saved: &SpeculativeActivationCheckpoint,
    ) -> Result<Box<dyn PreparedSpeculativeActivationRestore + 'a>, SpeculativeControlError> {
        self.prepare_control(saved).map_err(Into::into)
    }
    fn readmit_activation_interventions(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        self.readmit_control(plan).map_err(Into::into)
    }
    fn set_prefill_span(&mut self, span: Option<eredu_core::speculative::SpeculativePrefillSpan>) {
        self.prefill_span = span;
    }
    fn set_prefill_reduction_geometry(
        &mut self,
        geometry: eredu_core::speculative::SpeculativePrefillReductionGeometry,
    ) {
        self.reduction_geometry = Some(geometry);
        self.session.window_reductions = true;
    }
    fn complete_prefill_reductions(&mut self) -> Result<(), E> {
        if let Some(group) = &mut self.reductions {
            group
                .seal()
                .map_err(|error| signal(&self.failure, &self.map_error, error.into()))?;
        }
        Ok(())
    }
    fn finish_prefill_reductions(&mut self, success: bool) {
        let had_reductions = self.reductions.is_some();
        let report = self.reductions.take().map(|group| group.finish(success));
        if let Some(mut last) = self.held_prefill.take() {
            last.prefill_reductions = report.map(Into::into);
            // Capacity covers both the previous envelope and this withheld one.
            // Draining earlier records cannot shrink VecDeque capacity.
            debug_assert!(self.records.len() < self.records.capacity());
            self.records.push_back(last);
        }
        self.reduction_geometry = None;
        self.session.window_reductions = false;
        if had_reductions && !success {
            self.session.checkpoint_ready = false;
        }
    }
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.origin = origin;
    }
    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.records.pop_front()
    }
    fn take_activation_error(&mut self) -> Option<SpeculativeControlError> {
        self.failure.borrow_mut().take()
    }
    fn begin_activation_invocation(
        &mut self,
        phase: SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), E> {
        self.admit(phase, sequence).map_err(|error| {
            let error = error.into();
            signal(&self.failure, &self.map_error, error)
        })
    }
    fn with_activation_observer(
        &mut self,
        operation: &mut dyn FnMut(
            &mut dyn crate::ActivationObserver<P::Tensor, E>,
        ) -> Result<(), E>,
    ) -> Result<(), E> {
        let failure = &self.failure;
        let map = &self.map_error;
        self.provider.with_observer(
            &mut self.session,
            |error| signal(failure, map, error),
            &|error| observer::retain_sparse::<P::Error>(failure, error),
            operation,
        )
    }
    fn complete_activation_invocation(&mut self) -> Result<(), E> {
        if self.active.is_none()
            || self
                .session
                .transaction
                .is_some_and(|(_, status)| status != CaptureTransactionStatus::Committed)
        {
            let error = CaptureError::Invalid(
                "speculative activation has no successfully finished forward".into(),
            )
            .into();
            return Err(signal(&self.failure, &self.map_error, error));
        }
        crate::ActivationObserver::finish(self)?;
        if let Some(group) = &mut self.reductions {
            group
                .prepare(
                    self.session
                        .records
                        .as_deref()
                        .expect("active capture records"),
                    self.session
                        .interventions
                        .as_ref()
                        .and_then(|run| run.records.as_deref())
                        .unwrap_or(&[]),
                )
                .map_err(|error| signal(&self.failure, &self.map_error, error.into()))?;
        }
        Ok(())
    }
    fn finish_activation_invocation(&mut self, success: bool) {
        if !success {
            self.session.checkpoint_ready = false;
            if let Some((_, status)) = &mut self.session.transaction {
                *status = CaptureTransactionStatus::Aborted;
            }
        }
        let step = self.session.take_step();
        if let (Some((invocation, origin, phase)), Some(mut captures)) = (self.active.take(), step)
        {
            if !success {
                captures.outcome = CaptureStepOutcome::Aborted;
            }
            if let Some(group) = &mut self.reductions {
                group.finish_window(
                    &captures.records,
                    &captures.interventions,
                    invocation,
                    success,
                );
            }
            let envelope = SpeculativeActivationCapture {
                admission_identity: self.admission_identity.clone(),
                invocation,
                origin,
                phase,
                prefill_span: self.prefill_span,
                completed: success,
                captures: eredu_core::capture::CapturedStepDelivery::Legacy(captures),
                prefill_reductions: None,
            };
            if self.reductions.is_some() {
                if let Some(previous) = self.held_prefill.replace(envelope) {
                    self.records.push_back(previous);
                }
            } else {
                self.records.push_back(envelope);
            }
        }
        // Rejected admission can leave exact geometry with no record batch.
        self.session.invocation = None;
        self.routed_path = None;
    }
}

// The shared scheduler owns shifted seed coordinates. Both collectors consume
// this exact projection; a span/shape never grants source or occurrence authority.
pub(super) fn invocation_window(
    phase: SpeculativeActivationPhase,
    span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
) -> Result<Option<CaptureInvocationWindow>, CaptureError> {
    span.map(|span| {
        let shifted = phase == SpeculativeActivationPhase::PredictionPrefill
            && span.token_start != span.hidden_start;
        Ok(CaptureInvocationWindow {
            logical_sequence: span
                .prompt_tokens
                .checked_sub(u64::from(shifted))
                .ok_or(CaptureError::Overflow)?,
            start: span.hidden_start,
        })
    })
    .transpose()
}
