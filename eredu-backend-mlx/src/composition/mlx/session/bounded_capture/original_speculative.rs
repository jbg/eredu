//! Source-only outer observer; native phases install their local funded worker.
use crate::{
    MlxTensor,
    backend::error::Error,
    composition::mlx::{
        model::retain_planning_error, speculative::OriginalSpeculativeNumericalSources,
    },
};
use eredu_core::{
    DistributedCommitEpoch, SpeculativeRequestId,
    speculative::{
        SpeculativeActivationCapture, SpeculativeActivationOrigin, SpeculativeActivationPhase,
        SpeculativeCaptureScope, SpeculativeControlError, SpeculativePrefillSpan,
    },
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    ActivationObserver, ExpertPass,
    capture::{
        CaptureProtocolError, OriginalSpeculativeCapture, OriginalSpeculativeCaptureInvocation,
    },
    inspection::SpeculativeActivationObserver,
    working_memory::{OriginalCaptureSource, OriginalInterventionSource},
};
use std::mem::{size_of, size_of_val};

mod control;

struct Observer {
    state: OriginalSpeculativeCapture,
    failure: Option<SpeculativeControlError>,
    control: control::Source,
    // Array-free account alias only, retained after source state and errors.
    funding: HostMetadataFunding,
}
impl Observer {
    fn signal<E: std::error::Error + Send + Sync + 'static>(&mut self, cause: E) -> Error {
        let error = retain_planning_error(cause, self.funding.clone());
        match error.split_retained_original() {
            Ok((signal, control)) => {
                if self.failure.is_none() {
                    self.failure = Some(SpeculativeControlError::Backend(control));
                }
                signal
            }
            // An inline H refusal needs no additional error allocation. The
            // existing driver preserves it when no separately retained alias exists.
            Err(error) => error,
        }
    }
}

/// Compiled/validated declarations enter here once. This constructor performs
/// no native operations and never borrows the ordinary NativeCapture backend.
pub(in crate::composition::mlx) fn prepare(
    source: OriginalCaptureSource,
    interventions: Option<OriginalInterventionSource>,
    intervention_scopes: &[SpeculativeCaptureScope],
    scopes: &[SpeculativeCaptureScope],
    identity: &str,
    request: SpeculativeRequestId,
    sources: &OriginalSpeculativeNumericalSources,
) -> Result<Option<Box<dyn SpeculativeActivationObserver<MlxTensor, Error>>>, Error> {
    let funding = sources.metadata_funding();
    let controls = [
        size_of::<Observer>(),
        size_of::<Option<OriginalInterventionSource>>(),
        size_of::<&[SpeculativeCaptureScope]>(),
        OriginalInterventionSource::validation_control_bytes().ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?,
        size_of::<eredu_runtime::working_memory::OriginalEmbeddedCaptureLineage>(),
        size_of::<
            Result<
                eredu_runtime::working_memory::OriginalEmbeddedCaptureLineage,
                eredu_runtime::working_memory::SpeculativeRequestError,
            >,
        >(),
        size_of::<Box<Observer>>(),
        size_of::<Option<Box<dyn SpeculativeActivationObserver<MlxTensor, Error>>>>(),
        size_of::<Result<Option<Box<dyn SpeculativeActivationObserver<MlxTensor, Error>>>, Error>>(
        ),
        size_of::<
            Result<
                OriginalSpeculativeCapture,
                eredu_runtime::capture::OriginalSpeculativeCaptureError,
            >,
        >(),
        size_of::<Option<SpeculativeControlError>>(),
        size_of::<(
            OriginalCaptureSource,
            &[SpeculativeCaptureScope],
            &str,
            SpeculativeRequestId,
            &OriginalSpeculativeNumericalSources,
        )>(),
        OriginalCaptureSource::validation_control_bytes().ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?,
    ];
    funding
        .reserve_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    source
        .validate_pool(sources.pool())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let lineage = sources
        .request()
        .prepare_embedded_capture_lineage(&source)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let state =
        OriginalSpeculativeCapture::prepare(source, scopes, identity, request, funding.clone())
            .and_then(|state| state.with_lineage(lineage))
            .map_err(|cause| sources.retain_startup_error(cause))?;
    let state = match interventions {
        None => state,
        Some(source) => {
            source
                .validate_pool(sources.pool())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            state
                .with_interventions(source, intervention_scopes)
                .map_err(|cause| sources.retain_startup_error(cause))?
        }
    };
    let control = control::Source::prepare(sources)?;
    Ok(Some(Box::new(Observer {
        state,
        control,
        failure: None,
        funding: funding.clone(),
    })))
}

impl ActivationObserver<MlxTensor, Error> for Observer {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn supports_prefill_spans(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        self.state.invocation().map_or(true, |source| {
            source.requires_sequence_readout().unwrap_or(true)
        })
    }
    fn original_speculative_capture(&self) -> Option<OriginalSpeculativeCaptureInvocation<'_>> {
        self.state.invocation()
    }
    fn retain_original_speculative_capture(
        &mut self,
        capture: SpeculativeActivationCapture,
    ) -> Result<(), CaptureProtocolError> {
        self.state.receive(capture)
    }
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        _: DistributedCommitEpoch,
        _: ExpertPass,
    ) -> Result<(), Error> {
        Err(self.signal(CaptureProtocolError::Invocation))
    }
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        Err(self.signal(CaptureProtocolError::Invocation))
    }
    fn intervene(&mut self, _: &str, _: &MlxTensor) -> Result<Option<MlxTensor>, Error> {
        Err(self.signal(CaptureProtocolError::Invocation))
    }
}
impl SpeculativeActivationObserver<MlxTensor, Error> for Observer {
    fn activation_checkpoint_bytes(&self) -> Option<u64> {
        if self.failure.is_some() { return None; }
        self.state.control_storage_bytes()
    }
    fn activation_checkpoint(&self) -> Result<eredu_runtime::capture::SpeculativeActivationCheckpoint, SpeculativeControlError> {
        if self.failure.is_some() { return Err(SpeculativeControlError::Invalid("failed internal capture boundary")); }
        self.state.save_control().map_err(|cause| SpeculativeControlError::Backend(
            retain_planning_error(cause, self.funding.clone()).into_backend_failure()))
    }
    fn prepare_activation_restore<'a>(&'a mut self, saved: &eredu_runtime::capture::SpeculativeActivationCheckpoint)
        -> Result<Box<dyn eredu_runtime::capture::PreparedSpeculativeActivationRestore + 'a>, SpeculativeControlError>
    {
        if self.failure.is_some() { return Err(SpeculativeControlError::Invalid("failed internal capture boundary")); }
        let funding = self.funding.clone();
        self.state.prepare_control(saved).map_err(move |cause| SpeculativeControlError::Backend(
            retain_planning_error(cause, funding).into_backend_failure()))
    }
    fn validate_activation_readmission(&self, plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        _: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), SpeculativeControlError> {
        if self.failure.is_some() { return Err(SpeculativeControlError::Invalid("failed internal capture boundary")); }
        self.control.validate(&self.state, plan)
            .map_err(|cause| SpeculativeControlError::Backend(cause.into_backend_failure()))
    }
    fn readmit_activation_interventions(&mut self, plan: eredu_core::speculative::AdmittedSpeculativeActivations)
        -> Result<(), SpeculativeControlError>
    {
        if self.failure.is_some() { return Err(SpeculativeControlError::Invalid("failed internal capture boundary")); }
        let (source, inherited) = self.control.candidate(&self.state, &plan)
            .map_err(|cause| SpeculativeControlError::Backend(cause.into_backend_failure()))?;
        let funding = self.funding.clone();
        self.state.prepare_readmitted_control(&plan, source, self.control.pool(), inherited)
            .map_err(move |cause| SpeculativeControlError::Backend(
                retain_planning_error(cause, funding).into_backend_failure()))?.commit();
        Ok(())
    }
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.state.set_origin(origin);
    }
    fn set_prefill_reduction_geometry(
        &mut self,
        geometry: eredu_core::speculative::SpeculativePrefillReductionGeometry,
    ) {
        self.state.set_prefill_reduction_geometry(geometry);
    }
    fn complete_prefill_reductions(&mut self) -> Result<(), Error> {
        self.state
            .complete_prefill_reductions()
            .map_err(|cause| self.signal(cause))
    }
    fn finish_prefill_reductions(&mut self, success: bool) {
        self.state.finish_prefill_reductions(success);
    }
    fn set_prefill_span(&mut self, span: Option<SpeculativePrefillSpan>) {
        self.state.set_prefill_span(span);
    }
    fn begin_activation_invocation(
        &mut self,
        phase: SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), Error> {
        self.state
            .begin(phase, sequence)
            .map_err(|cause| self.signal(cause))
    }
    fn complete_activation_invocation(&mut self) -> Result<(), Error> {
        self.state.complete().map_err(|cause| self.signal(cause))
    }
    fn finish_activation_invocation(&mut self, success: bool) {
        self.state.finish(success);
    }
    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.state.take()
    }
    fn take_activation_error(&mut self) -> Option<SpeculativeControlError> {
        self.failure.take()
    }
}
