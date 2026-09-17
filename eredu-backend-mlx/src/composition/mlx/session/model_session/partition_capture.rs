//! Native composition of retained architecture layouts with the shared collector.
use super::*;
use eredu_core::{capture::*, ObservationSupportStatus};
use eredu_runtime::capture::{partition::*, CaptureSession};
use std::sync::Arc;

pub(super) struct LoadedPartitionCapture {
    pub(super) discovery: CaptureDiscovery,
    execution: String,
    setup: eredu_runtime::CommunicationSessionIdentity,
    layouts: eredu_architectures::component_partition::ComponentPartitionLayouts,
}

impl LoadedPartitionCapture {
    /// Borrow only identities already established by loaded public discovery.
    /// No source resolution, declaration clone or lazy metadata population.
    pub(super) fn layouts(&self) -> &eredu_architectures::component_partition::ComponentPartitionLayouts {
        &self.layouts
    }
    pub(super) fn source_labels(&self) -> (&str, &str, eredu_runtime::CommunicationSessionIdentity) {
        (&self.discovery.artifact_identity, &self.execution, self.setup)
    }
    pub(super) fn identity(
        &self,
        overlay: Option<&str>,
    ) -> Result<PartitionCaptureIdentity, CaptureError> {
        PartitionCaptureIdentity::for_session(
            self.discovery.artifact_identity.clone(),
            self.execution.clone(),
            self.setup,
            overlay.map(str::to_owned),
        )
    }
}

impl MlxModelSession {
    pub(super) fn loaded_partition_capture(
        &self,
    ) -> Result<Option<Arc<LoadedPartitionCapture>>, CaptureError> {
        let Some(transport) = &self.payload.distributed else {
            return Ok(None);
        };
        self.partition_capture.get_or_init(|| {
            let prepared = self.capture_discovery.as_ref().ok_or_else(|| CaptureError::Unsupported(
                "session has no retained capture catalog".into(),
            ))?;
            let layouts = prepared.component_partition_layouts(1024)
                .map_err(|error| CaptureError::Unsupported(error.to_string()))?
                .ok_or_else(|| CaptureError::Invalid("distributed capture has no retained architecture placement".into()))?;
            let setup = transport.session_identity();
            if layouts.topology().world_size() != setup.participant_count() {
                return Err(CaptureError::Invalid("capture placement differs from native setup".into()));
            }
            transport.capture_wait()?;
            transport.estimate_capture_gather(16)?;
            let mut discovery = prepared.capture_with_partition_support(&layouts, |point| {
                let Some(members) = layouts.capture_hook_members(&point.path) else {
                    return ObservationSupportStatus::Unverified(
                        "global observation ownership is not yet implemented for this point".into(),
                    );
                };
                match transport.estimate_capture_hook(&members) {
                    Ok(_) => ObservationSupportStatus::Supported,
                    Err(error) => ObservationSupportStatus::Unverified(error.to_string()),
                }
            })?;
            discovery.support.capture.transformations.retain(|kind| matches!(kind,
                CaptureTransformKind::Preview | CaptureTransformKind::Slice | CaptureTransformKind::FullTensor |
                CaptureTransformKind::Summary | CaptureTransformKind::Histogram |
                CaptureTransformKind::TokenScores | CaptureTransformKind::TopCandidates | CaptureTransformKind::RoutedUnits
            ));
            discovery.support.capture.conditions = vec![
                "Component, normalized-input and readout observations with retained global ownership and independent invocation-group failure agreement; one token-ID sequence".into(),
                "Every model rank participates in the same capture plan and forward; records publish after shared completion and commit".into(),
                "Vocabulary reductions require the complete ordered final logits on the retained publication owner; probabilities normalize the full model vocabulary".into(),
                "Global producer, transport, decoder and assembly credits are reserved before model work; restore refunds neither credits nor epochs".into(),
                "Skip-on-limit decisions and consumed credits are agreed before model work; record envelopes and coordination remain mandatory".into(),
                "Logical storage bounds exclude native private allocator workspace; deferred projection-input factories use a conservative native bound".into(),
            ];
            Ok(Arc::new(LoadedPartitionCapture {
                discovery, execution: prepared.execution_identity().into(), setup, layouts,
            }))
        }).clone().map(Some)
    }
}

pub(super) fn estimate(
    shape: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<PartitionCaptureNativeEstimate, CaptureError> {
    let width = shape
        .last()
        .copied()
        .ok_or_else(|| CaptureError::Invalid("scalar observation capture".into()))?;
    let rows = elements(&shape[..shape.len() - 1])?;
    Ok(PartitionCaptureNativeEstimate {
        capture: super::super::bounded_capture::estimate_shape(shape, selection, slice)?,
        generated_creation_bytes: crate::backend::nn::fp8::projection_input_capture_storage(
            elements(shape)?,
            rows,
            width,
        )
        .ok_or(CaptureError::Overflow)?,
    })
}

fn observer<'a>(
    capture: &'a mut CaptureSession,
    stream: &'a Stream,
    domain: Option<CaptureTokenDomain<'a>>,
    prediction: u64,
    transport: &'a MlxDistributedSession,
    loaded: &'a LoadedPartitionCapture,
    identity: PartitionCaptureIdentity,
) -> impl RuntimeActivationObserver<MlxTensor, Error> + 'a {
    let max_record_bytes = capture.plan().plan().limits.per_step.encoded_bytes;
    PartitionCaptureObserver::for_step(
        capture,
        super::super::bounded_capture::NativeCapture {
            partition: Some(transport),
            stream,
            domain,
        },
        transport,
        &loaded.layouts,
        prediction,
        PartitionCaptureReceiptLimits {
            max_producers: loaded.setup.participant_count(),
            max_fragments: 65_536,
            max_record_bytes,
        },
        estimate,
        |error: PartitionCaptureObserverError<Error>| match error {
            PartitionCaptureObserverError::Capture(error) => {
                super::super::bounded_capture::capture_error(error)
            }
            error => Error::observation(error),
        },
    )
    .with_session_identity(identity)
    .with_interventions()
}

struct RejectedCapture(CaptureError, Option<eredu_core::HostPreparationAuthority>);
impl RuntimeActivationObserver<MlxTensor, Error> for RejectedCapture {
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        Err(Error::observation(self.0.clone()).retain_ordinary_capture(self.1.clone()))
    }
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        Err(Error::observation(self.0.clone()).retain_ordinary_capture(self.1.clone()))
    }
}

/// Keeps native and layout owners alive for the entire shared forward. Any
/// cold composition failure becomes local observer preparation failure so peers
/// can agree rejection before their hooks or mutable model execution.
pub(super) fn with_observer<R>(
    session: &mut MlxModelSession,
    capture: &mut CaptureSession,
    stream: &Stream,
    domain: Option<CaptureTokenDomain<'_>>,
    prediction: u64,
    submit: impl FnOnce(
        &mut MlxModelSession,
        &mut dyn RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<R, Error>,
) -> Result<R, Error> {
    let host = capture.ordinary_error_custody().cloned();
    let loaded = session.loaded_partition_capture();
    let transport = session.payload.distributed.clone();
    let mut observer: Box<dyn RuntimeActivationObserver<MlxTensor, Error> + '_> =
        match (transport.as_ref(), loaded.as_ref()) {
            (None, Ok(None)) => Box::new(super::super::bounded_capture::observer(
                capture, stream, domain, prediction,
            )),
            (Some(transport), Ok(Some(loaded))) => {
                match loaded.identity(session.payload.parameter_state.active.as_deref()) {
                    Ok(identity) => Box::new(observer(
                        capture, stream, domain, prediction, transport, loaded, identity,
                    )),
                    Err(error) => Box::new(RejectedCapture(error, host.clone())),
                }
            }
            (_, Err(error)) => Box::new(RejectedCapture(error.clone(), host.clone())),
            _ => Box::new(RejectedCapture(
                CaptureError::Invalid(
                    "native capture owner and retained placement disagree".into(),
                ),
                host.clone(),
            )),
        };
    let result = submit(session, &mut *observer);
    drop(observer);
    result.map_err(|error| error.retain_ordinary_capture(host))
}

#[cfg(test)]
mod tests;
