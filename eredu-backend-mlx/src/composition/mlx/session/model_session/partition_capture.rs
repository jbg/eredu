//! Native composition of retained architecture layouts with the shared collector.
use super::*;
use eredu_core::capture::*;

impl TextCaptureNative<MlxRetainedSession> for MlxSessionMechanisms {
    type Transport = MlxDistributedSession;
    type Collector = MlxCaptureCollector;

    fn intervention_estimator() -> impl eredu_core::intervention::InterventionEstimator + 'static {
        super::super::intervention::NativeInterventionEstimator
    }

    fn capture_transport(
        retained: &MlxRetainedSession,
    ) -> Option<(
        &Self::Transport,
        eredu_core::communication::CommunicationSessionIdentity,
    )> {
        let transport = retained.distributed.as_ref()?;
        Some((transport, transport.session_identity()))
    }

    fn intervention_mechanisms() -> eredu_core::intervention::InterventionMechanisms {
        super::super::intervention::mechanisms()
    }
}

impl CaptureWorkspace for MlxSessionMechanisms {
    fn capture_workspace() -> CaptureWorkspaceModel {
        super::super::bounded_capture::CAPTURE_WORKSPACE
    }
}

pub(in crate::composition::mlx::session) fn generated_creation_bytes(
    shape: &[u64],
) -> Result<u64, CaptureError> {
    let width = shape
        .last()
        .copied()
        .ok_or_else(|| CaptureError::Invalid("scalar observation capture".into()))?;
    let rows = elements(&shape[..shape.len() - 1])?;
    crate::backend::nn::fp8::projection_input_capture_storage(elements(shape)?, rows, width)
        .ok_or(CaptureError::Overflow)
}

#[doc(hidden)]
pub struct MlxCaptureCollector;

impl CaptureWorkspace for MlxCaptureCollector {
    fn capture_workspace() -> CaptureWorkspaceModel {
        super::super::bounded_capture::CAPTURE_WORKSPACE
    }
}

impl CaptureTensorOperations<MlxDistributedSession> for MlxCaptureCollector {
    type Context = Stream;
    type Tensor = MlxTensor;
    type Error = Error;
    type Native<'a> = super::super::bounded_capture::NativeCapture<'a>;

    fn native<'a>(
        stream: &'a Stream,
        domain: Option<CaptureTokenDomain<'a>>,
        partition: Option<&'a MlxDistributedSession>,
    ) -> Self::Native<'a> {
        super::super::bounded_capture::NativeCapture {
            stream,
            domain,
            partition,
        }
    }

    fn generated_creation_bytes(shape: &[u64]) -> Result<u64, CaptureError> {
        generated_creation_bytes(shape)
    }
}
