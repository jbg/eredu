//! Speculative discovery against retained layout and native transport facts.

use crate::{
    capture::partition::{PartitionCaptureDiscoveryLayout, PartitionCaptureHookTransport},
    speculative::SpeculativeActivationExecution,
};
use eredu_core::communication::CommunicationManifest;
use eredu_core::communication::CommunicationSessionIdentity;
use eredu_core::{
    capture::CaptureError, intervention::InterventionMechanisms,
    speculative::SpeculativeActivationDiscovery, CollectiveGroupId, ObservationPoint,
    ObservationSupportStatus,
};
use std::sync::{Arc, OnceLock};

/// Retained model semantics for partitioned speculative capture.
pub trait SpeculativePartitionCaptureDiscovery: Send + Sync {
    /// Compiles exact layouts and captures their semantic support projection.
    fn speculative_layout(
        self: Arc<Self>,
        execution: &SpeculativeActivationExecution,
        max_ranks: usize,
    ) -> Result<Option<SpeculativePartitionLayout>, CaptureError>;

    /// Exact target execution identity.
    fn execution_identity(&self) -> &str;

    /// Exact retained tensor group and selected communication manifest.
    fn tensor_parallel_communication(&self) -> Option<(CollectiveGroupId, &CommunicationManifest)>;
}

type SupportProjection = dyn Fn(
        &InterventionMechanisms,
        &str,
        Option<&str>,
        &mut dyn FnMut(&ObservationPoint) -> ObservationSupportStatus,
    ) -> Result<SpeculativeActivationDiscovery, CaptureError>
    + Send
    + Sync;

/// The same retained layout drives discovery and subsequent capture.
pub struct SpeculativePartitionLayout {
    layout: Arc<dyn PartitionCaptureDiscoveryLayout>,
    project: Box<SupportProjection>,
}

impl SpeculativePartitionLayout {
    /// Couples exact placements to their model-owned support projection.
    pub fn new(
        layout: Arc<dyn PartitionCaptureDiscoveryLayout>,
        project: impl Fn(
                &InterventionMechanisms,
                &str,
                Option<&str>,
                &mut dyn FnMut(&ObservationPoint) -> ObservationSupportStatus,
            ) -> Result<SpeculativeActivationDiscovery, CaptureError>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            layout,
            project: Box::new(project),
        }
    }

    /// Provides invocation placements to the capture observer.
    pub fn layout(&self) -> Arc<dyn PartitionCaptureDiscoveryLayout> {
        Arc::clone(&self.layout)
    }

    /// Combines selected producer ownership with native collector facts.
    pub fn discover(
        &self,
        transport: &impl PartitionCaptureHookTransport,
        mechanisms: &InterventionMechanisms,
        session_identity: &str,
        overlay: Option<&str>,
    ) -> Result<SpeculativeActivationDiscovery, CaptureError> {
        super::super::partition::capture_wait(transport)?;
        transport.estimate_capture_gather(16)?;
        let mut discovery = (self.project)(mechanisms, session_identity, overlay, &mut |point| {
            let Some(members) = self.layout.capture_hook_members(&point.path) else {
                return ObservationSupportStatus::Unverified(
                    "prediction point has no retained producer ownership".into(),
                );
            };
            match super::super::partition::estimate_hook(transport, &members) {
                Ok(_) => ObservationSupportStatus::Supported,
                Err(error) => ObservationSupportStatus::Unverified(error.to_string()),
            }
        })?;
        discovery.captures.support.capture.conditions.push("Every model rank participates in the same invocation plan; committed receipts retain independent prediction and physical sequence coordinates".into());
        Ok(discovery)
    }
}

/// Per-loaded-session lazy layout compilation, shared by speculative requests.
#[derive(Clone, Default)]
pub struct SpeculativePartitionLayouts(
    Arc<OnceLock<Result<Arc<SpeculativePartitionLayout>, CaptureError>>>,
);

impl SpeculativePartitionLayouts {
    /// Resolves placement once and checks agreement with the realized setup.
    pub fn resolve(
        &self,
        prepared: Option<Arc<dyn SpeculativePartitionCaptureDiscovery>>,
        execution: Option<&SpeculativeActivationExecution>,
        setup: CommunicationSessionIdentity,
    ) -> Result<Arc<SpeculativePartitionLayout>, CaptureError> {
        self.0
            .get_or_init(|| {
                let prepared = prepared.ok_or_else(|| {
                    CaptureError::Unsupported("session has no retained activation catalog".into())
                })?;
                let execution = execution.ok_or_else(|| {
                    CaptureError::Unsupported(
                        "selected prediction executor has no complete internal activation path"
                            .into(),
                    )
                })?;
                let layout = prepared
                    .speculative_layout(execution, 1024)?
                    .ok_or_else(|| {
                        CaptureError::Invalid(
                            "distributed prediction has no retained placement".into(),
                        )
                    })?;
                if layout.layout.capture_world_size() != setup.participant_count() {
                    return Err(CaptureError::Invalid(
                        "prediction placement differs from native communication setup".into(),
                    ));
                }
                Ok(Arc::new(layout))
            })
            .clone()
    }
}
