//! Loaded partition capture admission over neutral semantic and transport facts.

use super::{PartitionCaptureHookTransport, PartitionCaptureIdentity, PartitionCaptureLayout};
use eredu_core::communication::CommunicationSessionIdentity;

use eredu_core::{
    artifact::DeferredArtifactIdentity,
    capture::{CaptureDeclarations, CaptureDiscovery, CaptureError, CaptureTransformKind},
    ObservationPoint, ObservationSupportStatus,
};

/// Retained invocation geometry needed for cold capture support.
pub trait PartitionCaptureDiscoveryLayout:
    PartitionCaptureLayout + crate::intervention::PartitionActivationLayout + Send + Sync + 'static
{
    /// Exact world size selected for the observation placements.
    fn capture_world_size(&self) -> usize;

    /// Invocation-group members that need exact hook failure agreement.
    fn capture_hook_members(&self, path: &str) -> Option<Vec<usize>>;
}

/// Side-effect-free facts needed to bind a retained partition capture layout.
pub trait PartitionCaptureCollectorFacts {
    /// Checks whether bounded native completion is available.
    fn check_capture_wait(&self) -> Result<(), CaptureError>;

    /// Checks whether the native receipt gather has a bound.
    fn check_capture_gather(&self, words: usize) -> Result<(), CaptureError>;

    /// Checks whether hook agreement is bounded for the selected participants.
    fn check_capture_hook(&self, members: &[usize]) -> Result<(), CaptureError>;
}

impl<T: PartitionCaptureHookTransport> PartitionCaptureCollectorFacts for T {
    fn check_capture_wait(&self) -> Result<(), CaptureError> {
        super::capture_wait(self).map(drop)
    }

    fn check_capture_gather(&self, words: usize) -> Result<(), CaptureError> {
        self.estimate_capture_gather(words).map(drop)
    }

    fn check_capture_hook(&self, members: &[usize]) -> Result<(), CaptureError> {
        super::estimate_hook(self, members).map(drop)
    }
}

/// Portable projection of architecture declarations for distributed capture.
pub trait PartitionCaptureDiscovery {
    /// Exact rank-local layouts produced by the retained selection.
    type Layout: PartitionCaptureDiscoveryLayout;

    /// Compiles the retained layouts within the supplied rank bound.
    fn capture_layout(&self, max_ranks: usize) -> Result<Option<Self::Layout>, CaptureError>;

    /// Combines retained observation declarations with the supplied collector facts.
    fn project_capture_support(
        &self,
        layout: &Self::Layout,
        collector: &mut dyn FnMut(&ObservationPoint) -> ObservationSupportStatus,
    ) -> Result<CaptureDeclarations, CaptureError>;

    /// Returns the deferred content identity without hashing checkpoint data.
    fn capture_artifact_identity(&self) -> &DeferredArtifactIdentity;

    /// Returns the exact retained execution identity.
    fn capture_execution_identity(&self) -> &str;

    /// Binds one portable capture projection to its native transport facts.
    fn bind_partition_capture(
        &self,
        transport: &dyn PartitionCaptureCollectorFacts,
        setup: CommunicationSessionIdentity,
    ) -> Result<LoadedPartitionCapture, CaptureError> {
        let layouts = self.capture_layout(1024)?.ok_or_else(|| {
            CaptureError::Invalid(
                "distributed capture has no retained architecture placement".into(),
            )
        })?;
        if layouts.capture_world_size() != setup.participant_count() {
            return Err(CaptureError::Invalid(
                "capture placement differs from native setup".into(),
            ));
        }
        transport.check_capture_wait()?;
        transport.check_capture_gather(16)?;
        let mut declarations = self.project_capture_support(&layouts, &mut |point| {
            let Some(members) = layouts.capture_hook_members(&point.path) else {
                return ObservationSupportStatus::Unverified(
                    "global observation ownership is not yet implemented for this point".into(),
                );
            };
            match transport.check_capture_hook(&members) {
                Ok(_) => ObservationSupportStatus::Supported,
                Err(error) => ObservationSupportStatus::Unverified(error.to_string()),
            }
        })?;
        declarations.support.capture.transformations.retain(|kind| {
            matches!(
                kind,
                CaptureTransformKind::Preview
                    | CaptureTransformKind::Slice
                    | CaptureTransformKind::FullTensor
                    | CaptureTransformKind::Summary
                    | CaptureTransformKind::Histogram
                    | CaptureTransformKind::TokenScores
                    | CaptureTransformKind::TopCandidates
                    | CaptureTransformKind::RoutedUnits
            )
        });
        declarations.support.capture.conditions = vec![
            "Component, normalized-input and readout observations with retained global ownership and independent invocation-group failure agreement; one token-ID sequence".into(),
            "Every model rank participates in the same capture plan and forward; records publish after shared completion and commit".into(),
            "Vocabulary reductions require the complete ordered final logits on the retained publication owner; probabilities normalize the full model vocabulary".into(),
            "Global producer, transport, decoder and assembly credits are reserved before model work; restore refunds neither credits nor epochs".into(),
            "Skip-on-limit decisions and consumed credits are agreed before model work; record envelopes and coordination remain mandatory".into(),
            "Logical storage bounds exclude native private allocator workspace; deferred projection-input factories use a conservative native bound".into(),
        ];
        Ok(LoadedPartitionCapture {
            declarations,
            artifact: self.capture_artifact_identity().clone(),
            execution: self.capture_execution_identity().into(),
            setup,
            layouts: Box::new(layouts),
        })
    }
}

/// Portable capture declarations coupled to the exact retained execution and layout.
pub struct LoadedPartitionCapture {
    declarations: CaptureDeclarations,
    artifact: DeferredArtifactIdentity,
    execution: String,
    setup: CommunicationSessionIdentity,
    layouts: Box<dyn PartitionCaptureDiscoveryLayout>,
}

impl LoadedPartitionCapture {
    /// Capture support without resolving checkpoint content identity.
    pub const fn declarations(&self) -> &CaptureDeclarations {
        &self.declarations
    }

    /// Exact retained observation placements.
    pub fn layouts(&self) -> &dyn PartitionCaptureDiscoveryLayout {
        self.layouts.as_ref()
    }

    /// Participant count for the selected native communication setup.
    pub fn participant_count(&self) -> usize {
        self.setup.participant_count()
    }

    /// Resolves content identity only for callers that request it.
    pub fn content_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        Ok(CaptureDiscovery {
            artifact_identity: self
                .artifact
                .resolve()
                .map_err(|error| CaptureError::Invalid(error.to_string()))?
                .to_string(),
            catalog: self.declarations.catalog.clone(),
            support: self.declarations.support.clone(),
        })
    }

    /// Binds this exact capture setup to a snapshot-compatible parameter overlay.
    pub fn identity(
        &self,
        content_exact: bool,
        overlay: Option<&str>,
    ) -> Result<PartitionCaptureIdentity, CaptureError> {
        PartitionCaptureIdentity::for_session(
            if content_exact {
                Some(self.content_discovery()?.artifact_identity)
            } else {
                None
            },
            self.execution.clone(),
            self.setup,
            overlay.map(str::to_owned),
        )
    }
}
