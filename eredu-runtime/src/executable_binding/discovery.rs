//! Architecture projections retained by the executable and its sessions.

use crate::{
    capture::{
        partition::{LoadedPartitionCapture, PartitionCaptureCollectorFacts},
        SpeculativePartitionCaptureDiscovery,
    },
    memory_forecast::LoadedMemoryGeometry,
    prediction_resources::EmbeddedPredictionTopology,
    speculative::{ExternalPredictionTargetOperationKind, SpeculativeActivationExecution},
    ForwardCaptureBundle, ParameterPartitionLayout, PredictionTensorAssemblyPlan,
    ReplicatedTextMaterializationTask,
};
use eredu_core::communication::CommunicationSessionIdentity;
use eredu_core::{
    capture::{CaptureDeclarations, CaptureDiscovery, CaptureError, CaptureReservation},
    intervention::{InterventionDiscovery, InterventionMechanisms},
    parameters::ParameterError,
    resources::ResourceDescriptionError,
    speculative::SpeculativeActivationDiscovery,
    BackendFailure, CapabilityError,
};

/// Retained target activation placement for one ordered assistant capture.
pub trait ExternalTargetCapturePlacement {
    /// Resolves the selected producer layout against the invocation's shape.
    fn resolve(&self, decoder_shape: [u64; 2]) -> Result<ForwardCaptureBundle, CaptureError>;

    /// Rank that owns this retained layout.
    fn rank(&self) -> usize;
}

/// Retained static target operation placement for an external assistant.
pub trait ExternalTargetStaticPlacement {
    /// Resolves one invocation without reading native tensors or source bytes.
    fn resolve(
        &self,
        kind: ExternalPredictionTargetOperationKind,
        input_shape: &[i32],
    ) -> Result<PredictionTensorAssemblyPlan, eredu_nn::Error>;
}

/// Portable projections exposed by a fully checked executable.
pub trait LoadedModelDiscovery: SpeculativePartitionCaptureDiscovery {
    /// Cold capture declaration without resolving artifact content identity.
    fn capture_declarations(&self) -> CaptureDeclarations;

    /// Capture report coupled to the exact artifact content identity.
    fn capture(&self) -> Result<CaptureDiscovery, CaptureError>;

    /// Whether exact content identity has already been resolved.
    fn identity_is_resolved(&self) -> bool;

    /// Exact intervention declarations with the supplied mechanism facts.
    fn intervention(
        &self,
        mechanisms: &InterventionMechanisms,
    ) -> Result<InterventionDiscovery, CaptureError>;

    /// Speculative activation support for local execution.
    fn speculative_activations(
        &self,
        execution: &SpeculativeActivationExecution,
        mechanisms: &InterventionMechanisms,
        session_identity: &str,
        overlay: Option<&str>,
    ) -> Result<SpeculativeActivationDiscovery, CaptureError>;

    /// Combines selected ordinary partition layouts and collector facts.
    fn bind_partition_capture(
        &self,
        collector: &dyn PartitionCaptureCollectorFacts,
        setup: CommunicationSessionIdentity,
    ) -> Result<LoadedPartitionCapture, CaptureError>;

    /// Loaded generation memory geometry.
    fn generation_memory(&self) -> Result<&LoadedMemoryGeometry, CapabilityError>;

    /// Embedded prediction geometry selected for the model.
    fn embedded_prediction_topology(
        &self,
    ) -> Result<Option<&EmbeddedPredictionTopology>, ResourceDescriptionError>;

    /// Global target and auxiliary tasks retained by the partition.
    fn partition_parameter_tasks(
        &self,
    ) -> Option<(
        &[ReplicatedTextMaterializationTask],
        &[ReplicatedTextMaterializationTask],
    )>;

    /// Resolves rank-local parameter ownership under prepaid accounting.
    fn parameter_partition_layout_for_rank(
        &self,
        parameter: &str,
        global_rank: usize,
        reservation: &mut dyn CaptureReservation,
    ) -> Result<Option<ParameterPartitionLayout>, ParameterError>;

    /// Prepares target capture producers for the specified ordered paths.
    fn prepare_target_capture_placement(
        &self,
        paths: &[String],
    ) -> Result<Option<Box<dyn ExternalTargetCapturePlacement>>, BackendFailure>;

    /// Prepares exact static parameter owners for an assistant operation.
    fn prepare_target_static_placement(
        &self,
        kind: ExternalPredictionTargetOperationKind,
    ) -> Result<Option<Box<dyn ExternalTargetStaticPlacement>>, eredu_nn::Error>;
}
