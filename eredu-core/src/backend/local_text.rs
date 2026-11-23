//! Uninhabited resources for text targets without distributed control.

use crate::{
    communication::CommunicationSessionIdentity,
    consensus::{BoundedConsensusTransport, ConsensusTransport},
    parameters::{ParameterCoordinationError, ParameterError, ParameterOperationTransport},
    run_preparation::TextPreparationTransport,
    BackendFailure, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    Submission,
};

/// No native distributed transport is attached to a local target.
#[derive(Clone, Copy)]
pub enum NoTextTransport {}

/// Completion of an operation on an uninhabited resource.
pub enum NoTextCompletion {}

impl Completion for NoTextCompletion {
    type Error = BackendFailure;
    fn is_complete(&self) -> Result<bool, BackendFailure> {
        match *self {}
    }
    fn wait(&self) -> Result<(), BackendFailure> {
        match *self {}
    }
}

impl BoundedCompletion for NoTextCompletion {
    fn wait_bounded(
        self,
        _: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, BackendFailure> {
        match self {}
    }
}

impl ConsensusTransport for NoTextTransport {
    type Error = BackendFailure;
    fn participant_count(&self) -> usize {
        match *self {}
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, BackendFailure> {
        match *self {}
    }
}

impl BoundedConsensusTransport for NoTextTransport {
    type Completion = NoTextCompletion;
    type GatherOutput = Self;
    fn submit_all_gather_words(
        &self,
        _: &[u32],
    ) -> Result<Submission<Self, Self::Completion>, BackendFailure> {
        match *self {}
    }
    fn resolve_all_gather_words(&self, _: Self) -> Result<Vec<u32>, BackendFailure> {
        match *self {}
    }
}

impl TextPreparationTransport for NoTextTransport {
    fn preparation_rank(&self) -> usize {
        match *self {}
    }
    fn ensure_preparation_active(&self) -> Result<(), BackendFailure> {
        match *self {}
    }
    fn fence_preparation(&self) {
        match *self {}
    }
}

impl ParameterOperationTransport for NoTextTransport {
    fn parameter_rank(&self) -> usize {
        match *self {}
    }
    fn parameter_setup(&self) -> CommunicationSessionIdentity {
        match *self {}
    }
    fn parameter_wait(&self) -> Result<BoundedCompletionWait, ParameterError> {
        match *self {}
    }
    fn estimate_parameter_gather(
        &self,
        _: usize,
    ) -> Result<crate::capture::CaptureUsage, ParameterError> {
        match *self {}
    }
    fn ensure_parameter_active(&self) -> Result<(), BackendFailure> {
        match *self {}
    }
    fn fail_parameter_operation(&self, _: &ParameterCoordinationError) {
        match *self {}
    }
}

/// Exact score readout from an already settled host-accessible causal invocation.
pub trait HostCausalVerification: crate::BackendProvider {
    /// Native text tensors and the context used by the installed executable.
    type Values: crate::backend::NativeTextValues<
        Error = BackendFailure,
        Scores: Clone + 'static,
        InvocationTensor: 'static,
        Embedded = Self::Embedded,
    >;

    /// Native tensor and completion primitives for an embedded predictor.
    type Embedded: crate::speculative::embedded::EmbeddedPredictionOperators<
        Tensor = <Self::Values as crate::backend::NativeTextValues>::Tensor,
        Logits = <Self::Assistant as crate::speculative::assistant::ExternalAssistantOperators>::Logits,
        Error = BackendFailure,
        Capture = Self::ContextBinding,
    >;

    /// Native tensor and materialization primitives for a conditioned assistant.
    type Assistant: crate::speculative::assistant::ExternalAssistantOperators<
        Text = Self::Values,
        Tensor = <Self::Values as crate::backend::NativeTextValues>::Tensor,
        Error = BackendFailure,
        Capture = Self::ContextBinding,
    >;

    /// Exact native queues used by every local draft.
    type Queues: crate::speculative::SpeculativeQueueMechanisms<
        Queue = <Self::Values as crate::backend::NativeTextValues>::Context,
        Error = BackendFailure,
    >;

    /// Role-specific queues and collector handles carried by local invocations.
    type ContextBinding: crate::capture::SpeculativeCapturePrimitives<
            Queues = Self::Queues,
            Transport = NoTextTransport,
            Tensor = <Self::Values as crate::backend::NativeTextValues>::Tensor,
            InvocationError = BackendFailure,
        > + crate::capture::SpeculativeCaptureTypes<
            Queues = Self::Queues,
            Transport = NoTextTransport,
            InvocationTensor = <Self::Values as crate::backend::NativeTextValues>::InvocationTensor,
            ObserverError = BackendFailure,
            Scores = <Self::Values as crate::backend::NativeTextValues>::Scores,
        > + HostSpeculativeScoreRead<
            Logits = <Self::Assistant as crate::speculative::assistant::ExternalAssistantOperators>::Logits,
        >;

    /// Exact placement used by an already realized local backend.
    fn execution_device(&self) -> Result<crate::DevicePlan, BackendFailure>;

    /// Samples memory facts for the exact verification queue.
    fn verification_memory(
        context: &<Self::Values as crate::backend::NativeTextValues>::Context,
    ) -> Result<crate::capability::ParameterMemoryFacts, BackendFailure>;

    /// Native wire copy used when an independent draft has a separate placement.
    type InputTransfer: crate::input::InputTensorTransfer<
        Tensor = <Self::Values as crate::backend::NativeTextValues>::InputTensor,
        Context = <Self::Values as crate::backend::NativeTextValues>::Context,
    >;

    /// Opens tensor inspection and movement for prepared draft inputs.
    fn verification_input_transfer(
        source: &<Self::Values as crate::backend::NativeTextValues>::Context,
    ) -> Self::InputTransfer;

    /// Materializes a nonempty span of committed or proposed token IDs.
    fn verification_tokens(
        tokens: &[u32],
        context: &<Self::Values as crate::backend::NativeTextValues>::Context,
    ) -> Result<<Self::Values as crate::backend::NativeTextValues>::InvocationTensor, BackendFailure>;

    /// Reads one vocabulary row selected by its zero-based decoder position.
    fn verification_score(
        scores: &<Self::Values as crate::backend::NativeTextValues>::Scores,
        position: usize,
        context: &<Self::Values as crate::backend::NativeTextValues>::Context,
    ) -> Result<Vec<f32>, BackendFailure>;

    /// Conservative bytes retained by the full native verification output.
    fn verification_storage(
        scores: &<Self::Values as crate::backend::NativeTextValues>::Scores,
    ) -> Option<u64>;
}

/// Reads one settled native predictor vocabulary row for portable host sampling.
pub trait HostSpeculativeScoreRead: crate::capture::SpeculativeCaptureTypes {
    /// Backend-owned score row.
    type Logits: Clone;
    /// Reads exactly the selected vocabulary row, with no device sampling.
    fn read_predictor_scores(
        logits: &Self::Logits,
        queue: &<Self::Queues as crate::speculative::SpeculativeQueueMechanisms>::Queue,
    ) -> Result<Vec<f32>, BackendFailure>;
}

impl crate::capture::PartitionCaptureTransport for NoTextTransport {
    fn capture_rank(&self) -> usize {
        match *self {}
    }

    fn capture_wait(&self) -> Option<BoundedCompletionWait> {
        match *self {}
    }

    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        match *self {}
    }

    fn estimate_capture_gather(
        &self,
        _: usize,
    ) -> Result<crate::capture::CaptureUsage, crate::capture::CaptureError> {
        match *self {}
    }

    fn fence_capture_exchange(&self) {
        match *self {}
    }
}

impl crate::capture::PartitionCaptureHookTransport for NoTextTransport {
    type HookOutput = Self;

    fn estimate_capture_hook(
        &self,
        _: &[usize],
    ) -> Result<crate::capture::CaptureUsage, crate::capture::CaptureError> {
        match *self {}
    }

    fn submit_capture_hook(
        &self,
        _: &[usize],
        _: bool,
    ) -> Result<Submission<Self, Self::Completion>, BackendFailure> {
        match *self {}
    }

    fn resolve_capture_hook(&self, _: Self) -> Result<bool, BackendFailure> {
        match *self {}
    }
}

impl crate::communication::CommunicationSetupTransport for NoTextTransport {
    fn setup_nonce(&self) -> Option<[u8; 32]> {
        match *self {}
    }

    fn capabilities(&self) -> crate::communication::CommunicationCapabilities {
        match *self {}
    }

    fn topology_capabilities(&self) -> crate::communication::CommunicationTopologyCapabilities {
        match *self {}
    }
}
