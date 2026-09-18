//! MLX execution primitives for speculative model sessions.

/// External assistant adapters over neutral family equations and state.
pub mod external;
/// MLX semantic-generation resource adapter over the portable runtime driver.
pub mod scheduler;

pub use scheduler::SpeculativeComponentTimingGuard;

use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    CompletionCancellationMode, SamplingPlacement, SpeculativeDraftRandomPosition,
    SpeculativeExecutionTopology, SpeculativeSampling, TokenizerCompatibilityProof,
};
use eredu_runtime::{
    SelectedSpeculativeRealization, SpeculativeMechanism, SpeculativeMechanismCapabilities,
    SpeculativeSampler,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random,
    transforms::{async_eval_with_event, eval},
    Array, Event, Stream,
};

use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    MlxTensor,
};

mod assistant;
pub(crate) use assistant::{ExternalCompletedSource, retain_external_evidence,retain_external_evidence_for_roots,retain_external_evidence_for_placement};
pub(crate) mod autoregressive;
pub(in crate::composition::mlx) mod capture_error_transport;
mod completion;
pub(crate) use completion::TypedSpeculativeCompletion;
mod execution_streams;
mod numerical_sources;
mod original_domains;
pub(crate) mod embedded_native;
mod embedded_sources;
mod external_sources;
pub(crate) use external_sources::{OriginalExternalSources, ExternalInvocationSource};
pub(crate) use embedded_sources::{OriginalEmbeddedSources, EmbeddedInvocationSource, EmbeddedNumericalInvocation};
pub(crate) use numerical_sources::{OriginalSpeculativeNumericalSources, OriginalSpeculativeNumericalPreparation};
pub(in crate::composition::mlx) mod original_completion;
mod sampling;
pub(in crate::composition::mlx) use sampling::logits::{IndependentLogits, LogitsSource};
pub(in crate::composition::mlx) use sampling::numerical::{
    PendingModelLogits, completed_logits as completed_prediction_logits,
    logits_row as completed_prediction_logits_row,
    token_ids as prepared_prediction_token_ids,
    registered_copy_input, OriginalNumericalValue, repeated_token_input, concatenate_token_inputs,
    tensor_range as prepared_prediction_tensor_range, tensor_axis_range as prepared_prediction_tensor_axis_range, tensor_concatenate as prepared_prediction_tensor_concatenate,
    RegisteredTensorSource, CompletedTensorSource, registered_range as prepared_registered_token_range,
};
pub(crate) mod state_snapshot;

pub use assistant::MlxDrafter;
pub(crate) use assistant::{
    speculative_mechanism_capabilities, MlxAssistantPreparationVisitor, MlxExternalAssistant,
};
pub use completion::MlxSpeculativeCompletion;
pub use execution_streams::SpeculativeExecutionStreams;
#[cfg(test)]
use sampling::array_probability_at;
pub(crate) use sampling::validate_control_capture;
pub use sampling::{MlxSpeculativeSampling, MlxSpeculativeSeed};

#[cfg(test)]
mod completion_tests;
#[cfg(test)]
mod external_materialization_tests;

mod tensor_sources;
pub(in crate::composition::mlx) use tensor_sources::{completed_tensor_source,registered_tensor_source,validate_registered_tensor_inputs,validate_registered_tensor_inputs_at,validate_input_evidence};
