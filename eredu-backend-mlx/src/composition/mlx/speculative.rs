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
pub(super) mod memory_observation;
mod parameter_policy;
pub(crate) mod autoregressive;
mod completion;
mod execution_streams;
mod sampling;
pub(crate) mod state_snapshot;

pub use assistant::MlxDrafter;
pub(crate) use assistant::{
    speculative_mechanism_capabilities, MlxAssistantPreparationVisitor, MlxExternalAssistant,
};
pub use completion::MlxSpeculativeCompletion;
pub use execution_streams::SpeculativeExecutionStreams;
pub(in crate::composition::mlx) use execution_streams::BorrowedTargetObserver;
#[cfg(test)]
use sampling::array_probability_at;
pub(crate) use sampling::validate_control_capture;
pub use sampling::MlxSpeculativeSampling;

#[cfg(test)]
mod test_fixture;
#[cfg(test)]
mod completion_tests;
