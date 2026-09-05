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
use std::cell::RefCell;

use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    MlxTensor,
};

mod assistant;
mod completion;
mod execution_streams;
mod sampling;

pub use assistant::MlxDrafter;
pub(crate) use assistant::{
    speculative_mechanism_capabilities, MlxAssistantPreparationVisitor, MlxExternalAssistant,
};
pub use completion::MlxSpeculativeCompletion;
pub use execution_streams::SpeculativeExecutionStreams;
#[cfg(test)]
use sampling::array_probability_at;
pub use sampling::MlxSpeculativeSampling;

#[cfg(test)]
mod completion_tests;
#[cfg(test)]
mod external_materialization_tests;
