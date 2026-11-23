//! MLX execution primitives for speculative model sessions.

/// External assistant adapters over neutral family equations and state.
pub mod external;

use eredu_core::speculative::{SpeculativeMechanism, SpeculativeMechanismCapabilities};
use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    CompletionCancellationMode, SpeculativeDraftRandomPosition,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random,
    transforms::{async_eval_with_event, eval},
    Array, Event, Stream,
};

use crate::{
    backend::error::Error, backend::nn::shared::MlxNeuralBackend, backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend, MlxTensor,
};

mod assistant;
pub(crate) mod autoregressive;
mod completion;
mod execution_streams;
pub(super) mod memory_observation;
mod sampling;
pub(crate) mod state_snapshot;

pub(crate) use assistant::speculative_mechanism_capabilities;
pub use assistant::MlxAssistantBinder;
#[doc(hidden)]
pub use assistant::MlxDraftBinding;
pub(super) use assistant::MlxDraftStorage;
pub use completion::MlxSpeculativeCompletion;
pub(super) use sampling::MlxSpeculativeSamplingPrimitives;
