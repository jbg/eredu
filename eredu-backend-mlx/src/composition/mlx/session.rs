//! Architecture-erased MLX model-session execution.

use eredu_core::{
    BackendSession, Completion, InputModality, InspectableBackendSession, InspectedOutput,
    ModelRuntime, ObservationRequest, ObservationSet, ObservationValue, SessionAdmission,
    SessionAuthority, Submission, SubmissionLease, TensorObservation, TensorObservationData,
    TextGenerationBackend, TextGenerationConfig, TextSamplingStrategy, TokenFilter, TokenOutput,
};
use eredu_nn::Tensor as _;
use eredu_runtime::{
    ActivationObserver as RuntimeActivationObserver, GenerationSampler, MirostatV2Sampler, Sampler,
    SamplingBackend,
};
use ref_cast::RefCast;
use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Dtype, Stream,
};
use std::path::Path;

use crate::{
    backend::error::Error,
    backend::nn::tensor::{TokenValidationBatch, TokenValidationScope},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    backend::runtime::media::input,
    MlxTensor,
};
#[cfg(any(feature = "image", feature = "audio"))]
use crate::{backend::runtime::media::PreparedModelInput, composition::mlx::ModelProcessor};
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::SpeculativeCapability;
use eredu_runtime::CacheResidencyPolicy;

use super::{
    execution::{decode_model, prefill_model},
    Executable, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel,
};

mod generation;
pub(crate) mod bounded_capture;
mod model_session;
mod observation;
mod output_completion;
pub(super) use crate::backend::submission_recovery as recovery;

pub use generation::MlxTextGenerationState;
pub use model_session::{MlxModelInput, MlxModelSession};
pub use output_completion::{
    MlxModelOutput, MlxSessionCompletion, MlxTextCompletion, MlxTextToken,
};

use generation::{sample_text_submission, MlxTextSampler};
#[cfg(test)]
use model_session::model_submission;
use observation::{observe_tensor, ArrayObserverAdapter, InspectionCollector};
use output_completion::MlxSessionCompletionKind;

#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod tests;
