//! Architecture-erased MLX model-session execution.

use eredu_core::{
    BackendSession, Completion, InputModality, InspectableBackendSession, InspectedOutput,
    ModelRuntime, ObservationRequest, ObservationSet, ObservationValue, SessionAdmission,
    SessionAuthority, Submission, SubmissionLease, TensorObservation, TensorObservationData,
    TextGenerationBackend, TextGenerationConfig, TokenFilter, TokenOutput,
};
use eredu_nn::Tensor as _;
use eredu_runtime::{ActivationObserver as RuntimeActivationObserver, Sampler, SamplingBackend};
#[cfg(test)]
use eredu_runtime::{GenerationSampler, MirostatV2Sampler};
use ref_cast::RefCast;
use safemlx::{
    Array, Dtype, Stream,
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
};
use std::path::Path;

use crate::{
    MlxTensor,
    backend::error::Error,
    backend::nn::tensor::{TokenValidationBatch, TokenValidationScope},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    backend::runtime::media::input,
};
#[cfg(any(feature = "image", feature = "audio"))]
use crate::{backend::runtime::media::PreparedModelInput, composition::mlx::ModelProcessor};
use eredu_core::SpeculativeCapability;
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheOptions};
use eredu_runtime::CacheResidencyPolicy;

use super::{Executable, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel};

pub(crate) mod bounded_capture;
pub(in crate::composition::mlx) mod capture_workspace;
mod generation;
pub(crate) mod intervention;
mod model_session;
pub(in crate::composition::mlx) use model_session::control_slot::{
    PreparedControlSlotError, error as prepared_control_slot_error,
};
pub(crate) use model_session::text_quote::{prediction_scope_facts, preparation_scope_facts};
mod observation;
mod output_completion;
mod text_snapshot;
pub(super) use crate::backend::submission_recovery as recovery;

pub use generation::{MlxTextGenerationState, MlxTextSamplingState};
pub(crate) use model_session::{CompletedOriginalModelInput, OriginalInterventionDeclaration};
pub use model_session::{
    MlxHostInputUploadError, MlxModelInput, MlxModelSession, MlxNativeTextState,
    MlxOriginalPreparedModelInput, MlxOriginalPreparedNativeInput, MlxPreparedInputMaterializer,
    MlxPreparedModelInputBindError, MlxPreparedModelInputError, MlxPreparedModelInputPlan,
    MlxPreparedNativeInputError, MlxPreparedNativeInputPlan,
};
pub(in crate::composition::mlx) use model_session::{
    OriginalModelPartitionPreparation, OriginalModelPartitionSource, SpeculativePartitionBinding,
};
pub use output_completion::{
    MlxModelOutput, MlxSessionCompletion, MlxTextCompletion, MlxTextToken,
};

pub(crate) use generation::MlxTextSampler;
use generation::sample_text_submission;
#[cfg(test)]
use model_session::model_submission;
use observation::{ArrayObserverAdapter, InspectionCollector, observe_tensor};
use output_completion::MlxSessionCompletionKind;

#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(in crate::composition::mlx) use model_session::saved_array_copy::decoder::tests as paired_copy_fixture;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(in crate::composition::mlx) use model_session::saved_array_copy::decoder::resume_driver::failure_tests as resume_failure_fixture;
