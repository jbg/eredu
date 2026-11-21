//! Architecture-erased MLX model-session execution.

use eredu_core::{Completion, Submission, TensorObservation, TensorObservationData, TokenOutput};
use eredu_nn::Tensor as _;
use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error,
    backend::nn::tensor::{TokenValidationBatch, TokenValidationScope},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    backend::runtime::media::input,
    MlxTensor,
};

use super::{MlxBackend, MlxCompletion, MlxDistributedSession};

pub(crate) mod bounded_capture;

mod generation;
pub(crate) mod intervention;
pub(super) mod model_session;
mod observation;
mod output_completion;
mod score_readout;
mod text_snapshot;
pub(super) use crate::backend::submission_recovery as recovery;

pub use model_session::MlxSessionMechanisms;
pub use output_completion::{
    MlxModelOutput, MlxSessionCompletion, MlxTextCompletion, MlxTextToken,
};

use observation::observe_tensor;
use output_completion::MlxSessionCompletionKind;

pub(super) fn capture_capabilities() -> eredu_core::capture::CaptureCapabilities {
    #[cfg(feature = "tensor-capture")]
    return bounded_capture::capabilities();
    #[cfg(not(feature = "tensor-capture"))]
    eredu_core::capture::CaptureCapabilities::default()
}
