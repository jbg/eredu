//! Cold-path neutral execution selection and MLX mechanism composition.

pub mod automatic;
mod capability;
pub mod distributed;
mod execution;
mod inspection;
mod load_request;
pub mod loading;
mod model;
mod prepared_speculative;
mod processor;
pub mod realtime;
pub(crate) mod replicated_text;
mod session;
pub mod speculative;
pub mod structural;

pub use inspection::{inspect_model, inspect_model_preparation, MlxInspectionOptions};
pub use load_request::MlxLoadRequest;
pub use loading::{MlxModelConfig, MlxSelectedPreparation};
pub(crate) use model::Executable;
#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) use processor::ModelProcessor;
pub use session::{
    MlxModelInput, MlxModelOutput, MlxModelSession, MlxNativeTextState, MlxSessionCompletion,
    MlxTextSamplingState,
};

pub(crate) use crate::backend::{
    error::Error, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel,
};
