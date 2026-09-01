//! Cold-path model-family selection and MLX session composition.

pub mod artifact;
pub mod automatic;
mod capability;
pub mod distributed;
mod execution;
mod inspection;
pub mod loading;
mod model;
mod prepared_speculative;
#[cfg(any(feature = "image", feature = "audio"))]
mod processor;
mod realization;
pub mod realtime;
mod replicated_text;
mod session;
pub mod speculative;
pub mod structural;

pub use inspection::{inspect_model, MlxInspectionOptions};
pub(crate) use loading::validate_gguf_quantization_source;
pub(crate) use model::Executable;
#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) use processor::ModelProcessor;
pub use session::{MlxModelInput, MlxModelOutput, MlxModelSession, MlxSessionCompletion};

pub(crate) use crate::backend::{
    error::Error, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel, ModelLoadOptions,
};
