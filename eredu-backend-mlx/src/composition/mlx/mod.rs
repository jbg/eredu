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
pub mod realtime;
mod session;
pub mod speculative;
pub mod structural;

pub use execution::{submit_decode, submit_prefill};
pub use inspection::{inspect_model, MlxInspectionOptions};
pub use loading::validate_gguf_quantization_source;
pub use model::Executable;
#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) use processor::ModelProcessor;
pub use session::{MlxModelInput, MlxModelOutput, MlxModelSession, MlxSessionCompletion};

pub use crate::backend::{
    error::Error, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel, ModelLoadOptions,
};
