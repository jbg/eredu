//! Cold-path model-family selection and MLX session composition.

pub mod artifact;
pub mod automatic;
mod capability;
pub mod distributed;
mod inspection;
pub mod loading;
mod model;
mod prepared_speculative;
#[cfg(feature = "media")]
mod processor;
pub mod realtime;
mod session;
pub mod speculative;
pub mod structural;

pub use inspection::{inspect_model, MlxInspectionOptions};
pub use loading::{gguf_eos_token_ids, validate_gguf_quantization_source};
pub use model::{Model, ModelCache};
#[cfg(feature = "media")]
pub use processor::{load_processor, ModelProcessor};
pub use session::{submit_decode_with_cache, submit_prefill_with_cache};
pub use session::{MlxModelInput, MlxModelOutput, MlxModelSession, MlxSessionCompletion};

pub use crate::backend::mlx::{
    error::Error, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel, ModelLoadOptions,
};
