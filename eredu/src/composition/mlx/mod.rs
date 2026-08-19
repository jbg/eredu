//! Cold-path model-family selection and MLX session composition.

pub mod automatic;
mod capability;
mod family;
mod inspection;
pub(crate) mod loading;
mod model;
mod prepared_speculative;
pub mod realtime;
mod session;
pub mod speculative;
pub(crate) mod structural;

pub use capability::available_memory;
#[cfg(test)]
pub(crate) use family::ResolvedModelConfig;
pub(crate) use family::{resolve_model_config, ModelConfigResolutionError};
pub use inspection::{inspect_model, MlxInspectionOptions};
pub(crate) use loading::{gguf_eos_token_ids, validate_gguf_quantization_source};
pub(crate) use model::validate_gemma4_drafter;
pub use model::{Model, ModelCache};
pub(crate) use session::{submit_decode_with_cache, submit_prefill_with_cache};
pub use session::{
    MlxGeneration, MlxModelInput, MlxModelOutput, MlxModelSession, MlxSessionCompletion,
    MlxTextCompletion, MlxTextGenerationState, MlxTextToken,
};

pub(crate) use crate::backend::mlx::{
    error::Error, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel, MlxModelKind,
    ModelLoadOptions,
};
