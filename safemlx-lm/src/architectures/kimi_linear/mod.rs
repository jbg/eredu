//! Moonshot Kimi Linear text architecture.

pub mod layerwise;
pub mod model;

pub use layerwise::{
    load_kimi_linear_expert_cache_model, load_kimi_linear_layerwise_model,
    Generate as LayerwiseGenerate, KimiLinearLayerwiseModel,
};
pub use model::{
    load_gguf, load_model, load_model_quantized, load_tokenizer, Cache, Generate, Model, ModelArgs,
    ModelInput,
};
