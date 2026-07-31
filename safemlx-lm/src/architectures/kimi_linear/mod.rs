//! Moonshot Kimi Linear text architecture.

pub mod layerwise;
pub mod model;

pub use layerwise::{
    load_kimi_linear_layerwise_model, load_kimi_linear_sparse_expert_cache_model,
    load_kimi_linear_sparse_expert_cache_model_with_dense_layers, Generate as LayerwiseGenerate,
    KimiLinearLayerwiseModel,
};
pub use model::{
    load_gguf, load_model, load_model_quantized, load_tokenizer, Cache, Generate, Model, ModelArgs,
    ModelInput,
};
