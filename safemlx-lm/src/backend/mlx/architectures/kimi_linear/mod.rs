//! Moonshot Kimi Linear text architecture.

pub(crate) mod checkpoint;
pub mod layerwise;
pub mod model;

pub use layerwise::{
    load_kimi_linear_expert_cache_model, load_kimi_linear_layerwise_model,
    Generate as LayerwiseGenerate, KimiLinearLayerwiseModel,
};
#[cfg(test)]
pub use model::{load_tokenizer, Cache, Generate, Model, ModelArgs, ModelInput};
