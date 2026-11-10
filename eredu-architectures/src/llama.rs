//! Backend-neutral Llama/Mistral/Phi-4 decoder implementation.

pub(crate) mod checkpoint;
mod config;

pub(crate) use checkpoint::{gguf_parameter_name, translate_gguf_weight_name_for_model};
pub use checkpoint::{
    gguf_plan, load_time_quantization, safetensors_plan, translate_gguf_weight_name,
    with_checkpoint_formats, SafetensorsPlanError,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, ConfigError, ModelArgs,
};

pub use crate::decoder::{
    cache_layout, cache_layout_with_attention_geometry, create_caches, dense_parameter_description,
    partition_local_geometry, state_layout, validate_caches, Attention, AttentionInput,
    AttentionProjection, Config, ForwardContext, LayeredInput, Mlp, PartitionLocalGeometry,
    PartitionStaticModules, StaticModules, TransformerBlock,
};

use eredu_nn::Error;
use eredu_runtime::{ModelStateIdentity, ParallelPlanError, StateLayout};

/// Declares Llama's cache compatibility identity independently of its state storage backend.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Llama owned layer range overflowed"))?;
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Llama owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "llama",
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

/// Complete rank-local geometry for the canonical neutral Llama model.
pub type LocalGeometry = crate::decoder::LocalGeometry<ModelArgs>;

/// Derives Llama unit, vocabulary, and state geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    declaration: &crate::decoder::DecoderDeclaration,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    crate::decoder::local_geometry(args, declaration, layout, |config, layer, layout| {
        crate::decoder::PartitionedConfig::local_block_config(config, layer, layout)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))
    })
}

/// Shared dense-decoder lifecycle specialized to Llama configuration policy.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs>;

/// Genuinely pipeline-local neutral Llama/Mistral model.
pub type PartitionedLayeredModel<B> = crate::decoder::PartitionedLayeredModel<B, ModelArgs>;

impl crate::decoder::PartitionedConfig for ModelArgs {
    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), eredu_nn::Error> {
        if query_heads <= 0 || key_value_heads <= 0 || intermediate <= 0 {
            return Err(eredu_nn::Error::backend(
                "local Llama geometry must be positive",
            ));
        }
        self.num_attention_heads = query_heads;
        self.num_key_value_heads = key_value_heads;
        self.intermediate_size = intermediate;
        Ok(())
    }
}

/// Projects formats only after proving that geometry, sharing and equations are unchanged.
pub(crate) fn project_declaration_formats(
    declaration: &crate::decoder::DecoderDeclaration,
    source: &ModelArgs,
    selected: &ModelArgs,
) -> Result<crate::decoder::DecoderDeclaration, Error> {
    declaration.validate_for(source)?;
    let mut policy = selected.clone();
    policy.quantization = source.quantization;
    policy.quantized_weights = source.quantized_weights.clone();
    policy.quantized_weight_configs = source.quantized_weight_configs.clone();
    if policy.architecture_fingerprint() != source.architecture_fingerprint() {
        return Err(Error::backend(
            "Llama format projection changes architecture policy",
        ));
    }
    declaration.with_formats(selected)
}
