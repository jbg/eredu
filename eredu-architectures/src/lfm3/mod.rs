//! Backend-neutral LFM3 tri-hybrid decoder architecture.

/// Speech input feature policy and encoder.
pub mod audio;
pub mod declarations;
pub mod omni;
mod projected_prompt;
pub use declarations::Declaration;

/// Exact heterogeneous block equations and residual order.
pub mod block;
/// Pure SafeTensors/GGUF plans and canonical name translation.
pub mod checkpoint;
/// Strict configuration normalization and heterogeneous state geometry.
pub mod config;
/// Dense SwiGLU feed-forward policy.
pub mod feed_forward;
/// Semantic placement and rank-local construction geometry.
pub mod parallel;
/// Shared complete and selected execution over exact TP/PP resource geometry.
pub mod partitioned;
/// Independently constructible hybrid vision encoder and spatial projector.
pub mod vision;

pub use block::{Block, BlockGeometry, TokenMixer};
pub use checkpoint::{
    gguf_plan, load_time_quantization, safetensors_plan, translate_gguf_weight_name, unit_recipes,
    with_checkpoint_formats,
};

pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs,
    OperatorPolicy, RopeConfig,
};
pub use feed_forward::{DenseSwiGlu, FeedForward};
pub use parallel::{
    local_block_geometry, local_geometry, partition_local_geometry, partitioned_state_layout,
    LocalGeometry, PartitionLocalGeometry,
};
pub use partitioned::{
    CompleteLfm3Geometry, CompleteLfm3Kernel, Lfm3Geometry, Lfm3Kernel, PartitionedLayeredModel,
};

use crate::decoder::StaticModuleSpec;
use eredu_core::cache::PromptCacheTopology;
use eredu_nn::Error;
use eredu_runtime::{ModelStateIdentity, StateLayout};

mod language_communication;
mod language_partition;
pub use language_communication::LanguagePipelineCommunication;
pub use language_partition::{PreparedLanguagePartition, SelectedLanguagePartition};

/// Shared cold and executable declaration for validated incremental text passes.
pub(crate) const CHUNKED_TEXT_PREFILL: bool = true;

/// Input to the LFM3 language traversal, including the requested readout.
pub enum Lfm3Input<'a, T: eredu_nn::Tensor> {
    /// Ordinary vocabulary output from token IDs and an optional mask.
    Text(crate::decoder::LayeredInput<'a, T>),
    /// Speech-conditioned input and normalized hidden output for Audio.
    Audio(&'a audio::AudioPrompt<'a, T>),
    /// Tokens conditioned on projected image and video rows.
    Vision(&'a vision::VisionPrompt<'a, T>),
    /// Jointly conditioned image, video, and speech positions.
    Omni(&'a omni::OmniPrompt<'a, T>),
}

impl<'a, T: eredu_nn::Tensor> Lfm3Input<'a, T> {
    fn tokens(&self) -> &'a T {
        match self {
            Self::Text(input) => input.tokens,
            Self::Audio(prompt) => prompt.tokens(),
            Self::Vision(prompt) => prompt.tokens(),
            Self::Omni(prompt) => prompt.tokens(),
        }
    }

    /// Number of positions in a batch-one VL pipeline invocation.
    pub fn vl_pipeline_sequence_len(&self) -> Result<usize, Error> {
        match self.tokens().shape() {
            [1, positions] if *positions > 0 => Ok(*positions as usize),
            _ => Err(Error::backend(
                "VL pipeline requires nonempty batch-one tokens",
            )),
        }
    }

    /// Exact token and projected-media roots for a VL pipeline invocation.
    pub fn vl_pipeline_roots(&self) -> impl Iterator<Item = &T> {
        let vision = match self {
            Self::Vision(prompt) => Some(*prompt),
            _ => None,
        };
        std::iter::once(self.tokens()).chain(self.mask()).chain(
            vision
                .into_iter()
                .flat_map(|prompt| prompt.projected_roots()),
        )
    }

    /// Describes VL pipeline inputs for all-rank agreement before execution.
    pub fn vl_pipeline_descriptor(&self, context: &T::Context) -> Result<[u64; 8], Error> {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(match self {
            Self::Vision(_) => [1],
            Self::Text(_) => [0],
            Self::Audio(_) | Self::Omni(_) => {
                return Err(Error::backend("VL pipeline expects text or vision input"))
            }
        });
        for dimension in self.tokens().shape() {
            digest.update(dimension.to_le_bytes());
        }
        for id in self.tokens().to_i32_vec(context)? {
            digest.update(id.to_le_bytes());
        }
        match self {
            Self::Vision(prompt) => prompt.update_pipeline_digest(&mut digest, context)?,
            Self::Text(input) => {
                digest.update([u8::from(input.mask.is_some())]);
                if let Some(mask) = input.mask {
                    update_pipeline_tensor_digest(&mut digest, mask, context)?;
                }
            }
            Self::Audio(_) | Self::Omni(_) => unreachable!("checked VL input"),
        }
        let mut descriptor = [0x4c464d335649534e, 0, 0, 0, 0, 0, 0, 0];
        for (word, bytes) in descriptor[1..]
            .iter_mut()
            .zip(digest.finalize().chunks_exact(8))
        {
            *word = u64::from_le_bytes(bytes.try_into().expect("eight digest bytes"));
        }
        Ok(descriptor)
    }

    fn mask(&self) -> Option<&'a T> {
        match self {
            Self::Text(input) => input.mask,
            Self::Audio(_) | Self::Vision(_) | Self::Omni(_) => None,
        }
    }
}

fn update_pipeline_tensor_digest<T: eredu_nn::Tensor>(
    digest: &mut sha2::Sha256,
    tensor: &T,
    context: &T::Context,
) -> Result<(), Error> {
    use sha2::Digest;
    digest.update((tensor.shape().len() as u64).to_le_bytes());
    for dimension in tensor.shape() {
        digest.update(dimension.to_le_bytes());
    }
    for value in tensor
        .cast_float(eredu_nn::TensorElementType::F32, context)?
        .to_f32_vec(context)?
    {
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(())
}

/// Architecture-owned values retained for one LFM3 forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    normalized_audio_output: bool,
}

/// Complete LFM3 execution through the shared typed component adapter.
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompleteLfm3Kernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Declares cache identity independently of its backend realization.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("LFM3 owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "LFM3 owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "lfm3",
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

pub(crate) use block::execution_topology;

pub(crate) fn static_spec(args: &ModelArgs) -> StaticModuleSpec {
    let embedding_name = "model.embed_tokens.weight";
    StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: embedding_name.into(),
        normalization_weight: "model.embedding_norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.norm_eps,
        normalization_offset: 0.0,
        embedding_format: args.weight_quantization_for(embedding_name).into(),
        head_format: args.weight_quantization_for("lm_head.weight").into(),
        tied_head: args.tie_word_embeddings,
    }
}
