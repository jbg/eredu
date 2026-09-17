//! Immutable full-attention prediction declarations and the ordinary typed builders.
use super::*;
use crate::decoder::{
    ModuleMetadata,
    construction_specs::{copy_grouped, copy_linear, copy_normalization, copy_selector},
};

pub(crate) fn require_source_compiler<B: NeuralBackend>(
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), Error> {
    if B::construction_metadata(context).is_some_and(|context| context.uses_checked_metadata()) {
        return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
    }
    Ok(())
}

pub(super) fn linear_spec(
    config: &HybridConfig, prefix: &str, input: i32, output: i32, bias: bool,
) -> Result<LinearSpec, Error> {
    linear_spec_with_metadata(config, prefix, input, output, bias, ModuleMetadata::ordinary())
}
fn linear_spec_with_metadata(
    config: &HybridConfig, prefix: &str, input: i32, output: i32, bias: bool,
    metadata: ModuleMetadata<'_>,
) -> Result<LinearSpec, Error> {
    metadata.controls::<(LinearSpec, String, Option<ParameterSpec>, &HybridConfig, &str)>()?;
    let weight = metadata.text(format_args!("{prefix}.weight"))?;
    Ok(LinearSpec {
        input, output,
        weight: metadata.plain_parameter(&weight)?,
        bias: bias.then(|| metadata.named_parameter(format_args!("{prefix}.bias"))).transpose()?,
        format: metadata.format(&weight, config.linear_format(&weight))?,
    })
}
fn norm(
    config: &HybridConfig, name: String, dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    norm_with_metadata(config, &name, dimensions, ModuleMetadata::ordinary())
}
fn norm_with_metadata(
    config: &HybridConfig, name: &str, dimensions: i32, metadata: ModuleMetadata<'_>,
) -> Result<NormalizationConstructionSpec, Error> {
    metadata.controls::<(NormalizationConstructionSpec, &HybridConfig, &str)>()?;
    Ok(NormalizationConstructionSpec {
        groups: None, dimensions, epsilon: config.rms_norm_eps,
        scale: NormalizationScale::LearnedOffset {
            weight: metadata.plain_parameter(name)?, offset: 1.0,
        },
    })
}

#[derive(Debug)]
pub(super) struct MlpSpec {
    gate: LinearSpec,
    up: LinearSpec,
    down: LinearSpec,
}
impl MlpSpec {
    pub(super) fn new(config: &HybridConfig, prefix: &str, intermediate: i32) -> Result<Self, Error> {
        Self::new_with_metadata(config, prefix, intermediate, ModuleMetadata::ordinary())
    }
    pub(super) fn new_with_metadata(
        config: &HybridConfig, prefix: &str, intermediate: i32, metadata: ModuleMetadata<'_>,
    ) -> Result<Self, Error> {
        metadata.controls::<(Self, &HybridConfig, &str, i32)>()?;
        Ok(Self {
            gate: linear_spec_with_metadata(config, &metadata.text(format_args!("{prefix}.gate_proj"))?,
                config.hidden_size, intermediate, false, metadata)?,
            up: linear_spec_with_metadata(config, &metadata.text(format_args!("{prefix}.up_proj"))?,
                config.hidden_size, intermediate, false, metadata)?,
            down: linear_spec_with_metadata(config, &metadata.text(format_args!("{prefix}.down_proj"))?,
                intermediate, config.hidden_size, false, metadata)?,
        })
    }
    pub(super) fn instantiate<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Mlp<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, Mlp<B>, &Self)>()?;
        Ok(Mlp::from_parts(
            B::linear(copy_linear::<B>(&self.gate, context)?, context)?,
            B::linear(copy_linear::<B>(&self.up, context)?, context)?,
            B::linear(copy_linear::<B>(&self.down, context)?, context)?,
            None,
        ))
    }
}

#[derive(Debug)]
pub(super) struct AttentionSpec {
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    query: LinearSpec,
    key: LinearSpec,
    value: LinearSpec,
    output: LinearSpec,
    query_norm: NormalizationConstructionSpec,
    key_norm: NormalizationConstructionSpec,
    rotary: RotarySpec,
}
impl AttentionSpec {
    pub(super) fn new(config: &HybridConfig, root: &str) -> Result<Self, Error> {
        Self::new_with_metadata(config, root, ModuleMetadata::ordinary())
    }
    pub(super) fn new_with_metadata(
        config: &HybridConfig, root: &str, metadata: ModuleMetadata<'_>,
    ) -> Result<Self, Error> {
        metadata.controls::<(Self, &HybridConfig, &str, String, RotarySpec)>()?;
        let prefix = metadata.text(format_args!("{root}.self_attn"))?;
        let linear = |field: &str, input, output| {
            linear_spec_with_metadata(
                config,
                &metadata.text(format_args!("{prefix}.{field}"))?,
                input,
                output,
                config.attention_bias,
                metadata,
            )
        };
        metadata.borrowed_controls(&linear)?;
        Ok(Self {
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            query: linear(
                "q_proj",
                config.hidden_size,
                2 * config.num_attention_heads * config.head_dim,
            )?,
            key: linear(
                "k_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            value: linear(
                "v_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            output: linear(
                "o_proj",
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            )?,
            query_norm: norm_with_metadata(config, &metadata.text(format_args!("{prefix}.q_norm.weight"))?, config.head_dim, metadata)?,
            key_norm: norm_with_metadata(config, &metadata.text(format_args!("{prefix}.k_norm.weight"))?, config.head_dim, metadata)?,
            rotary: RotarySpec {
                arithmetic: eredu_nn::RotaryArithmetic::Native,
                dimensions: config.rope_dimensions(),
                base: config.rope_theta(),
                traditional: false,
                algorithm: config.rotary_algorithm_with_metadata(metadata)?,
            },
        })
    }
    pub(super) fn instantiate<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, Attention<B>, &Self, RotarySpec)>()?;
        Attention::from_gated_parts(
            self.heads,
            self.kv_heads,
            self.head_dim,
            B::linear(copy_linear::<B>(&self.query, context)?, context)?,
            B::linear(copy_linear::<B>(&self.key, context)?, context)?,
            B::linear(copy_linear::<B>(&self.value, context)?, context)?,
            B::linear(copy_linear::<B>(&self.output, context)?, context)?,
            Some(B::normalization(
                copy_normalization::<B>(&self.query_norm, context)?,
                context,
            )?),
            Some(B::normalization(
                copy_normalization::<B>(&self.key_norm, context)?,
                context,
            )?),
            Some(B::rotary(self.rotary, context)?),
            None,
        )
    }
}

#[derive(Debug)]
pub(super) struct RoutedSpec {
    layer: usize,
    router: TopKGroupSelectorSpec,
    experts: GroupedGatedProductSpec,
    shared: MlpSpec,
    gate: LinearSpec,
}
impl RoutedSpec {
    pub(super) fn new(
        config: &HybridConfig,
        layer: usize,
        prefix: &str,
        routed_spec: Option<GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        let prefix = format!("{prefix}.mlp");
        let router_name = format!("{prefix}.gate.weight");
        Ok(Self {
            layer,
            router: TopKGroupSelectorSpec::new(
                config.hidden_size,
                parameter(&router_name)?,
                crate::linear_format::standard_linear_format(
                    &router_name,
                    config.quantization.into(),
                )?,
                config.routing_spec()?,
            )?,
            experts: match routed_spec {
                Some(spec) => spec,
                None => expert_bank_spec_at(config, &format!("{prefix}.experts"))?,
            },
            shared: MlpSpec::new(
                config,
                &format!("{prefix}.shared_expert"),
                config.shared_expert_intermediate_size,
            )?,
            gate: linear_spec(
                config,
                &format!("{prefix}.shared_expert_gate"),
                config.hidden_size,
                1,
                false,
            )?,
        })
    }
    pub(super) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SharedRoutedGatedProduct<B>, Error> {
        ModuleMetadata::new::<B>(context)
            .controls::<(Self, SharedRoutedGatedProduct<B>, &Self)>()?;
        Ok(SharedRoutedGatedProduct {
            layer: self.layer,
            resident_unit_coordinates: None,
            router: B::top_k_group_selector(copy_selector::<B>(&self.router, context)?, context)?,
            experts: B::grouped_gated_product(copy_grouped::<B>(&self.experts, context)?, context)?,
            shared_expert: self.shared.instantiate::<B>(context)?,
            shared_expert_gate: B::linear(copy_linear::<B>(&self.gate, context)?, context)?,
        })
    }
}
#[derive(Debug)]
enum FeedForwardSpec {
    Dense(MlpSpec),
    Routed(RoutedSpec),
}
#[derive(Debug)]
pub(crate) struct PredictionBlockSpec {
    attention: AttentionSpec,
    feed_forward: FeedForwardSpec,
    input_norm: NormalizationConstructionSpec,
    post_attention_norm: NormalizationConstructionSpec,
}
impl PredictionBlockSpec {
    pub(crate) fn new(config: &HybridConfig, depth: usize) -> Result<Self, Error> {
        if depth >= usize::try_from(config.mtp_num_hidden_layers).map_err(Error::backend)? {
            return Err(Error::backend(format!(
                "Qwen hybrid MTP depth {depth} is outside {} configured layers",
                config.mtp_num_hidden_layers
            )));
        }
        let root = format!("mtp.layers.{depth}");
        Ok(Self {
            attention: AttentionSpec::new(config, &root)?,
            feed_forward: if config.is_moe() {
                FeedForwardSpec::Routed(RoutedSpec::new(
                    config,
                    config.num_hidden_layers as usize + depth,
                    &root,
                    None,
                )?)
            } else {
                FeedForwardSpec::Dense(MlpSpec::new(
                    config,
                    &format!("{root}.mlp"),
                    config.intermediate_size,
                )?)
            },
            input_norm: norm(
                config,
                format!("{root}.input_layernorm.weight"),
                config.hidden_size,
            )?,
            post_attention_norm: norm(
                config,
                format!("{root}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Block<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, Block<B>, &Self)>()?;
        let mixer = TokenMixer::Attention(self.attention.instantiate::<B>(context)?);
        let feed_forward = match &self.feed_forward {
            FeedForwardSpec::Dense(spec) => FeedForward::Dense(spec.instantiate::<B>(context)?),
            FeedForwardSpec::Routed(spec) => FeedForward::Routed(spec.instantiate::<B>(context)?),
        };
        Ok(Block {
            mixer,
            feed_forward,
            input_norm: B::normalization(
                copy_normalization::<B>(&self.input_norm, context)?,
                context,
            )?,
            post_attention_norm: B::normalization(
                copy_normalization::<B>(&self.post_attention_norm, context)?,
                context,
            )?,
        })
    }
}

/// Initial target-unit declarations; the recurrent branch borrows the same
/// immutable model configuration when invoking its metadata-aware constructor.
#[derive(Debug)]
enum TargetMixerSpec {
    Linear,
    Attention(AttentionSpec),
}
#[derive(Debug)]
pub(crate) struct TargetBlockSpec {
    layer: usize,
    mixer: TargetMixerSpec,
    feed_forward: FeedForwardSpec,
    input_norm: NormalizationConstructionSpec,
    post_attention_norm: NormalizationConstructionSpec,
}
impl TargetBlockSpec {
    pub(crate) fn new(
        config: &HybridConfig, layer: usize, routed_spec: Option<GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        let policy = config.layer_schedule.get(layer).copied()
            .ok_or_else(|| Error::backend(format!("Qwen hybrid has no layer {layer}")))?;
        let root = format!("model.layers.{layer}");
        Ok(Self {
            layer,
            mixer: match policy {
                HybridLayerPolicy::LinearAttention => TargetMixerSpec::Linear,
                HybridLayerPolicy::SelfAttention(_) => TargetMixerSpec::Attention(AttentionSpec::new(config, &root)?),
            },
            feed_forward: if config.is_moe() {
                FeedForwardSpec::Routed(RoutedSpec::new(config, layer, &root, routed_spec)?)
            } else {
                FeedForwardSpec::Dense(MlpSpec::new(config, &format!("{root}.mlp"), config.intermediate_size)?)
            },
            input_norm: norm(config, format!("{root}.input_layernorm.weight"), config.hidden_size)?,
            post_attention_norm: norm(config, format!("{root}.post_attention_layernorm.weight"), config.hidden_size)?,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self, config: &HybridConfig, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Block<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, &HybridConfig, Block<B>)>()?;
        let mixer = match &self.mixer {
            TargetMixerSpec::Linear => TokenMixer::Linear(LinearAttention::new(config, self.layer, context)?),
            TargetMixerSpec::Attention(spec) => TokenMixer::Attention(spec.instantiate::<B>(context)?),
        };
        let feed_forward = match &self.feed_forward {
            FeedForwardSpec::Dense(spec) => FeedForward::Dense(spec.instantiate::<B>(context)?),
            FeedForwardSpec::Routed(spec) => FeedForward::Routed(spec.instantiate::<B>(context)?),
        };
        Ok(Block {
            mixer, feed_forward,
            input_norm: B::normalization(copy_normalization::<B>(&self.input_norm, context)?, context)?,
            post_attention_norm: B::normalization(copy_normalization::<B>(&self.post_attention_norm, context)?, context)?,
        })
    }
}
