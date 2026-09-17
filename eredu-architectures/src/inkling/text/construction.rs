//! Immutable declarations for the released dense prediction-layer builder.
use super::*;
use crate::decoder::{
    ModuleMetadata,
    construction_specs::{copy_convolution, copy_linear, copy_normalization, copy_parameter},
};
fn parameter(name: String) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
fn linear(
    args: &TextArgs,
    name: String,
    input: i32,
    output: i32,
    bias: Option<String>,
) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        format: crate::linear_format::standard_linear_format(&name, args.linear_format_for(&name))?,
        weight: parameter(name)?,
        bias: bias.map(parameter).transpose()?,
    })
}
fn norm(
    args: &TextArgs,
    name: String,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec::learned(
        dimensions,
        args.rms_norm_eps,
        parameter(name)?,
    ))
}
fn convolution(
    args: &TextArgs,
    name: String,
    channels: i32,
) -> Result<CausalDepthwiseConvolutionSpec, Error> {
    Ok(CausalDepthwiseConvolutionSpec {
        channels,
        kernel_size: args.sconv_kernel_size,
        weight: parameter(name)?,
        bias: None,
        activation: ConvolutionActivation::Identity,
    })
}
#[derive(Debug)]
pub(super) struct AttentionSpec {
    query_heads: i32,
    key_value_heads: i32,
    head_dimensions: i32,
    relative_dimensions: i32,
    relative_extent: i32,
    policy: AttentionPolicy,
    log_scaling_floor: Option<i32>,
    log_scaling_alpha: f32,
    query: LinearSpec,
    key: LinearSpec,
    value: LinearSpec,
    relative: LinearSpec,
    output: LinearSpec,
    query_norm: NormalizationConstructionSpec,
    key_norm: NormalizationConstructionSpec,
    relative_projection: ParameterSpec,
    key_convolution: CausalDepthwiseConvolutionSpec,
    value_convolution: CausalDepthwiseConvolutionSpec,
}
impl AttentionSpec {
    pub(super) fn new(
        args: &TextArgs,
        policy: AttentionPolicy,
        block_root: &str,
    ) -> Result<Self, Error> {
        let local = policy.window().is_some();
        let query_heads = args.query_heads(local);
        let key_value_heads = args.key_value_heads(local);
        let head_dimensions = args.attention_head_dim(local);
        let relative_extent = policy
            .window()
            .map(|window| window.get() as i32)
            .unwrap_or(args.rel_extent);
        let prefix = format!("{block_root}.self_attn");
        let projection = |field: &str, input, output, bias: bool| {
            linear(
                args,
                format!("{prefix}.{field}.weight"),
                input,
                output,
                bias.then(|| format!("{prefix}.{field}.bias")),
            )
        };
        Ok(Self {
            query_heads,
            key_value_heads,
            head_dimensions,
            relative_dimensions: args.d_rel,
            relative_extent,
            policy,
            log_scaling_floor: args.log_scaling_n_floor,
            log_scaling_alpha: args.log_scaling_alpha,
            query: projection(
                "q_proj",
                args.hidden_size,
                query_heads * head_dimensions,
                args.q_bias,
            )?,
            key: projection(
                "k_proj",
                args.hidden_size,
                key_value_heads * head_dimensions,
                false,
            )?,
            value: projection(
                "v_proj",
                args.hidden_size,
                key_value_heads * head_dimensions,
                false,
            )?,
            relative: projection("r_proj", args.hidden_size, query_heads * args.d_rel, false)?,
            output: projection(
                "o_proj",
                query_heads * head_dimensions,
                args.hidden_size,
                args.o_bias,
            )?,
            query_norm: norm(args, format!("{prefix}.q_norm.weight"), head_dimensions)?,
            key_norm: norm(args, format!("{prefix}.k_norm.weight"), head_dimensions)?,
            relative_projection: parameter(format!("{prefix}.rel_proj"))?,
            key_convolution: convolution(
                args,
                format!("{prefix}.k_sconv.weight"),
                key_value_heads * head_dimensions,
            )?,
            value_convolution: convolution(
                args,
                format!("{prefix}.v_sconv.weight"),
                key_value_heads * head_dimensions,
            )?,
        })
    }
    pub(super) fn instantiate<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(
            Self,
            Attention<B>,
            &Self,
            [i32; 2],
            [i32; 3],
            CausalDepthwiseConvolution<B>,
        )>()?;
        Ok(Attention {
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dimensions: self.head_dimensions,
            relative_dimensions: self.relative_dimensions,
            relative_extent: self.relative_extent,
            policy: self.policy,
            log_scaling_floor: self.log_scaling_floor,
            log_scaling_alpha: self.log_scaling_alpha,
            query: B::linear(copy_linear::<B>(&self.query, context)?, context)?,
            key: B::linear(copy_linear::<B>(&self.key, context)?, context)?,
            value: B::linear(copy_linear::<B>(&self.value, context)?, context)?,
            relative: B::linear(copy_linear::<B>(&self.relative, context)?, context)?,
            output: B::linear(copy_linear::<B>(&self.output, context)?, context)?,
            query_norm: B::normalization(
                copy_normalization::<B>(&self.query_norm, context)?,
                context,
            )?,
            key_norm: B::normalization(copy_normalization::<B>(&self.key_norm, context)?, context)?,
            relative_projection: Parameter::unloaded(
                copy_parameter::<B>(&self.relative_projection, context)?,
                &[self.relative_dimensions, self.relative_extent],
                context,
            )?,
            key_convolution: CausalDepthwiseConvolution::new(
                copy_convolution::<B>(&self.key_convolution, context)?,
                context,
            )?,
            value_convolution: CausalDepthwiseConvolution::new(
                copy_convolution::<B>(&self.value_convolution, context)?,
                context,
            )?,
        })
    }
}
#[derive(Debug)]
pub(super) struct DenseSpec {
    gate: LinearSpec,
    up: LinearSpec,
    down: LinearSpec,
    global_scale: ParameterSpec,
}
impl DenseSpec {
    pub(super) fn new(args: &TextArgs, root: &str) -> Result<Self, Error> {
        let prefix = format!("{root}.dense");
        let width = args.dense_intermediate_size();
        Ok(Self {
            gate: linear(
                args,
                format!("{prefix}.gate_proj.weight"),
                args.hidden_size,
                width,
                None,
            )?,
            up: linear(
                args,
                format!("{prefix}.up_proj.weight"),
                args.hidden_size,
                width,
                None,
            )?,
            down: linear(
                args,
                format!("{prefix}.down_proj.weight"),
                width,
                args.hidden_size,
                None,
            )?,
            global_scale: parameter(format!("{root}.dense_global_scale"))?,
        })
    }
    pub(super) fn instantiate<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DenseMlp<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, &Self, DenseMlp<B>, [i32; 1])>()?;
        Ok(DenseMlp {
            gate: B::linear(copy_linear::<B>(&self.gate, context)?, context)?,
            up: B::linear(copy_linear::<B>(&self.up, context)?, context)?,
            down: B::linear(copy_linear::<B>(&self.down, context)?, context)?,
            global_scale: Parameter::unloaded(
                copy_parameter::<B>(&self.global_scale, context)?,
                &[1],
                context,
            )?,
        })
    }
}

#[derive(Debug)]
enum FeedForwardSpec {
    Dense(DenseSpec),
    Sparse(SparseSpec),
}
#[derive(Debug)]
pub(super) struct SparseSpec {
    routed_count: i32,
    shared_count: i32,
    top_k: i32,
    coefficient_scale: f32,
    hidden_size: i32,
    router_weight: ParameterSpec,
    router_bias: ParameterSpec,
    global_scale: ParameterSpec,
    routed: GroupedGatedProductSpec,
    shared: GroupedGatedProductSpec,
}
impl SparseSpec {
    pub(super) fn new(
        args: &TextArgs,
        root: &str,
        realization: Option<crate::inkling::ExpertBankRealization>,
    ) -> Result<Self, Error> {
        let prefix = format!("{root}.moe");
        let (routed, shared) = match realization {
            Some(realization) => (realization.routed, realization.shared),
            None => (
                expert_bank_spec_at(args, &prefix, "experts", args.n_routed_experts)?,
                expert_bank_spec_at(args, &prefix, "shared_experts", args.n_shared_experts)?,
            ),
        };
        Ok(Self {
            routed_count: args.n_routed_experts,
            shared_count: args.n_shared_experts,
            top_k: args.num_experts_per_tok,
            coefficient_scale: args.route_scale,
            hidden_size: args.hidden_size,
            router_weight: parameter(format!("{prefix}.router.weight"))?,
            router_bias: parameter(format!("{prefix}.router.bias"))?,
            global_scale: parameter(format!("{prefix}.router.global_scale"))?,
            routed,
            shared,
        })
    }
    pub(super) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SparseMlp<B>, Error> {
        ModuleMetadata::new::<B>(context)
            .controls::<(&Self, SparseMlp<B>, [i32; 2], [i32; 1])>()?;
        Ok(SparseMlp {
            routed_count: self.routed_count,
            shared_count: self.shared_count,
            top_k: self.top_k,
            coefficient_scale: self.coefficient_scale,
            router_weight: Parameter::unloaded(
                copy_parameter::<B>(&self.router_weight, context)?,
                &[self.routed_count + self.shared_count, self.hidden_size],
                context,
            )?,
            router_bias: Parameter::unloaded(
                copy_parameter::<B>(&self.router_bias, context)?,
                &[self.routed_count],
                context,
            )?,
            global_scale: Parameter::unloaded(
                copy_parameter::<B>(&self.global_scale, context)?,
                &[1],
                context,
            )?,
            routed_experts: B::grouped_gated_product(
                crate::decoder::construction_specs::copy_grouped::<B>(&self.routed, context)?,
                context,
            )?,
            shared_experts: B::grouped_gated_product(
                crate::decoder::construction_specs::copy_grouped::<B>(&self.shared, context)?,
                context,
            )?,
        })
    }
}

/// The original dense prediction API uses the same common layer construction.
#[derive(Debug)]
pub(crate) struct DenseLayerSpec(DecoderLayerSpec);
impl DenseLayerSpec {
    pub(crate) fn new(
        args: &TextArgs,
        attention: AttentionPolicy,
        root: &str,
        layer: usize,
        shared_expert_layer: usize,
    ) -> Result<Self, Error> {
        Ok(Self(DecoderLayerSpec::new(
            args,
            LayerPolicy {
                attention,
                feed_forward: FeedForwardPolicy::Dense,
            },
            root,
            layer,
            shared_expert_layer,
            None,
        )?))
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DecoderLayer<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, DecoderLayer<B>)>()?;
        self.0.instantiate::<B>(context)
    }
}
#[derive(Debug)]
pub(crate) struct DecoderLayerSpec {
    layer: usize,
    shared_expert_layer: usize,
    input_norm: NormalizationConstructionSpec,
    attention: AttentionSpec,
    attention_convolution: CausalDepthwiseConvolutionSpec,
    post_attention_norm: NormalizationConstructionSpec,
    feed_forward: FeedForwardSpec,
    feed_forward_convolution: CausalDepthwiseConvolutionSpec,
}
impl DecoderLayerSpec {
    pub(crate) fn matches_realization(
        &self,
        source: Option<&crate::inkling::ExpertBankRealization>,
    ) -> bool {
        match (&self.feed_forward, source) {
            (FeedForwardSpec::Dense(_), None) => true,
            (FeedForwardSpec::Sparse(spec), Some(source)) => {
                spec.routed == source.routed && spec.shared == source.shared
            }
            _ => false,
        }
    }
    pub(crate) fn new(
        args: &TextArgs,
        policy: LayerPolicy,
        root: &str,
        layer: usize,
        shared_expert_layer: usize,
        realization: Option<crate::inkling::ExpertBankRealization>,
    ) -> Result<Self, Error> {
        Ok(Self {
            layer,
            shared_expert_layer,
            input_norm: norm(
                args,
                format!("{root}.input_layernorm.weight"),
                args.hidden_size,
            )?,
            attention: AttentionSpec::new(args, policy.attention, root)?,
            attention_convolution: convolution(
                args,
                format!("{root}.attn_sconv.weight"),
                args.hidden_size,
            )?,
            post_attention_norm: norm(
                args,
                format!("{root}.post_attention_layernorm.weight"),
                args.hidden_size,
            )?,
            feed_forward: match policy.feed_forward {
                FeedForwardPolicy::Dense => FeedForwardSpec::Dense(DenseSpec::new(args, root)?),
                FeedForwardPolicy::SparseMoe => {
                    FeedForwardSpec::Sparse(SparseSpec::new(args, root, realization)?)
                }
            },
            feed_forward_convolution: convolution(
                args,
                format!("{root}.mlp_sconv.weight"),
                args.hidden_size,
            )?,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DecoderLayer<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(
            Self,
            &Self,
            DecoderLayer<B>,
            [i32; 3],
            CausalDepthwiseConvolution<B>,
        )>()?;
        Ok(DecoderLayer {
            layer: self.layer,
            shared_expert_layer: self.shared_expert_layer,
            input_norm: B::normalization(
                copy_normalization::<B>(&self.input_norm, context)?,
                context,
            )?,
            attention: self.attention.instantiate::<B>(context)?,
            attention_convolution: CausalDepthwiseConvolution::new(
                copy_convolution::<B>(&self.attention_convolution, context)?,
                context,
            )?,
            post_attention_norm: B::normalization(
                copy_normalization::<B>(&self.post_attention_norm, context)?,
                context,
            )?,
            feed_forward: match &self.feed_forward {
                FeedForwardSpec::Dense(spec) => FeedForward::Dense(spec.instantiate::<B>(context)?),
                FeedForwardSpec::Sparse(spec) => {
                    FeedForward::Sparse(spec.instantiate::<B>(context)?)
                }
            },
            feed_forward_convolution: CausalDepthwiseConvolution::new(
                copy_convolution::<B>(&self.feed_forward_convolution, context)?,
                context,
            )?,
        })
    }
}
