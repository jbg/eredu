//! Immutable pooling, index and rotary declarations for the existing worker.
use super::*;
use crate::decoder::{ModuleMetadata, construction_specs::*};
use eredu_nn::LowRankProjectionSpec;

fn linear_spec(args: &V4Args, name: String, input: i32, output: i32) -> Result<LinearSpec, Error> {
    Ok(LinearSpec {
        input,
        output,
        weight: parameter(&name)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(&name, args.linear_format_for(&name))?,
    })
}
#[derive(Debug, Clone, Copy)]
struct RotarySpec {
    values: V4FrequencyValues,
}
impl RotarySpec {
    fn new(args: &V4Args, base: f32, yarn: bool, scale: i32) -> Self {
        Self {
            values: V4FrequencyValues::new(
                args.qk_rope_head_dim,
                base,
                if yarn {
                    args.rope_scaling.as_ref()
                } else {
                    None
                },
                scale,
            ),
        }
    }
    fn instantiate<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<V4Rotary<B::Tensor>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, V4Rotary<B::Tensor>, &Self)>()?;
        Ok(V4Rotary {
            rotary_dimensions: self.values.dimensions,
            frequency_scale: self.values.frequency_scale,
            frequencies: self.values.initialize(context)?,
        })
    }
}
#[derive(Debug)]
struct CompressorSpec {
    ratio: i32,
    head_dimensions: i32,
    overlapping: bool,
    output: i32,
    wkv: LinearSpec,
    wgate: LinearSpec,
    ape: ParameterSpec,
    norm: NormalizationConstructionSpec,
    rope: RotarySpec,
}
impl CompressorSpec {
    fn new(args: &V4Args, ratio: i32, head_dimensions: i32, root: &str) -> Result<Self, Error> {
        let overlapping = ratio == 4;
        let output = head_dimensions * if overlapping { 2 } else { 1 };
        Ok(Self {
            ratio,
            head_dimensions,
            overlapping,
            output,
            wkv: linear_spec(args, format!("{root}.wkv.weight"), args.hidden_size, output)?,
            wgate: linear_spec(
                args,
                format!("{root}.wgate.weight"),
                args.hidden_size,
                output,
            )?,
            ape: parameter(format!("{root}.ape"))?,
            norm: NormalizationConstructionSpec::learned(
                head_dimensions,
                args.rms_norm_eps,
                parameter(format!("{root}.norm.weight"))?,
            ),
            rope: RotarySpec::new(args, args.compress_rope_theta, true, ratio),
        })
    }
    fn instantiate<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Compressor<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, Compressor<B>, &Self)>()?;
        Ok(Compressor {
            ratio: self.ratio,
            head_dimensions: self.head_dimensions,
            overlapping: self.overlapping,
            wkv: B::linear(copy_linear::<B>(&self.wkv, context)?, context)?,
            wgate: B::linear(copy_linear::<B>(&self.wgate, context)?, context)?,
            ape: Parameter::unloaded(
                copy_parameter::<B>(&self.ape, context)?,
                &[self.ratio, self.output],
                context,
            )?,
            norm: B::normalization(copy_normalization::<B>(&self.norm, context)?, context)?,
            rope: self.rope.instantiate::<B>(context)?,
        })
    }
}
#[derive(Debug)]
struct IndexerSpec {
    heads: i32,
    head_dimensions: i32,
    top_k: i32,
    wq_b: LinearSpec,
    weights_projection: LinearSpec,
    compressor: CompressorSpec,
}
impl IndexerSpec {
    fn new(args: &V4Args, ratio: i32, root: &str) -> Result<Self, Error> {
        Ok(Self {
            heads: args.index_n_heads,
            head_dimensions: args.index_head_dim,
            top_k: args.index_topk,
            wq_b: linear_spec(
                args,
                format!("{root}.wq_b.weight"),
                args.q_lora_rank,
                args.index_n_heads * args.index_head_dim,
            )?,
            weights_projection: linear_spec(
                args,
                format!("{root}.weights_proj.weight"),
                args.hidden_size,
                args.index_n_heads,
            )?,
            compressor: CompressorSpec::new(
                args,
                ratio,
                args.index_head_dim,
                &format!("{root}.compressor"),
            )?,
        })
    }
    fn instantiate<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Indexer<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, Indexer<B>, &Self)>()?;
        Ok(Indexer {
            heads: self.heads,
            head_dimensions: self.head_dimensions,
            top_k: self.top_k,
            wq_b: B::linear(copy_linear::<B>(&self.wq_b, context)?, context)?,
            weights_projection: B::linear(
                copy_linear::<B>(&self.weights_projection, context)?,
                context,
            )?,
            compressor: self.compressor.instantiate::<B>(context)?,
        })
    }
}
#[derive(Debug)]
pub(crate) struct V4AttentionSpec {
    heads: i32,
    head_dimensions: i32,
    groups: i32,
    output_rank: i32,
    scale: f32,
    normalization_epsilon: f32,
    policy: V4AttentionPolicy,
    query: LowRankProjectionSpec,
    wkv: LinearSpec,
    kv_norm: NormalizationConstructionSpec,
    wo_a: LinearSpec,
    wo_b: LinearSpec,
    sinks: ParameterSpec,
    compressor: Option<CompressorSpec>,
    indexer: Option<IndexerSpec>,
    rope: RotarySpec,
}
impl V4AttentionSpec {
    pub(crate) fn new(args: &V4Args, layer: usize, root: &str) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let policy = args
            .attention_policy(layer)
            .ok_or_else(|| Error::backend(format!("missing V4 attention policy {layer}")))?;
        let ratio = match policy {
            V4AttentionPolicy::Local => 0,
            V4AttentionPolicy::Compressed { ratio } => ratio,
        };
        let query = ProjectionPolicy {
            first_weight: Some(format!("{root}.wq_a.weight")),
            normalization_weight: format!("{root}.q_norm.weight"),
            second_weight: format!("{root}.wq_b.weight"),
            input_dimensions: args.hidden_size,
            rank: args.q_lora_rank,
            output_dimensions: args.num_attention_heads * args.head_dim,
            epsilon: args.rms_norm_eps,
            first_format: args.linear_format_for(&format!("{root}.wq_a.weight")),
            second_format: args.linear_format_for(&format!("{root}.wq_b.weight")),
        }
        .specification()?;
        Ok(Self {
            heads: args.num_attention_heads,
            head_dimensions: args.head_dim,
            groups: args.o_groups,
            output_rank: args.o_lora_rank,
            scale: (args.head_dim as f32).sqrt().recip(),
            normalization_epsilon: args.rms_norm_eps,
            policy,
            query,
            wkv: linear_spec(
                args,
                format!("{root}.wkv.weight"),
                args.hidden_size,
                args.head_dim,
            )?,
            kv_norm: NormalizationConstructionSpec::learned(
                args.head_dim,
                args.rms_norm_eps,
                parameter(format!("{root}.kv_norm.weight"))?,
            ),
            wo_a: linear_spec(
                args,
                format!("{root}.wo_a.weight"),
                args.num_attention_heads * args.head_dim / args.o_groups,
                args.o_groups * args.o_lora_rank,
            )?,
            wo_b: linear_spec(
                args,
                format!("{root}.wo_b.weight"),
                args.o_groups * args.o_lora_rank,
                args.hidden_size,
            )?,
            sinks: parameter(format!("{root}.attn_sink"))?,
            compressor: (ratio != 0)
                .then(|| {
                    CompressorSpec::new(args, ratio, args.head_dim, &format!("{root}.compressor"))
                })
                .transpose()?,
            indexer: (ratio == 4)
                .then(|| IndexerSpec::new(args, ratio, &format!("{root}.indexer")))
                .transpose()?,
            rope: RotarySpec::new(
                args,
                if ratio == 0 {
                    args.rope_theta
                } else {
                    args.compress_rope_theta
                },
                ratio != 0,
                1,
            ),
        })
    }
    pub(crate) fn instantiate<
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    >(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, Attention<B>, &Self)>()?;
        Ok(Attention {
            heads: self.heads,
            head_dimensions: self.head_dimensions,
            groups: self.groups,
            output_rank: self.output_rank,
            scale: self.scale,
            normalization_epsilon: self.normalization_epsilon,
            policy: self.policy,
            query: LowRankProjection::new(copy_low_rank::<B>(&self.query, context)?, context)?,
            wkv: B::linear(copy_linear::<B>(&self.wkv, context)?, context)?,
            kv_norm: B::normalization(copy_normalization::<B>(&self.kv_norm, context)?, context)?,
            wo_a: B::linear(copy_linear::<B>(&self.wo_a, context)?, context)?,
            wo_b: B::linear(copy_linear::<B>(&self.wo_b, context)?, context)?,
            sinks: Parameter::unloaded(
                copy_parameter::<B>(&self.sinks, context)?,
                &[self.heads],
                context,
            )?,
            compressor: self
                .compressor
                .as_ref()
                .map(|spec| spec.instantiate::<B>(context))
                .transpose()?,
            indexer: self
                .indexer
                .as_ref()
                .map(|spec| spec.instantiate::<B>(context))
                .transpose()?,
            rope: self.rope.instantiate::<B>(context)?,
        })
    }
}
