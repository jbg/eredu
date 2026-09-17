//! One immutable MLA declaration and its shared typed backend builder.
use super::*;
use crate::{
    decoder::ModuleMetadata,
    deepseek::mtp::construction::{copy_linear, copy_low_rank, copy_normalization},
};
#[derive(Debug)]
enum QuerySpec {
    Direct(LinearSpec),
    LowRank(eredu_nn::LowRankProjectionSpec),
}
#[derive(Debug)]
pub(crate) struct AttentionSpec {
    heads: i32,
    nope: i32,
    rope_dimensions: i32,
    value_dimensions: i32,
    latent_dimensions: i32,
    scale: f32,
    query: QuerySpec,
    kv_a: LinearSpec,
    kv_norm: NormalizationConstructionSpec,
    kv_b: LinearSpec,
    output: LinearSpec,
    rotary: RotarySpec,
}
impl AttentionSpec {
    pub(crate) fn new(args: &V3Args, layer: usize) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let root = format!("model.layers.{layer}.self_attn");
        let query_width = args
            .num_attention_heads
            .checked_mul(args.qk_nope_head_dim + args.qk_rope_head_dim)
            .ok_or_else(|| Error::backend("V3 query width overflowed"))?;
        let linear = |name: String, input, output| -> Result<_, Error> {
            let format = args.linear_format_for(&name);
            Ok(LinearSpec {
                input,
                output,
                weight: parameter(&name)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(&name, format)?,
            })
        };
        let query = if let Some(rank) = args.q_lora_rank {
            let spec = ProjectionPolicy {
                first_weight: Some(format!("{root}.q_a_proj.weight")),
                normalization_weight: format!("{root}.q_a_layernorm.weight"),
                second_weight: format!("{root}.q_b_proj.weight"),
                input_dimensions: args.hidden_size,
                rank,
                output_dimensions: query_width,
                epsilon: args.rms_norm_eps,
                first_format: args.linear_format_for(&format!("{root}.q_a_proj.weight")),
                second_format: args.linear_format_for(&format!("{root}.q_b_proj.weight")),
            }
            .specification()?;
            spec.validate()?;
            QuerySpec::LowRank(spec)
        } else {
            QuerySpec::Direct(linear(
                format!("{root}.q_proj.weight"),
                args.hidden_size,
                query_width,
            )?)
        };
        let rotary_algorithm = args
            .rope_scaling
            .as_ref()
            .map_or(eredu_nn::RotaryAlgorithm::Default, |yarn| {
                yarn.rotary_algorithm()
            });
        let scale = ((args.qk_nope_head_dim + args.qk_rope_head_dim) as f32)
            .sqrt()
            .recip()
            * args
                .rope_scaling
                .as_ref()
                .map_or(1.0, |yarn| yarn.attention_multiplier());
        Ok(Self {
            heads: args.num_attention_heads,
            nope: args.qk_nope_head_dim,
            rope_dimensions: args.qk_rope_head_dim,
            value_dimensions: args.v_head_dim,
            latent_dimensions: args.kv_lora_rank,
            scale,
            query,
            kv_a: linear(
                format!("{root}.kv_a_proj_with_mqa.weight"),
                args.hidden_size,
                args.kv_lora_rank + args.qk_rope_head_dim,
            )?,
            kv_norm: NormalizationConstructionSpec::learned(
                args.kv_lora_rank,
                args.rms_norm_eps,
                parameter(format!("{root}.kv_a_layernorm.weight"))?,
            ),
            kv_b: linear(
                format!("{root}.kv_b_proj.weight"),
                args.kv_lora_rank,
                args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
            )?,
            output: linear(
                format!("{root}.o_proj.weight"),
                args.num_attention_heads * args.v_head_dim,
                args.hidden_size,
            )?,
            rotary: RotarySpec {
                arithmetic: eredu_nn::RotaryArithmetic::Native,
                dimensions: args.qk_rope_head_dim,
                base: args.rope_theta,
                traditional: false,
                algorithm: rotary_algorithm,
            },
        })
    }
    pub(crate) fn instantiate<B: BlockwiseAttentionBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(
            Self,
            Attention<B>,
            QueryProjection<B>,
            LowRankProjection<B>,
            RotarySpec,
            &Self,
        )>()?;
        let query = match &self.query {
            QuerySpec::Direct(spec) => {
                QueryProjection::Direct(B::linear(copy_linear::<B>(spec, context)?, context)?)
            }
            QuerySpec::LowRank(spec) => QueryProjection::LowRank(LowRankProjection::new(
                copy_low_rank::<B>(spec, context)?,
                context,
            )?),
        };
        Ok(Attention {
            heads: self.heads,
            nope: self.nope,
            rope_dimensions: self.rope_dimensions,
            value_dimensions: self.value_dimensions,
            latent_dimensions: self.latent_dimensions,
            scale: self.scale,
            query,
            kv_a: B::linear(copy_linear::<B>(&self.kv_a, context)?, context)?,
            kv_norm: B::normalization(copy_normalization::<B>(&self.kv_norm, context)?, context)?,
            kv_b: B::linear(copy_linear::<B>(&self.kv_b, context)?, context)?,
            output: B::linear(copy_linear::<B>(&self.output, context)?, context)?,
            rotary: B::rotary(self.rotary, context)?,
        })
    }
}
