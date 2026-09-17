use super::*;
use crate::decoder::{construction_specs::copy_linear, ModuleMetadata};
#[derive(Debug)]
pub(crate) struct AttentionSpec {
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    window: Option<i32>,
    query: LinearSpec,
    key: LinearSpec,
    value: LinearSpec,
    output: LinearSpec,
}
pub(crate) fn linear_spec(
    args: &ModelArgs,
    prefix: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    linear_spec_with_metadata(args, prefix, input, output, bias, ModuleMetadata::ordinary())
}
pub(crate) fn linear_spec_with_metadata(
    args: &ModelArgs, prefix: &str, input: i32, output: i32, bias: bool,
    metadata: ModuleMetadata<'_>,
) -> Result<LinearSpec, Error> {
    metadata.controls::<(LinearSpec, &ModelArgs, &str, String, Option<ParameterSpec>)>()?;
    let weight = metadata.text(format_args!("{prefix}.weight"))?;
    Ok(LinearSpec {
        input, output,
        weight: metadata.plain_parameter(&weight)?,
        bias: bias.then(|| metadata.named_parameter(format_args!("{prefix}.bias"))).transpose()?,
        format: metadata.format(&weight, args.weight_quantization_for(&weight).into())?,
    })
}

impl AttentionSpec {
    pub(crate) fn new(
        args: &ModelArgs,
        attention: eredu_core::AttentionPolicy,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
    ) -> Result<Self, Error> {
        Self::new_with_metadata(args, attention, prefix, heads, kv_heads, ModuleMetadata::ordinary())
    }
    pub(crate) fn new_with_metadata(
        args: &ModelArgs, attention: eredu_core::AttentionPolicy, prefix: &str,
        heads: i32, kv_heads: i32, metadata: ModuleMetadata<'_>,
    ) -> Result<Self, Error> {
        metadata.controls::<(Self, &ModelArgs, &str, i32, i32, eredu_core::AttentionPolicy)>()?;
        let linear = |field: &str, input, output| {
            linear_spec_with_metadata(
                args,
                &metadata.text(format_args!("{prefix}.{field}"))?,
                input,
                output,
                args.attention_bias,
                metadata,
            )
        };
        metadata.borrowed_controls(&linear)?;
        Ok(Self {
            heads,
            kv_heads,
            head_dim: args.head_dim,
            window: attention.sliding_window_i32().map_err(|cause| metadata.error(format_args!("{cause}")))?,
            query: linear("q_proj", args.hidden_size, heads * args.head_dim)?,
            key: linear("k_proj", args.hidden_size, kv_heads * args.head_dim)?,
            value: linear("v_proj", args.hidden_size, kv_heads * args.head_dim)?,
            output: linear("o_proj", heads * args.head_dim, args.hidden_size)?,
        })
    }
    pub(crate) fn instantiate<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, Attention<B>)>()?;
        Attention::from_parts(
            self.heads,
            self.kv_heads,
            self.head_dim,
            B::linear(copy_linear::<B>(&self.query, context)?, context)?,
            B::linear(copy_linear::<B>(&self.key, context)?, context)?,
            B::linear(copy_linear::<B>(&self.value, context)?, context)?,
            B::linear(copy_linear::<B>(&self.output, context)?, context)?,
            None,
            None,
            None,
            self.window,
        )
    }
}
