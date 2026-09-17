//! Nemotron-H unpositioned grouped-query attention policy.

use eredu_nn::{Error, LinearSpec, NeuralBackend, ParameterSpec, Tensor};

use crate::decoder::Attention;

use super::{LayerPolicy, ModelArgs};
mod construction;
pub(crate) use construction::{AttentionSpec, linear_spec, linear_spec_with_metadata};

/// Builds the exact no-RoPE attention operator used by a scheduled unit.
pub fn new_attention<B: NeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    query_heads: i32,
    key_value_heads: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
    metadata.controls::<(&ModelArgs, usize, i32, i32, eredu_core::AttentionPolicy, String)>()?;
    let attention = match args.layer_schedule.get(layer) {
        Some(LayerPolicy::SelfAttention(attention)) => *attention,
        policy => {
            return Err(metadata.error(format_args!(
                "Nemotron-H layer {layer} is not attention: {policy:?}"
            )))
        }
    };
    new_attention_at(
        args,
        attention,
        &metadata.text(format_args!("model.layers.{layer}.attention"))?,
        query_heads,
        key_value_heads,
        context,
    )
}

/// Builds an attention operator at an explicit target or MTP parameter path.
pub fn new_attention_at<B: NeuralBackend>(
    args: &ModelArgs,
    attention: eredu_core::AttentionPolicy,
    prefix: &str,
    query_heads: i32,
    key_value_heads: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    construction::AttentionSpec::new_with_metadata(args, attention, prefix, query_heads, key_value_heads, crate::decoder::ModuleMetadata::new::<B>(context))?
        .instantiate::<B>(context)
    }
