//! Nemotron-H unpositioned grouped-query attention policy.

use eredu_nn::{Error, LinearSpec, NeuralBackend, ParameterSpec, Tensor};

use crate::decoder::Attention;

use super::{LayerPolicy, ModelArgs};

/// Builds the exact no-RoPE attention operator used by a scheduled unit.
pub fn new_attention<B: NeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    query_heads: i32,
    key_value_heads: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    let attention = match args.layer_schedule.get(layer) {
        Some(LayerPolicy::SelfAttention(attention)) => *attention,
        policy => {
            return Err(Error::backend(format!(
                "Nemotron-H layer {layer} is not attention: {policy:?}"
            )))
        }
    };
    new_attention_at(
        args,
        attention,
        &format!("model.layers.{layer}.attention"),
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
    let linear = |field: &str, input, output| {
        let weight = format!("{prefix}.{field}.weight");
        B::linear(
            LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                bias: args
                    .attention_bias
                    .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                    .transpose()
                    .map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    &weight,
                    args.weight_quantization_for(&weight).into(),
                )?,
            },
            context,
        )
    };
    Attention::from_parts(
        query_heads,
        key_value_heads,
        args.head_dim,
        linear("q_proj", args.hidden_size, query_heads * args.head_dim)?,
        linear("k_proj", args.hidden_size, key_value_heads * args.head_dim)?,
        linear("v_proj", args.hidden_size, key_value_heads * args.head_dim)?,
        linear("o_proj", query_heads * args.head_dim, args.hidden_size)?,
        None,
        None,
        None,
        attention.sliding_window_i32().map_err(Error::backend)?,
    )
}
