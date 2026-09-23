//! Shared construction specifications used by module builders and cold preparation.
use super::*;
use eredu_runtime::execution_topology::*;

fn linear<C: Config>(
    config: &C,
    prefix: &str,
    field: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    let weight = format!("{prefix}.{field}.weight");
    Ok(LinearSpec {
        input,
        output,
        weight: parameter_spec(config, &weight).map_err(Error::backend)?,
        bias: bias
            .then(|| parameter_spec(config, format!("{prefix}.{field}.bias")))
            .transpose()
            .map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            &weight,
            config.linear_format(&weight),
        )?,
    })
}

/// Exact projection specifications consumed by the ordinary attention constructor.
pub(super) fn attention_projections<C: Config>(
    config: &C,
    layer: usize,
) -> Result<Vec<(String, LinearSpec)>, Error> {
    let f = config.block_parameter_fields().validate()?;
    let prefix = format!("{}.layers.{layer}.{}", config.parameter_root(), f.attention);
    let hidden = config.hidden_size();
    let q = config
        .num_attention_heads()
        .checked_mul(config.head_dim())
        .ok_or_else(|| Error::backend("query width overflow"))?;
    let kv = config
        .num_key_value_heads()
        .checked_mul(config.head_dim())
        .ok_or_else(|| Error::backend("key/value width overflow"))?;
    let mut projections = Vec::new();
    let mut add = |field: &str, input, output, bias| -> Result<(), Error> {
        projections.push((
            field.into(),
            linear(config, &prefix, field, input, output, bias)?,
        ));
        Ok(())
    };
    match config.attention_projection_layout() {
        AttentionProjectionLayout::Split => {
            add(
                f.attention_query,
                hidden,
                q,
                config.attention_bias(AttentionProjection::Query),
            )?;
            add(
                f.attention_key,
                hidden,
                kv,
                config.attention_bias(AttentionProjection::Key),
            )?;
            if !config.external_attention_value(layer) {
                add(
                    f.attention_value,
                    hidden,
                    kv,
                    config.attention_bias(AttentionProjection::Value),
                )?;
            }
        }
        AttentionProjectionLayout::Fused { field } => {
            let width = q
                .checked_add(
                    kv.checked_mul(2)
                        .ok_or_else(|| Error::backend("fused KV width overflow"))?,
                )
                .ok_or_else(|| Error::backend("fused QKV width overflow"))?;
            add(
                field,
                hidden,
                width,
                config.attention_bias(AttentionProjection::Query),
            )?;
        }
    }
    if let Some((field, _)) = config.attention_output_gate() {
        add(field, hidden, q, false)?;
    }
    add(
        f.attention_output,
        q,
        hidden,
        config.attention_bias(AttentionProjection::Output),
    )?;
    Ok(projections)
}

/// Exact projection specifications consumed by the ordinary gated MLP constructor.
pub(super) fn gated_projections<C: Config>(
    config: &C,
    layer: usize,
) -> Result<Vec<(String, LinearSpec)>, Error> {
    let f = config.block_parameter_fields().validate()?;
    let prefix = format!(
        "{}.layers.{layer}.{}",
        config.parameter_root(),
        f.feed_forward
    );
    let mut projections = Vec::new();
    let mut add = |field: &str, input, output| -> Result<(), Error> {
        projections.push((
            field.into(),
            linear(config, &prefix, field, input, output, config.mlp_bias())?,
        ));
        Ok(())
    };
    match config.gated_projection_layout() {
        GatedProjectionLayout::Split => {
            add(
                f.feed_forward_gate,
                config.hidden_size(),
                config.intermediate_size(),
            )?;
            add(
                f.feed_forward_up,
                config.hidden_size(),
                config.intermediate_size(),
            )?;
        }
        GatedProjectionLayout::Fused { field } => add(
            field,
            config.hidden_size(),
            config
                .intermediate_size()
                .checked_mul(2)
                .ok_or_else(|| Error::backend("fused gate width overflow"))?,
        )?,
    }
    add(
        f.feed_forward_output,
        config.intermediate_size(),
        config.hidden_size(),
    )?;
    Ok(projections)
}

/// Ordinary shared decoder module topology. No native allocations or calibration.
pub(crate) fn text<C: Config>(config: &C) -> Result<TextExecutionTopology, Error> {
    text_with_feed_forward(config, |layer| {
        let intermediate_size = u64::try_from(config.intermediate_size())
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("gated intermediate width must be positive"))?;
        Ok(FeedForwardTopology::Gated {
            intermediate_size,
            projections: gated_projections(config, layer)?
                .iter()
                .map(|(_, spec)| ProjectionTopology::from_spec(spec))
                .collect::<Result<_, _>>()?,
        })
    })
}

/// Compose the ordinary attention block with its actual feed-forward provider.
pub(crate) fn text_with_feed_forward<C: Config>(
    config: &C,
    mut feed_forward: impl FnMut(usize) -> Result<FeedForwardTopology, Error>,
) -> Result<TextExecutionTopology, Error> {
    config.validate_config()?;
    let positive = |n: i32| {
        u64::try_from(n)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("text topology dimensions must be positive"))
    };
    let mut missing = Vec::new();
    let layers = (0..usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?)
        .map(|layer| {
            if config.external_attention_value(layer) {
                missing.push(format!(
                    "layer {layer}: external attention-value provider invocation is not described"
                ));
            }
            if config.attention_output_gate().is_some() {
                missing.push(format!(
                    "layer {layer}: gated attention output activation is not described"
                ));
            }
            Ok(TextLayerTopology {
                mixer: TokenMixerTopology::Attention {
                    query_heads: positive(config.num_attention_heads())?,
                    kv_heads: positive(config.num_key_value_heads())?,
                    key_width: positive(config.head_dim())?,
                    value_width: positive(config.head_dim())?,
                    input_scores: config.attention_arithmetic()
                        == eredu_nn::AttentionArithmetic::InputScores,
                    softcap: config.attention_softcap().is_some(),
                    sinks: config.learned_attention_sinks(),
                    projections: attention_projections(config, layer)?
                        .iter()
                        .map(|(_, s)| ProjectionTopology::from_spec(s))
                        .collect::<Result<_, _>>()?,
                    query_key_normalization: config.query_key_norm_epsilon().is_some(),
                    rotary: config.rotary_enabled(),
                },
                feed_forward: feed_forward(layer)?,
                normalization_count: 2
                    + u64::from(config.attention_output_normalization(layer).is_some())
                    + u64::from(config.feed_forward_output_normalization(layer).is_some())
                    + u64::from(config.block_output_normalization(layer).is_some()),
            })
        })
        .collect::<Result<_, Error>>()?;
    let output = StaticModuleSpec::from_config(config).output_topology()?;
    Ok(TextExecutionTopology {
        hidden_size: positive(config.hidden_size())?,
        vocabulary_size: positive(config.vocabulary_size())?,
        layers,
        output,
        output_softcap: config.output_softcap().is_some(),
        selected_parameter_promotion_bytes: None,
        missing,
    })
}
