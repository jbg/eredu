//! Allocation-free projections of the same specs consumed by ordinary modules.
use super::{HybridConfig, HybridLayerPolicy};
use eredu_nn::Error;
use eredu_runtime::execution_topology::*;

fn layer(
    config: &HybridConfig,
    root: &str,
    policy: HybridLayerPolicy,
) -> Result<TextLayerTopology, Error> {
    let mixer = match policy {
        HybridLayerPolicy::SelfAttention(_) => TokenMixerTopology::Attention {
            query_heads: config.num_attention_heads as u64,
            kv_heads: config.num_key_value_heads as u64,
            key_width: config.head_dim as u64,
            value_width: config.head_dim as u64,
            input_scores: false,
            softcap: false,
            sinks: false,
            output_gate: true,
            projections: super::block::attention_projection_specs(config, root)?.iter().map(ProjectionTopology::from_spec).collect::<Result<_, _>>()?,
            query_key_normalization: true,
            rotary: true,
        },
        HybridLayerPolicy::LinearAttention => TokenMixerTopology::Unknown {
            reason: format!("{root}: gated delta recurrent attention, convolution and recurrent-update invocation topology is unavailable"),
        },
    };
    let feed_forward = if config.is_moe() {
        FeedForwardTopology::Unknown {
            reason: format!("{root}: routed experts with gated shared-expert reduction invocation topology is unavailable"),
        }
    } else {
        FeedForwardTopology::Gated {
            intermediate_size: config.intermediate_size as u64,
            projections: super::block::mlp_projection_specs(
                config,
                &format!("{root}.mlp"),
                config.intermediate_size,
            )?
            .iter()
            .map(ProjectionTopology::from_spec)
            .collect::<Result<_, _>>()?,
        }
    };
    Ok(TextLayerTopology {
        input_projections: Vec::new(),
        mixer,
        feed_forward,
        normalization_count: 2,
    })
}

fn stack(
    config: &HybridConfig,
    layers: Vec<TextLayerTopology>,
) -> Result<TextExecutionTopology, Error> {
    Ok(TextExecutionTopology {
        hidden_size: config.hidden_size as u64,
        vocabulary_size: config.vocab_size as u64,
        layers,
        output: super::model::static_module_spec(config).output_topology()?,
        output_invocations: 1,
        output_softcap: false,
        selected_parameter_promotion_bytes: None,
        selected_parameter_promotion_payloads: Default::default(),
        missing: Vec::new(),
    })
}

pub(crate) fn target(config: &HybridConfig) -> Result<TextExecutionTopology, Error> {
    config.validate().map_err(Error::backend)?;
    let layers = config
        .layer_schedule
        .iter()
        .copied()
        .enumerate()
        .map(|(index, policy)| layer(config, &format!("model.layers.{index}"), policy))
        .collect::<Result<_, _>>()?;
    stack(config, layers)
}

pub(crate) fn prediction(config: &HybridConfig) -> Result<TextExecutionTopology, Error> {
    config.validate().map_err(Error::backend)?;
    let fusion = ProjectionTopology::from_spec(&super::mtp::fusion_spec(config)?)?;
    let layers = (0..config.mtp_num_hidden_layers)
        .map(|index| {
            // Ordinary Block::new_mtp always selects full attention regardless of the
            // target schedule; every depth invokes the shared fusion and final norm.
            let mut layer = layer(
                config,
                &format!("mtp.layers.{index}"),
                HybridLayerPolicy::SelfAttention(eredu_core::AttentionPolicy::Full),
            )?;
            layer.input_projections.push(fusion.clone());
            layer.normalization_count += 3;
            Ok(layer)
        })
        .collect::<Result<_, Error>>()?;
    let mut topology = stack(config, layers)?;
    topology.output_invocations = config.mtp_num_hidden_layers as u64;
    Ok(topology)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_target_coverage_does_not_poison_full_attention_prediction() {
        let parsed = super::super::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_5_text", "vocab_size": 16, "hidden_size": 8,
            "num_hidden_layers": 2, "mtp_num_hidden_layers": 2,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
            "max_position_embeddings": 32, "intermediate_size": 16, "num_experts": 0,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 3,
            "tie_word_embeddings": true, "layer_types": ["linear_attention", "full_attention"]
        }))
        .unwrap();
        let target = target(&parsed.text).unwrap();
        assert!(
            matches!(&target.layers[0].mixer, TokenMixerTopology::Unknown { reason } if reason.contains("gated delta recurrent"))
        );
        let prediction = prediction(&parsed.text).unwrap();
        assert!(prediction
            .layers
            .iter()
            .all(|l| matches!(l.mixer, TokenMixerTopology::Attention { .. })));
        assert!(prediction.layers.iter().all(|l| l.normalization_count == 5));
    }
}
