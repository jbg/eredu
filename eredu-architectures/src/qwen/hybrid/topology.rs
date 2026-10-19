//! Allocation-free projections of the same specs consumed by ordinary modules.
use super::{HybridConfig, HybridLayerPolicy};
use eredu_nn::Error;
use eredu_runtime::execution_topology::*;

fn layer(
    config: &HybridConfig,
    root: &str,
    policy: HybridLayerPolicy,
    index: usize,
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
            projections: super::block::attention_projection_specs(config, root)?
                .iter()
                .map(ProjectionTopology::from_spec)
                .collect::<Result<_, _>>()?,
            query_key_normalization: true,
            rotary: true,
        },
        HybridLayerPolicy::LinearAttention => {
            let spec = super::linear_attention::recurrent_spec(config, index)?;
            TokenMixerTopology::GatedDelta {
                value_heads: spec.value_heads as u64,
                key_width: spec.key_head_dim as u64,
                value_width: spec.value_head_dim as u64,
                convolution_channels: spec.convolution.channels as u64,
                kernel: spec.convolution.kernel_size as u64,
                projections: [
                    &spec.input_qkv,
                    &spec.input_gate,
                    &spec.input_beta,
                    &spec.input_decay,
                    &spec.output,
                ]
                .into_iter()
                .map(ProjectionTopology::from_spec)
                .collect::<Result<_, _>>()?,
            }
        }
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
        iterative: None,
        hidden_size: config.hidden_size as u64,
        vocabulary_size: config.vocab_size as u64,
        layers,
        output: super::model::static_module_spec(config).output_topology()?,
        output_invocations: 1,
        output_softcap: false,
        selected_parameter_promotion_bytes: None,
        selected_parameter_promotion_payloads: Default::default(),
        projection_storage: Default::default(),
        projection_input_normalizations: Default::default(),
        f32_rms_normalization_gains: Default::default(),
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
        .map(|(index, policy)| layer(config, &format!("model.layers.{index}"), policy, index))
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
                index as usize,
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
