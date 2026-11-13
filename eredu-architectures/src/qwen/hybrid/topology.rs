//! Allocation-free projections of the same specs consumed by ordinary modules.
use super::{HybridConfig, HybridLayerPolicy};
use eredu_nn::Error;
use eredu_runtime::execution_topology::*;

fn layer(
    config: &HybridConfig,
    root: &str,
    policy: HybridLayerPolicy,
) -> Result<TextLayerTopology, Error> {
    let parameters = super::BlockParameters::at(config, root, policy)?;
    let mixer = match &parameters.mixer {
        super::specs::MixerParameters::Attention(spec) => TokenMixerTopology::Attention {
            query_heads: spec.projection.geometry.query_heads() as u64,
            kv_heads: spec.projection.geometry.key_value_heads() as u64,
            key_width: spec.projection.geometry.key_dimensions() as u64,
            value_width: spec.projection.geometry.value_dimensions() as u64,
            input_scores: false,
            float32_attention: false,
            softcap: false,
            sinks: false,
            output_gate: true,
            projections: spec
                .projection
                .projections()
                .into_iter()
                .map(ProjectionTopology::from_spec)
                .collect::<Result<_, _>>()?,
            query_key_normalization: true,
            rotary: true,
        },
        super::specs::MixerParameters::Recurrent(spec) => TokenMixerTopology::GatedDelta {
            value_heads: spec.geometry.value_heads() as u64,
            key_width: spec.geometry.key_dimensions() as u64,
            value_width: spec.geometry.value_dimensions() as u64,
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
        },
    };
    let feed_forward = if let super::specs::FeedForwardParameters::Routed(_) =
        &parameters.feed_forward
    {
        FeedForwardTopology::Unknown {
            reason: format!(
                "{root}: routed experts with gated shared-expert reduction invocation topology is unavailable"
            ),
        }
    } else {
        let super::specs::FeedForwardParameters::Dense(spec) = &parameters.feed_forward else {
            unreachable!()
        };
        FeedForwardTopology::Gated {
            intermediate_size: spec.down.input as u64,
            projections: spec
                .projections()
                .into_iter()
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
        output: ProjectionTopology::from_spec(&config.statics().output_projection_spec())?,
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
        .map(|(index, policy)| layer(config, &format!("model.layers.{index}"), policy))
        .collect::<Result<_, _>>()?;
    stack(config, layers)
}

pub(crate) fn prediction(config: &HybridConfig) -> Result<TextExecutionTopology, Error> {
    config.validate().map_err(Error::backend)?;
    let shared = config
        .prediction_shared()
        .ok_or_else(|| Error::backend("undeclared prediction shared scope"))?;
    let fusion = ProjectionTopology::from_spec(&shared.fusion)?;
    let layers = (0..config.mtp_num_hidden_layers)
        .map(|index| {
            // Prediction block declarations select full attention regardless of the
            // target schedule; every depth invokes the shared fusion and final norm.
            let mut layer = layer(
                config,
                &format!("mtp.layers.{index}"),
                HybridLayerPolicy::SelfAttention(eredu_core::AttentionPolicy::Full),
            )?;
            layer.input_projections.push(fusion.clone());
            layer.normalization_count += shared.normalizations().len() as u64;
            Ok(layer)
        })
        .collect::<Result<_, Error>>()?;
    let mut topology = stack(config, layers)?;
    topology.output_invocations = config.mtp_num_hidden_layers as u64;
    Ok(topology)
}
