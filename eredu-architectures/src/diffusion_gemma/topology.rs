//! Cold invocation geometry uses the same projection declarations as construction.
use super::FamilyConfig;
use crate::gemma4::FeedForwardPolicy;
use eredu_nn::{AttentionArithmetic, Error};
use eredu_runtime::execution_topology::*;

/// Shared context/refinement stack; shared matrices retain one parameter identity.
pub(crate) fn text(config: &FamilyConfig) -> Result<TextExecutionTopology, Error> {
    let args = &config.text;
    let mut layers = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let attention_spec = config.attention(layer)?;
        let dense_spec = config.dense(layer)?;
        let attention = attention_spec
            .projections()
            .into_iter()
            .map(ProjectionTopology::from_spec)
            .collect::<Result<Vec<_>, _>>()?;
        let dense = FeedForwardTopology::Gated {
            intermediate_size: dense_spec.product.gate.output as u64,
            projections: dense_spec
                .product
                .projections()
                .into_iter()
                .map(ProjectionTopology::from_spec)
                .collect::<Result<Vec<_>, _>>()?,
        };
        let sparse = policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe;
        let feed_forward = if sparse {
            let declaration = config
                .routed(layer)?
                .ok_or_else(|| Error::backend("missing shared routed source"))?;
            FeedForwardTopology::Additive {
                branches: vec![
                    dense,
                    FeedForwardTopology::from_grouped_specs(
                        &declaration.routed.selector,
                        &declaration.routed.experts,
                    )?,
                ],
            }
        } else {
            dense
        };
        layers.push(TextLayerTopology {
            input_projections: Vec::new(),
            mixer: TokenMixerTopology::Attention {
                query_heads: attention_spec.geometry.query_heads() as u64,
                kv_heads: attention_spec.geometry.key_value_heads() as u64,
                key_width: attention_spec.geometry.key_dimensions() as u64,
                value_width: attention_spec.geometry.value_dimensions() as u64,
                input_scores: args.attention_arithmetic == AttentionArithmetic::InputScores,
                softcap: false,
                sinks: false,
                projections: attention,
                output_gate: false,
                query_key_normalization: true,
                rotary: true,
            },
            feed_forward,
            // Sparse expert input/output gains and router input transform are distinct.
            normalization_count: 4 + if sparse { 4 } else { 0 },
        });
    }
    let pinned = config.pinned();
    let conditioning = config.conditioning();
    Ok(TextExecutionTopology {
        iterative: Some(IterativeWorkspaceTopology {
            query_rows: u64::from(config.canvas_length.get()),
            intermediate_size: conditioning.product.gate.output as u64,
            conditioning: conditioning
                .product
                .projections()
                .into_iter()
                .map(ProjectionTopology::from_spec)
                .collect::<Result<Vec<_>, _>>()?,
            score_buffer_copies: 8,
        }),
        hidden_size: args.hidden_size as u64,
        vocabulary_size: args.vocab_size as u64,
        layers,
        output: ProjectionTopology::from_spec(&pinned.readout_projection())?,
        output_invocations: 1,
        output_softcap: args.final_logit_softcapping.is_some(),
        selected_parameter_promotion_bytes: None,
        selected_parameter_promotion_payloads: Default::default(),
        projection_input_normalizations: Default::default(),
        f32_rms_normalization_gains: Default::default(),
        projection_storage: Default::default(),
        missing: Vec::new(),
    })
}
