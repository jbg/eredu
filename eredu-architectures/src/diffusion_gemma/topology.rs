//! Cold invocation geometry uses the same projection declarations as construction.
use super::FamilyConfig;
use crate::gemma4::FeedForwardPolicy;
use eredu_nn::AttentionValueSource;
use eredu_nn::{AttentionArithmetic, Error};
use eredu_runtime::execution_topology::*;

/// Shared context/refinement stack; shared matrices retain one parameter identity.
pub(crate) fn text(config: &FamilyConfig) -> Result<TextExecutionTopology, Error> {
    let args = &config.text;
    let root = "model.decoder.layers";
    let projection = |name: String, input, output, bias| {
        ProjectionTopology::from_spec(&crate::gemma4::text::linear_spec(
            args, &name, input, output, bias,
        )?)
    };
    let mut layers = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let h = args.hidden_size;
        let d = policy.head_dim.get() as i32;
        let kv = policy.num_key_value_heads.get() as i32;
        let q = args.num_attention_heads;
        let i = policy.intermediate_size.get() as i32;
        let prefix = format!("{root}.{layer}");
        let mut attention = vec![projection(
            format!("{prefix}.self_attn.q_proj.weight"),
            h,
            q * d,
            args.attention_bias,
        )?];
        if policy.key_value.owns_state() {
            attention.push(projection(
                format!("{prefix}.self_attn.k_proj.weight"),
                h,
                kv * d,
                args.attention_bias,
            )?);
        }
        if policy.key_value.value() == Some(AttentionValueSource::Projected) {
            attention.push(projection(
                format!("{prefix}.self_attn.v_proj.weight"),
                h,
                kv * d,
                args.attention_bias,
            )?);
        }
        attention.push(projection(
            format!("{prefix}.self_attn.o_proj.weight"),
            q * d,
            h,
            args.attention_bias,
        )?);
        let dense = FeedForwardTopology::Gated {
            intermediate_size: i as u64,
            projections: vec![
                projection(format!("{prefix}.mlp.gate_proj.weight"), h, i, false)?,
                projection(format!("{prefix}.mlp.up_proj.weight"), h, i, false)?,
                projection(format!("{prefix}.mlp.down_proj.weight"), i, h, false)?,
            ],
        };
        let sparse = policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe;
        let feed_forward = if sparse {
            let selector = crate::gemma4::text::selector_spec(args, layer, root)?;
            let bank = crate::gemma4::text::expert_bank_spec_at(
                args,
                &format!("{prefix}.experts.switch_glu"),
            )?;
            FeedForwardTopology::Additive {
                branches: vec![
                    dense,
                    FeedForwardTopology::from_grouped_specs(&selector, &bank)?,
                ],
            }
        } else {
            dense
        };
        layers.push(TextLayerTopology {
            input_projections: Vec::new(),
            mixer: TokenMixerTopology::Attention {
                query_heads: q as u64,
                kv_heads: kv as u64,
                key_width: d as u64,
                value_width: d as u64,
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
    let output_name = if args.tie_word_embeddings {
        "model.decoder.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    Ok(TextExecutionTopology {
        iterative: Some(IterativeWorkspaceTopology {
            query_rows: u64::from(config.canvas_length.get()),
            intermediate_size: args
                .layer_policy(0)
                .ok_or_else(|| Error::backend("empty shared stack"))?
                .intermediate_size
                .get() as u64,
            conditioning: vec![
                projection(
                    "model.decoder.self_conditioning.gate_proj.weight".into(),
                    args.hidden_size,
                    args.layer_policy(0).unwrap().intermediate_size.get() as i32,
                    false,
                )?,
                projection(
                    "model.decoder.self_conditioning.up_proj.weight".into(),
                    args.hidden_size,
                    args.layer_policy(0).unwrap().intermediate_size.get() as i32,
                    false,
                )?,
                projection(
                    "model.decoder.self_conditioning.down_proj.weight".into(),
                    args.layer_policy(0).unwrap().intermediate_size.get() as i32,
                    args.hidden_size,
                    false,
                )?,
            ],
            score_buffer_copies: 8,
        }),
        hidden_size: args.hidden_size as u64,
        vocabulary_size: args.vocab_size as u64,
        layers,
        output: projection(output_name.into(), args.hidden_size, args.vocab_size, false)?,
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
