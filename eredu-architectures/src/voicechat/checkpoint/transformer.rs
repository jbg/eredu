//! Released embedded-transformer aliases, encoding eligibility and gain conversions.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::tts_transformer_specs::TransformerDeclaration;
use eredu_gguf::GgmlType;
use eredu_nn::{
    DeclareLogicalParameters, NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};
use std::collections::{BTreeMap, BTreeSet};
pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declarations: [&TransformerDeclaration; 2],
) -> Result<(), String> {
    for declaration in declarations {
        let declaration = declaration.spec();
        let encoded = declaration
            .parameters
            .blocks
            .iter()
            .flat_map(|b| {
                b.attention
                    .projections()
                    .into_iter()
                    .chain(b.feed_forward.iter().flat_map(|spec| spec.projections()))
                    .map(|s| &s.weight.id)
            })
            .collect::<BTreeSet<_>>();
        for p in declaration
            .logical_parameters()
            .map_err(|e| e.to_string())?
        {
            out.push(ReleasedTensor {
                component: Component::Tts,
                key: p.parameter.id.as_str().into(),
                shape: p.shape,
                affine_source_format: p.linear_format.cloned(),
                integer: false,
                quantized: encoded.contains(&p.parameter.id),
            });
        }
    }
    Ok(())
}
fn offset_gain(spec: &NormalizationConstructionSpec) -> Result<&ParameterSpec, String> {
    match &spec.scale {
        NormalizationScale::LearnedOffset {
            weight,
            offset: 1.0,
        } => Ok(weight),
        _ => Err("published split TTS gain requires a unit normalization offset".into()),
    }
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declarations: [&TransformerDeclaration; 2],
) -> Result<(), String> {
    for (declaration, causal) in declarations.into_iter().zip([true, false]) {
        let declaration = declaration.spec();
        let shapes = declaration
            .logical_parameters()
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|p| (&p.parameter.id, p.shape))
            .collect::<BTreeMap<_, _>>();
        let mut add =
            |source: String, slot: &ParameterSpec, encoding, transform| -> Result<(), String> {
                out.push(SplitTensor {
                    source,
                    target: slot.id.as_str().into(),
                    shape: shapes
                        .get(&slot.id)
                        .ok_or_else(|| format!("unresolved TTS source slot {}", slot.id))?
                        .clone(),
                    encoding,
                    transform,
                });
                Ok(())
            };
        for (layer, b) in declaration.parameters.blocks.iter().enumerate() {
            let root = if causal {
                format!("blk.{layer}")
            } else {
                format!("tts.subword.blk.{layer}")
            };
            let crate::decoder::topology::AttentionInputParameters::Split {
                query,
                key,
                value: Some(value),
            } = &b.attention.input
            else {
                return Err("split TTS requires independent Q/K/V projections".into());
            };
            if b.attention.gate.is_some() {
                return Err("split TTS has no attention output gate".into());
            }
            for (alias, spec) in [
                ("attn_q", query),
                ("attn_k", key),
                ("attn_v", value),
                ("attn_output", &b.attention.output),
            ] {
                if spec.bias.is_some() {
                    return Err("published split TTS projection has no bias slot".into());
                }
                add(
                    format!("{root}.{alias}.weight"),
                    &spec.weight,
                    GgmlType::Q4_0,
                    Transform::Identity,
                )?;
            }
            let Some(eredu_nn::GatedProductSpec::Split(product)) = &b.feed_forward else {
                return Err("split TTS requires independent gated projections".into());
            };
            for (alias, spec) in [
                ("ffn_gate", &product.gate),
                ("ffn_up", &product.up),
                ("ffn_down", &product.down),
            ] {
                if spec.bias.is_some() {
                    return Err(
                        "published split TTS feed-forward projection has no bias slot".into(),
                    );
                }
                add(
                    format!("{root}.{alias}.weight"),
                    &spec.weight,
                    GgmlType::Q4_0,
                    Transform::Identity,
                )?;
            }
            for (alias, norm) in [
                ("attn_norm", Some(&b.normalizations.input)),
                ("ffn_norm", Some(&b.normalizations.post_attention)),
                (
                    "post_attention_norm",
                    b.normalizations.attention_output.as_ref(),
                ),
                (
                    "post_ffw_norm",
                    b.normalizations.feed_forward_output.as_ref(),
                ),
                ("attn_q_norm", b.query_key[0].as_ref()),
                ("attn_k_norm", b.query_key[1].as_ref()),
            ] {
                if let Some(norm) = norm {
                    add(
                        format!("{root}.{alias}.weight"),
                        offset_gain(norm)?,
                        GgmlType::F32,
                        Transform::SubtractOne,
                    )?;
                }
            }
            if b.sinks.is_some() || b.terminal.is_some() {
                return Err("published split TTS has no sink or invocation-terminal slot".into());
            }
        }
        let alias = if causal {
            "output_norm.weight"
        } else {
            "tts.subword.output_norm.weight"
        };
        add(
            alias.into(),
            offset_gain(&declaration.parameters.normalization)?,
            GgmlType::F32,
            Transform::SubtractOne,
        )?;
    }
    Ok(())
}
