//! Published Conformer source encodings and singleton-axis recipes over typed owners.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::perception_specs::PerceptionDeclaration;
use eredu_gguf::GgmlType;
use eredu_nn::{DeclareLogicalParameters, LayerNormalizationConstructionSpec, ParameterSpec};
use std::collections::{BTreeMap, BTreeSet};
pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &PerceptionDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    for block in &declaration.blocks {
        // Complete published GGUF variants encode these exact affine matrices
        // and query-head biases. Neural roles do not establish encoding support.
        let a = &block.attention;
        let mut encoded = block
            .feed_forward
            .iter()
            .flat_map(|ff| [&ff.network.up.weight.id, &ff.network.down.weight.id])
            .collect::<BTreeSet<_>>();
        encoded.extend([
            &a.projection.query.weight.id,
            &a.projection.key.weight.id,
            &a.projection.value.weight.id,
            &a.position.weight.id,
            &a.projection.output.weight.id,
            &a.content_bias.weight.id,
            &a.position_bias.weight.id,
        ]);
        for parameter in block.logical_parameters().map_err(|e| e.to_string())? {
            out.push(ReleasedTensor {
                component: Component::Perception,
                key: parameter.parameter.id.as_str().to_owned(),
                shape: parameter.shape,
                affine_source_format: parameter.linear_format.cloned(),
                integer: false,
                quantized: encoded.contains(&parameter.parameter.id),
            });
        }
    }
    Ok(())
}
fn norm_slots(
    spec: &LayerNormalizationConstructionSpec,
) -> Result<(&ParameterSpec, &ParameterSpec), String> {
    Ok((
        spec.weight
            .as_ref()
            .ok_or("released Conformer normalization requires a gain")?,
        spec.bias
            .as_ref()
            .ok_or("released Conformer normalization requires an offset")?,
    ))
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declaration: &PerceptionDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    for block in &declaration.blocks {
        let shapes = block
            .logical_parameters()
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|p| (&p.parameter.id, p.shape))
            .collect::<BTreeMap<_, _>>();
        let index = block.index;
        let mut add = |alias: &str,
                       slot: &ParameterSpec,
                       encoding: GgmlType,
                       squeeze: Option<usize>|
         -> Result<(), String> {
            let mut shape = shapes
                .get(&slot.id)
                .ok_or_else(|| format!("unresolved split Conformer parameter {}", slot.id))?
                .clone();
            if let Some(axis) = squeeze {
                if shape.get(axis) != Some(&1) {
                    return Err(format!(
                        "split Conformer parameter {} requires a singleton storage axis",
                        slot.id
                    ));
                }
                shape.remove(axis);
            }
            out.push(SplitTensor {
                source: format!("a.blk.{index}.{alias}"),
                target: slot.id.as_str().to_owned(),
                shape,
                encoding,
                transform: Transform::Identity,
            });
            Ok(())
        };
        let a = &block.attention;
        for (alias, slot) in [
            ("attn_q.weight", &a.projection.query.weight),
            ("attn_k.weight", &a.projection.key.weight),
            ("attn_v.weight", &a.projection.value.weight),
            ("attn_out.weight", &a.projection.output.weight),
            ("linear_pos.weight", &a.position.weight),
        ] {
            add(alias, slot, GgmlType::Q4_0, None)?;
        }
        for (suffix, ff) in [("", &block.feed_forward[0]), ("_1", &block.feed_forward[1])] {
            add(
                &format!("ffn_up{suffix}.weight"),
                &ff.network.up.weight,
                GgmlType::Q4_0,
                None,
            )?;
            add(
                &format!("ffn_down{suffix}.weight"),
                &ff.network.down.weight,
                GgmlType::Q4_0,
                None,
            )?;
        }
        for (alias, norm) in [
            "ffn_norm",
            "ln1",
            "norm_conv",
            "conv_norm",
            "ffn_norm_1",
            "ln2",
        ]
        .into_iter()
        .zip(&block.norms)
        {
            let (weight, bias) = norm_slots(norm)?;
            add(&format!("{alias}.weight"), weight, GgmlType::F32, None)?;
            add(&format!("{alias}.bias"), bias, GgmlType::F32, None)?;
        }
        add("pos_bias_u", &a.content_bias.weight, GgmlType::F32, None)?;
        add("pos_bias_v", &a.position_bias.weight, GgmlType::F32, None)?;
        add(
            "conv_pw1.weight",
            &block.pointwise_input.weight,
            GgmlType::F16,
            Some(2),
        )?;
        add(
            "conv_dw.weight",
            &block.depthwise.weight,
            GgmlType::F32,
            Some(1),
        )?;
        add(
            "conv_pw2.weight",
            &block.pointwise_output.weight,
            GgmlType::F16,
            Some(2),
        )?;
    }
    Ok(())
}
