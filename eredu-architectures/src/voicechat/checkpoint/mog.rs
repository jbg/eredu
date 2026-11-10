//! Released MoG source encodings and gain transforms referencing typed parameter slots.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::tts_mog_specs::MogDeclaration;
use eredu_gguf::GgmlType;
use eredu_nn::{
    DeclareLogicalParameters, NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &MogDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    // These exact slots are encoded by the complete GGUF variants. Their neural
    // roles alone establish neither a physical source format nor backend support.
    let mut encoded: BTreeSet<_> = declaration
        .blocks
        .iter()
        .flat_map(|block| block.product.projections().map(|spec| &spec.weight))
        .map(|p| &p.id)
        .collect();
    let p = &declaration.prediction;
    let c = &declaration.components;
    encoded.extend([
        &p.logits.weight.id,
        &c.residual.weight.id,
        &c.logs.weight.id,
        &c.means.weight.id,
        &c.low.weight.id,
    ]);
    for parameter in declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        out.push(ReleasedTensor {
            component: Component::Tts,
            key: parameter.parameter.id.as_str().to_owned(),
            shape: parameter.shape,
            affine_source_format: parameter.linear_format.cloned(),
            integer: false,
            quantized: encoded.contains(&parameter.parameter.id),
        });
    }
    Ok(())
}
fn gain(spec: &NormalizationConstructionSpec) -> Result<&ParameterSpec, String> {
    match &spec.scale {
        NormalizationScale::LearnedOffset { weight, offset } if *offset == 1.0 => Ok(weight),
        _ => Err("split MoG gain requires the declared unit offset".into()),
    }
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declaration: &MogDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    let shapes = declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (&p.parameter.id, p.shape))
        .collect::<BTreeMap<_, _>>();
    let mut add = |source: String,
                   slot: &ParameterSpec,
                   encoding: GgmlType,
                   transform: Transform|
     -> Result<(), String> {
        let shape = shapes
            .get(&slot.id)
            .ok_or_else(|| format!("unresolved split MoG parameter {}", slot.id))?
            .clone();
        out.push(SplitTensor {
            source,
            target: slot.id.as_str().to_owned(),
            shape,
            encoding,
            transform,
        });
        Ok(())
    };
    for (index, block) in declaration.blocks.iter().enumerate() {
        for (alias, slot) in [
            ("ffn_down", &block.product.down.weight),
            ("ffn_gate", &block.product.gate.weight),
            ("ffn_up", &block.product.up.weight),
        ] {
            add(
                format!("tts.mog.blk.{index}.{alias}.weight"),
                slot,
                GgmlType::Q4_0,
                Transform::Identity,
            )?;
        }
        for (alias, slot) in [
            ("post_norm", gain(&block.post)?),
            ("pre_norm", gain(&block.pre)?),
        ] {
            add(
                format!("tts.mog.blk.{index}.{alias}.weight"),
                slot,
                GgmlType::F32,
                Transform::SubtractOne,
            )?;
        }
    }
    let p = &declaration.prediction;
    let c = &declaration.components;
    for (alias, slot) in [
        ("else", &c.residual.weight),
        ("logits", &p.logits.weight),
        ("logs", &c.logs.weight),
        ("mus", &c.means.weight),
    ] {
        add(
            format!("tts.mog.{alias}.weight"),
            slot,
            GgmlType::Q4_0,
            Transform::Identity,
        )?;
    }
    add(
        "tts.mog.low_mat".into(),
        &c.low.weight,
        GgmlType::F16,
        Transform::Identity,
    )?;
    add(
        "tts.mog.norm.weight".into(),
        gain(&p.norm)?,
        GgmlType::F32,
        Transform::SubtractOne,
    )?;
    Ok(())
}
