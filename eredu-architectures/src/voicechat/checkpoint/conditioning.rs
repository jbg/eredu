//! Published conditioning storage aliases over typed character, ingress and fusion owners.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::tts_conditioning_specs::ConditioningDeclaration;
use eredu_gguf::GgmlType;
use eredu_nn::{DeclareLogicalParameters, NormalizationScale, ParameterSpec};
use std::collections::BTreeMap;

pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &ConditioningDeclaration,
) -> Result<(), String> {
    let d = declaration.spec();
    let c = &d.character;
    let i = &d.ingress;
    let f = &d.fusion;
    let mut encoded = vec![
        &c.tokens.weight.id,
        &c.projection.weight.id,
        &i.projection.parameter.id,
        &f.audio.weight.id,
        &f.text.weight.id,
    ];
    for flag in [&c.continuation, &c.special].into_iter().flatten() {
        encoded.push(&flag.weight.id);
    }
    for p in d.logical_parameters().map_err(|e| e.to_string())? {
        out.push(ReleasedTensor {
            component: Component::Tts,
            key: p.parameter.id.as_str().to_owned(),
            shape: p.shape,
            affine_source_format: p.linear_format.cloned(),
            integer: false,
            quantized: encoded.contains(&&p.parameter.id),
        });
    }
    Ok(())
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declaration: &ConditioningDeclaration,
) -> Result<(), String> {
    let d = declaration.spec();
    let shapes = d
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (p.parameter.id.as_str().to_owned(), p.shape))
        .collect::<BTreeMap<_, _>>();
    let mut add = |alias: &str,
                   slot: &ParameterSpec,
                   encoding,
                   transform,
                   scalar: bool|
     -> Result<(), String> {
        let mut shape = shapes
            .get(slot.id.as_str())
            .ok_or_else(|| format!("unresolved conditioning storage slot {}", slot.id))?
            .clone();
        if scalar {
            if !shape.is_empty() {
                return Err("split conditioning scalar has incompatible geometry".into());
            }
            shape.push(1);
        }
        out.push(SplitTensor {
            source: alias.into(),
            target: slot.id.as_str().into(),
            shape,
            encoding,
            transform,
        });
        Ok(())
    };
    let c = &d.character;
    let i = &d.ingress;
    let f = &d.fusion;
    add(
        "tts.audio_prompt_latents",
        &i.aria.parameter,
        GgmlType::F32,
        Transform::Identity,
        false,
    )?;
    add(
        "tts.audio_prompt_proj.weight",
        &i.projection.parameter,
        GgmlType::Q4_0,
        Transform::Identity,
        false,
    )?;
    add(
        "tts.bos_emb",
        &i.bos.parameter,
        GgmlType::F32,
        Transform::Identity,
        false,
    )?;
    add(
        "tts.null_emb",
        &i.null.parameter,
        GgmlType::F32,
        Transform::Identity,
        false,
    )?;
    add(
        "tts.subword.embed_tokens.weight",
        &c.tokens.weight,
        GgmlType::Q4_0,
        Transform::Identity,
        false,
    )?;
    add(
        "tts.subword.proj.weight",
        &c.projection.weight,
        GgmlType::Q4_0,
        Transform::Identity,
        false,
    )?;
    if let Some(flag) = &c.continuation {
        add(
            "tts.subword.cont_emb.weight",
            &flag.weight,
            GgmlType::Q4_0,
            Transform::Identity,
            false,
        )?;
    }
    if let Some(flag) = &c.special {
        add(
            "tts.subword.special_emb.weight",
            &flag.weight,
            GgmlType::Q4_0,
            Transform::Identity,
            false,
        )?;
    }
    for (alias, linear) in [("audio_proj", &f.audio), ("text_proj", &f.text)] {
        add(
            &format!("tts.fusion.{alias}.weight"),
            &linear.weight,
            GgmlType::Q4_0,
            Transform::Identity,
            false,
        )?;
        add(
            &format!("tts.fusion.{alias}.bias"),
            linear
                .bias
                .as_ref()
                .ok_or("split fusion requires affine bias")?,
            GgmlType::F32,
            Transform::Identity,
            false,
        )?;
    }
    add(
        "tts.fusion.gate",
        &f.gate.parameter,
        GgmlType::F32,
        Transform::Identity,
        false,
    )?;
    let NormalizationScale::LearnedOffset {
        weight,
        offset: 1.0,
    } = &f.norm.scale
    else {
        return Err("split fusion requires unit-offset RMS gain".into());
    };
    add(
        "tts.fusion.norm.weight",
        weight,
        GgmlType::F32,
        Transform::SubtractOne,
        false,
    )?;
    add(
        "tts.fusion.residual_scale",
        &f.residual_scale.parameter,
        GgmlType::F32,
        Transform::Identity,
        true,
    )?;
    Ok(())
}
