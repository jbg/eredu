//! Published RVQ feedback aliases and physical encodings over declared owners.
use super::{Component, ReleasedTensor, SplitTensor, Transform};
use crate::voicechat::tts_rvq_specs::RvqDeclaration;
use eredu_gguf::GgmlType;
use eredu_nn::DeclareLogicalParameters;
use std::collections::BTreeMap;

pub(super) fn released(
    out: &mut Vec<ReleasedTensor>,
    declaration: &RvqDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
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
            // Published complete variants encode the affine; the frozen bank
            // retains its ordinary floating dtype constraints.
            quantized: parameter.parameter.id == declaration.projection.embedding.weight.id,
        });
    }
    Ok(())
}
pub(super) fn split(
    out: &mut Vec<SplitTensor>,
    declaration: &RvqDeclaration,
) -> Result<(), String> {
    let declaration = declaration.spec();
    let shapes = declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| (&p.parameter.id, p.shape))
        .collect::<BTreeMap<_, _>>();
    for (alias, slot, encoding) in [
        ("codec.rvq_embs", &declaration.bank.table, GgmlType::F16),
        (
            "tts.embed_code.weight",
            &declaration.projection.embedding.weight,
            GgmlType::Q4_0,
        ),
    ] {
        out.push(SplitTensor {
            source: alias.into(),
            target: slot.id.as_str().to_owned(),
            shape: shapes
                .get(&slot.id)
                .ok_or_else(|| format!("unresolved RVQ storage slot {}", slot.id))?
                .clone(),
            encoding,
            transform: Transform::Identity,
        });
    }
    Ok(())
}
